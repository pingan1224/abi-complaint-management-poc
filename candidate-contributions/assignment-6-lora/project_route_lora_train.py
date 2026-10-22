import argparse
import csv
import gc
import json
import platform
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig


ROUTES = ("Card Billing Disputes", "Card Fraud & Security")


def read_examples(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        examples = list(csv.DictReader(source))
    if not examples:
        raise ValueError(f"No examples in {path}")
    for example in examples:
        if not example["consumer_complaint_narrative"].strip():
            raise ValueError("Blank complaint narratives are not allowed")
        if example["routing_destination"] not in ROUTES:
            raise ValueError(f"Unexpected routing label: {example['routing_destination']}")
    return examples


def make_prompt(narrative):
    return (
        "Task: Suggest one employee-facing destination for this web-submitted credit-card billing dispute. "
        "Choose exactly one: Card Billing Disputes or Card Fraud & Security. "
        "This is a suggestion only; an employee must confirm, override, reject, or defer it. "
        "Do not take action or address the customer.\nComplaint narrative:\n"
        + narrative.strip()
        + "\nSuggested destination: "
    )


def tokenized_example(tokenizer, example, max_length):
    prompt = make_prompt(example["consumer_complaint_narrative"])
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length - 16,
    )["input_ids"]
    input_ids = tokenizer(
        prompt + example["routing_destination"] + tokenizer.eos_token,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    shared_prefix_length = 0
    for prompt_token, full_token in zip(prompt_ids, input_ids):
        if prompt_token != full_token:
            break
        shared_prefix_length += 1
    answer_ids = input_ids[shared_prefix_length:]
    if not prompt_ids or not answer_ids or shared_prefix_length == len(input_ids):
        raise ValueError("Token sequence violates the configured context-length boundary")
    labels = [-100] * shared_prefix_length + answer_ids
    return {
        "course_record_id": example["course_record_id"],
        "prompt_ids": input_ids[:shared_prefix_length],
        "input_ids": input_ids,
        "labels": labels,
        "answer_ids": answer_ids,
        "prompt_boundary_token_count": len(prompt_ids) - shared_prefix_length,
        "reference": example["routing_destination"],
    }


def score_routes(model, tokenizer, prompt, max_length, device):
    scores = {}
    for route in ROUTES:
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length - 16,
        )["input_ids"]
        sequence = tokenizer(
            prompt + route + tokenizer.eos_token,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        shared_prefix_length = 0
        for prompt_token, full_token in zip(prompt_ids, sequence):
            if prompt_token != full_token:
                break
            shared_prefix_length += 1
        if shared_prefix_length == len(sequence):
            raise ValueError("Candidate response has no continuation tokens")
        inputs = torch.tensor([sequence], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits[0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_scores = [
            float(log_probs[shared_prefix_length + offset - 1, token_id].item())
            for offset, token_id in enumerate(sequence[shared_prefix_length:])
        ]
        scores[route] = {
            "sequence_log_likelihood": sum(token_scores),
            "mean_token_log_probability": sum(token_scores) / len(token_scores),
        }
    return scores


def classify(model, tokenizer, examples, max_length, device):
    predictions = {"sequence_log_likelihood": [], "mean_token_log_probability": []}
    billing_minus_fraud_margins = []
    for example in examples:
        prompt = make_prompt(example["consumer_complaint_narrative"])
        scores = score_routes(model, tokenizer, prompt, max_length, device)
        for method in predictions:
            predictions[method].append(max(ROUTES, key=lambda route: scores[route][method]))
        billing_minus_fraud_margins.append(
            scores["Card Billing Disputes"]["sequence_log_likelihood"]
            - scores["Card Fraud & Security"]["sequence_log_likelihood"]
        )
    return predictions, billing_minus_fraud_margins


def summarize_margins(references, margins):
    grouped = {}
    for route in ROUTES:
        values = [margin for reference, margin in zip(references, margins) if reference == route]
        grouped[route] = {
            "count": len(values),
            "mean_billing_minus_fraud_log_likelihood": sum(values) / len(values),
        }
    return {
        "count": len(margins),
        "mean_billing_minus_fraud_log_likelihood": sum(margins) / len(margins),
        "mean_absolute_margin": sum(abs(margin) for margin in margins) / len(margins),
        "mean_margin_by_reference_route": grouped,
    }


def metrics(references, predictions):
    matrix = {reference: {prediction: 0 for prediction in ROUTES} for reference in ROUTES}
    for reference, prediction in zip(references, predictions):
        matrix[reference][prediction] += 1
    per_route = {}
    for route in ROUTES:
        true_positive = matrix[route][route]
        false_positive = sum(matrix[other][route] for other in ROUTES if other != route)
        false_negative = sum(matrix[route][other] for other in ROUTES if other != route)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        per_route[route] = {
            "support": sum(matrix[route].values()),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    accuracy = sum(matrix[label][label] for label in ROUTES) / len(references)
    return {"accuracy": accuracy, "confusion_matrix_reference_rows_predicted_columns": matrix, "per_route": per_route}


def load_base(model_path, dtype, dequantize_compressed_tensors=False):
    load_kwargs = {
        "local_files_only": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": {"": "cuda"},
    }
    if dequantize_compressed_tensors:
        load_kwargs["quantization_config"] = CompressedTensorsConfig(dequantize=True)
    return AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="project_route_lora_config.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run through Slurm on the DGX Spark GPU node")
    model_path = Path(config["model_path"])
    if not list(model_path.glob("*.safetensors")):
        raise FileNotFoundError(f"Approved checkpoint weights not found at {model_path}")

    train_examples = read_examples(config["train_csv"])
    validation_examples = read_examples(config["internal_validation_csv"])
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer EOS token is required for causal response scoring")
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    train_tokens = [tokenized_example(tokenizer, row, config["max_length"]) for row in train_examples]
    boundary_adjusted_examples = sum(item["prompt_boundary_token_count"] > 0 for item in train_tokens)
    val_references = [row["routing_destination"] for row in validation_examples]

    dequantize_compressed_tensors = bool(config.get("dequantize_compressed_tensors", False))
    base = load_base(model_path, dtype, dequantize_compressed_tensors)
    base.config.use_cache = False
    base_predictions, base_margins = classify(base, tokenizer, validation_examples, config["max_length"], device)
    base_metrics = metrics(val_references, base_predictions["sequence_log_likelihood"])
    base_mean_score_metrics = metrics(val_references, base_predictions["mean_token_log_probability"])
    model = get_peft_model(
        base,
        LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=config["lora"]["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        ),
    )
    model.enable_input_require_grads()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    epoch_train_losses = []
    step_losses = []
    for epoch in range(config["epochs"]):
        model.train()
        order = list(range(len(train_tokens)))
        random.Random(seed + epoch).shuffle(order)
        losses = []
        for step, index in enumerate(order, start=1):
            item = train_tokens[index]
            input_ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
            labels = torch.tensor([item["labels"]], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                config["max_grad_norm"],
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step_losses.append(losses[-1])
            if step % 10 == 0 or step == len(order):
                print(f"TRAIN_PROGRESS epoch={epoch + 1} step={step}/{len(order)} loss={losses[-1]:.6f}", flush=True)
        epoch_train_losses.append(sum(losses) / len(losses))
        print(f"EPOCH {epoch + 1}/{config['epochs']} MEAN_TRAIN_LOSS {epoch_train_losses[-1]:.6f}", flush=True)

    adapter_path = Path(config["adapter_path"])
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    adapted_predictions, adapted_margins = classify(model, tokenizer, validation_examples, config["max_length"], device)
    adapted_metrics = metrics(val_references, adapted_predictions["sequence_log_likelihood"])
    adapted_mean_score_metrics = metrics(val_references, adapted_predictions["mean_token_log_probability"])

    del optimizer, model, base
    gc.collect()
    torch.cuda.empty_cache()
    reloaded_base = load_base(model_path, dtype, dequantize_compressed_tensors)
    reloaded = PeftModel.from_pretrained(reloaded_base, str(adapter_path), local_files_only=True)
    reloaded_predictions, reloaded_margins = classify(reloaded, tokenizer, validation_examples, config["max_length"], device)
    reloaded_metrics = metrics(val_references, reloaded_predictions["sequence_log_likelihood"])
    reloaded_mean_score_metrics = metrics(val_references, reloaded_predictions["mean_token_log_probability"])
    peak_cuda_memory_bytes = torch.cuda.max_memory_allocated()
    result = {
        "experiment": "employee_assisted_credit_card_billing_dispute_route_suggestion",
        "slurm_host": platform.node(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__,
        "model_path": str(model_path),
        "base_weight_training_precision": "BF16 dequantized from compressed-tensors FP8" if dequantize_compressed_tensors else str(dtype),
        "seed": seed,
        "scope_filter_for_data_selection_only": config["scope_filter"],
        "model_input": "consumer_complaint_narrative only",
        "target": "routing_destination; Clear labels only",
        "train_examples": len(train_examples),
        "internal_validation_examples": len(validation_examples),
        "official_heldout_used": False,
        "epochs": config["epochs"],
        "optimizer": "AdamW",
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "max_length": config["max_length"],
        "lora": config["lora"],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "mean_epoch_training_loss": epoch_train_losses,
        "training_step_loss_summary": {
            "step_count": len(step_losses),
            "first_50_steps_mean": sum(step_losses[:50]) / min(50, len(step_losses)),
            "last_50_steps_mean": sum(step_losses[-50:]) / min(50, len(step_losses)),
        },
        "examples_with_prompt_answer_token_boundary_merge": boundary_adjusted_examples,
        "base_internal_validation": base_metrics,
        "base_validation_sequence_log_likelihood_margin": summarize_margins(val_references, base_margins),
        "base_mean_token_score_sensitivity": base_mean_score_metrics,
        "adapted_internal_validation": adapted_metrics,
        "adapted_validation_sequence_log_likelihood_margin": summarize_margins(val_references, adapted_margins),
        "adapted_minus_base_margin_shift": {
            "count": len(base_margins),
            "mean_shift": sum(adapted - base for base, adapted in zip(base_margins, adapted_margins)) / len(base_margins),
            "mean_absolute_shift": sum(abs(adapted - base) for base, adapted in zip(base_margins, adapted_margins)) / len(base_margins),
            "sign_flip_count": sum((base > 0) != (adapted > 0) for base, adapted in zip(base_margins, adapted_margins)),
        },
        "adapted_mean_token_score_sensitivity": adapted_mean_score_metrics,
        "reloaded_adapter_internal_validation": reloaded_metrics,
        "reloaded_adapter_validation_sequence_log_likelihood_margin": summarize_margins(val_references, reloaded_margins),
        "reloaded_adapter_mean_token_score_sensitivity": reloaded_mean_score_metrics,
        "adapter_reload_predictions_identical": adapted_predictions == reloaded_predictions,
        "adapter_path": str(adapter_path),
        "elapsed_seconds_including_model_load_baseline_training_scoring_and_reload": time.time() - started,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        "limitations": [
            "Internal validation is a Development-only diagnostic, not the official Held-out Evaluation split.",
            "Course routing references are synthetic instructional labels, not bank policy or adjudicated ground truth.",
            "This experiment suggests a route only; human confirmation remains mandatory.",
        ],
    }
    result_path = Path(config["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    prediction_path = Path(config["prediction_path"])
    with prediction_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=["course_record_id", "reference_route", "base_route", "adapter_route"],
        )
        writer.writeheader()
        for row, base_route, adapter_route in zip(
            validation_examples,
            base_predictions["sequence_log_likelihood"],
            adapted_predictions["sequence_log_likelihood"],
        ):
            writer.writerow(
                {
                    "course_record_id": row["course_record_id"],
                    "reference_route": row["routing_destination"],
                    "base_route": base_route,
                    "adapter_route": adapter_route,
                }
            )
    print("RESULT_JSON", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
