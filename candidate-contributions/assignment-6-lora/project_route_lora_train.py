import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    answer_ids = tokenizer(
        example["routing_destination"] + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]
    if not prompt_ids or not answer_ids or len(prompt_ids) + len(answer_ids) > max_length:
        raise ValueError("Token sequence violates the configured context-length boundary")
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {
        "course_record_id": example["course_record_id"],
        "prompt_ids": prompt_ids,
        "input_ids": input_ids,
        "labels": labels,
        "answer_ids": answer_ids,
        "reference": example["routing_destination"],
    }


def score_routes(model, tokenizer, prompt_ids, max_length, device):
    scores = {}
    for route in ROUTES:
        answer_ids = tokenizer(route + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(answer_ids) > max_length:
            raise ValueError("Candidate response exceeds configured context length")
        sequence = prompt_ids + answer_ids
        inputs = torch.tensor([sequence], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits[0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_scores = [
            float(log_probs[len(prompt_ids) + offset - 1, token_id].item())
            for offset, token_id in enumerate(answer_ids)
        ]
        scores[route] = sum(token_scores) / len(token_scores)
    return scores


def classify(model, tokenizer, examples, max_length, device):
    predictions = []
    for example in examples:
        prompt = make_prompt(example["consumer_complaint_narrative"])
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length - 16,
        )["input_ids"]
        scores = score_routes(model, tokenizer, prompt_ids, max_length, device)
        predictions.append(max(ROUTES, key=lambda route: scores[route]))
    return predictions


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


def load_base(model_path, dtype):
    return AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to("cuda")


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
    train_tokens = [tokenized_example(tokenizer, row, config["max_length"]) for row in train_examples]
    val_references = [row["routing_destination"] for row in validation_examples]

    base = load_base(model_path, dtype)
    base.config.use_cache = False
    base_predictions = classify(base, tokenizer, validation_examples, config["max_length"], device)
    base_metrics = metrics(val_references, base_predictions)
    model = get_peft_model(
        base,
        LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=config["lora"]["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    epoch_train_losses = []
    started = time.time()
    for epoch in range(config["epochs"]):
        model.train()
        order = list(range(len(train_tokens)))
        random.Random(seed + epoch).shuffle(order)
        losses = []
        for index in order:
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
        epoch_train_losses.append(sum(losses) / len(losses))
        print(f"EPOCH {epoch + 1}/{config['epochs']} MEAN_TRAIN_LOSS {epoch_train_losses[-1]:.6f}", flush=True)

    adapter_path = Path(config["adapter_path"])
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    adapted_predictions = classify(model, tokenizer, validation_examples, config["max_length"], device)
    adapted_metrics = metrics(val_references, adapted_predictions)

    reloaded_base = load_base(model_path, dtype)
    reloaded = PeftModel.from_pretrained(reloaded_base, str(adapter_path), local_files_only=True)
    reloaded_predictions = classify(reloaded, tokenizer, validation_examples, config["max_length"], device)
    reloaded_metrics = metrics(val_references, reloaded_predictions)
    result = {
        "experiment": "employee_assisted_credit_card_billing_dispute_route_suggestion",
        "slurm_host": platform.node(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__,
        "model_path": str(model_path),
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
        "base_internal_validation": base_metrics,
        "adapted_internal_validation": adapted_metrics,
        "reloaded_adapter_internal_validation": reloaded_metrics,
        "adapter_reload_predictions_identical": adapted_predictions == reloaded_predictions,
        "adapter_path": str(adapter_path),
        "elapsed_seconds_including_scoring_and_reload": time.time() - started,
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
        for row, base_route, adapter_route in zip(validation_examples, base_predictions, adapted_predictions):
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
