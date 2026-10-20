"""Tiny, synthetic LoRA checkpoint experiment for the course DGX Spark.

Only `preflight` uses the Python standard library. `train` and `evaluate`
require the course-approved DGX GPU, local base checkpoint, and installed
PyTorch/Transformers/PEFT/Accelerate packages. No course complaint CSV is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM = (
    "You are testing a prototype card-complaint intake decision. "
    "Reply with exactly REVIEW when the narrative suggests fraud, an active security "
    "concern, or insufficient information for a safe decision. Otherwise reply "
    "with exactly ROUTINE. This is a synthetic classroom exercise, not a final bank decision."
)
ALLOWED = {"REVIEW", "ROUTINE"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "intended_base_model", "learning_focus", "task", "seed", "max_length",
        "max_steps", "learning_rate", "per_device_train_batch_size",
        "gradient_accumulation_steps", "lora_r", "lora_alpha",
        "lora_dropout", "target_modules", "max_new_tokens",
    }
    if set(config) != expected:
        raise ValueError(f"Configuration keys differ: {sorted(set(config) ^ expected)}")
    for key in ("seed", "max_length", "max_steps", "per_device_train_batch_size",
                "gradient_accumulation_steps", "lora_r", "lora_alpha", "max_new_tokens"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < config["learning_rate"] < 1:
        raise ValueError("learning_rate is outside (0, 1)")
    if not 0 <= config["lora_dropout"] < 1:
        raise ValueError("lora_dropout is outside [0, 1)")
    if config["target_modules"] != ["q_proj", "v_proj"]:
        raise ValueError("This controlled experiment targets q_proj and v_proj only")
    return config


def load_examples(path: Path) -> list[dict]:
    examples = []
    ids, narratives = set(), set()
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"id", "split", "narrative", "decision"}:
                raise ValueError(f"line {line_no}: unexpected or missing fields")
            if not re.fullmatch(r"SYN-[TE][0-9]{2}", row["id"]):
                raise ValueError(f"line {line_no}: only synthetic IDs are allowed")
            if row["id"] in ids:
                raise ValueError(f"line {line_no}: duplicate ID")
            if row["split"] not in {"train", "eval"}:
                raise ValueError(f"line {line_no}: split must be train or eval")
            if row["decision"] not in ALLOWED:
                raise ValueError(f"line {line_no}: invalid decision")
            narrative = row["narrative"].strip()
            if len(narrative) < 25:
                raise ValueError(f"line {line_no}: narrative too short")
            normalized = " ".join(narrative.lower().split())
            if normalized in narratives:
                raise ValueError(f"line {line_no}: duplicate narrative across splits")
            ids.add(row["id"])
            narratives.add(normalized)
            examples.append(row)
    counts = {split: sum(row["split"] == split for row in examples)
              for split in ("train", "eval")}
    if min(counts.values()) < 2:
        raise ValueError("Need at least two examples in each split")
    return examples


def preflight(config_path: Path, data_path: Path) -> tuple[dict, list[dict], dict]:
    config = load_config(config_path)
    examples = load_examples(data_path)
    summary = {
        "status": "local_format_check_only",
        "config_sha256": digest(config_path),
        "examples_sha256": digest(data_path),
        "train_count": sum(row["split"] == "train" for row in examples),
        "eval_count": sum(row["split"] == "eval" for row in examples),
        "synthetic_only": True,
        "heldout_course_file_opened": False,
    }
    return config, examples, summary


def check_local_32b_model(model_dir: Path) -> str:
    """Reject an accidental 8B checkpoint before allocating GPU memory."""
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError("Local model config.json is missing")
    model_config = json.loads(path.read_text(encoding="utf-8"))
    expected = {"model_type": "qwen2", "hidden_size": 5120, "num_hidden_layers": 64}
    mismatches = {key: (model_config.get(key), wanted) for key, wanted in expected.items()
                  if model_config.get(key) != wanted}
    if mismatches:
        raise ValueError(f"Local checkpoint is not the expected Qwen 2.5 32B architecture: {mismatches}")
    return digest(path)


def train(args: argparse.Namespace, config: dict, examples: list[dict], summary: dict) -> None:
    import torch
    import transformers
    import peft
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with bfloat16 support is required for this LoRA run")
    model_dir = Path(args.model).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError("Provide the administrator-approved local HF model directory")
    model_config_sha256 = check_local_32b_model(model_dir)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory so evidence is not overwritten")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0},
    )
    model.config.use_cache = False
    lora = LoraConfig(
        r=config["lora_r"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"], bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()
    if not 0 < trainable < total:
        raise RuntimeError("Adapter parameter check failed")

    class Samples(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.items = [encode(row) for row in rows]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    def encode(row):
        prefix = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["narrative"]},
        ]
        prompt_ids = tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=True)
        full_ids = tokenizer.apply_chat_template(
            prefix + [{"role": "assistant", "content": row["decision"]}],
            tokenize=True, add_generation_prompt=False,
        )
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Chat-template prefix mismatch for {row['id']}")
        if len(full_ids) > config["max_length"]:
            raise ValueError(f"Example {row['id']} is too long; do not silently truncate")
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        if all(label == -100 for label in labels):
            raise ValueError(f"Example {row['id']} has no supervised answer tokens")
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    def collate(batch):
        length = max(len(item["input_ids"]) for item in batch)
        result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for item in batch:
            pad = length - len(item["input_ids"])
            result["input_ids"].append(item["input_ids"] + [tokenizer.pad_token_id] * pad)
            result["attention_mask"].append(item["attention_mask"] + [0] * pad)
            result["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(values, dtype=torch.long) for key, values in result.items()}

    training = TrainingArguments(
        output_dir=str(output / "trainer"), max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], bf16=True, gradient_checkpointing=True,
        save_strategy="no", logging_steps=1, report_to="none", seed=config["seed"],
        remove_unused_columns=False, optim="adamw_torch",
    )
    trainer = Trainer(
        model=model, args=training,
        train_dataset=Samples([row for row in examples if row["split"] == "train"]),
        data_collator=collate,
    )
    start = time.monotonic()
    result = trainer.train()
    elapsed = round(time.monotonic() - start, 2)
    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)

    manifest = {
        **summary,
        "status": "dgx_training_completed",
        "utc_completed": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "model_directory": str(model_dir),
        "base_model_config_sha256": model_config_sha256,
        "model_config_name": model.config.to_dict().get("_name_or_path"),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "training_loss": result.training_loss,
        "train_runtime_seconds": elapsed,
        "adapter_files": {
            p.name: digest(p) for p in adapter_dir.iterdir() if p.is_file()
        },
        "effective_config": config,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "output": str(output),
                      "training_loss": result.training_loss}, indent=2))


def evaluate(args: argparse.Namespace, config: dict, examples: list[dict], summary: dict) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(args.model).expanduser().resolve()
    model_config_sha256 = check_local_32b_model(model_dir)
    output = Path(args.output).expanduser().resolve()
    adapter_dir = output / "adapter"
    if not (adapter_dir / "adapter_config.json").is_file() or not any(
        adapter_dir.glob("adapter_model.*")
    ):
        raise FileNotFoundError("Saved adapter configuration and weights are both required")
    if not (output / "run_manifest.json").is_file():
        raise FileNotFoundError("Training manifest is missing")
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["examples_sha256"] != summary["examples_sha256"]:
        raise ValueError("Example file changed after training")
    if manifest["config_sha256"] != summary["config_sha256"]:
        raise ValueError("Experiment config changed after training")
    if manifest["model_directory"] != str(model_dir):
        raise ValueError("Requested base model directory differs from training")
    if manifest["base_model_config_sha256"] != model_config_sha256:
        raise ValueError("Base model config changed after training")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model.eval()

    def generate(row):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["narrative"]},
        ]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to("cuda")
        with torch.inference_mode():
            ids = model.generate(
                **encoded, max_new_tokens=config["max_new_tokens"], do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(ids[0, encoded["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    rows = []
    for row in examples:
        if row["split"] != "eval":
            continue
        with model.disable_adapter():
            base_text = generate(row)
        adapted_text = generate(row)
        rows.append({
            "id": row["id"], "reference": row["decision"],
            "base_output": base_text, "adapted_output": adapted_text,
            "base_exact_match": base_text == row["decision"],
            "adapted_exact_match": adapted_text == row["decision"],
        })
    evidence = {
        "status": "dgx_adapter_reloaded_and_compared",
        "utc_completed": datetime.now(timezone.utc).isoformat(),
        "comparison_cases": len(rows),
        "base_exact_matches": sum(r["base_exact_match"] for r in rows),
        "adapted_exact_matches": sum(r["adapted_exact_match"] for r in rows),
        "synthetic_only": True,
        "cases": rows,
    }
    (output / "comparison.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "train", "evaluate"))
    parser.add_argument("--config", type=Path, default=ROOT / "experiment.json")
    parser.add_argument("--examples", type=Path, default=ROOT / "synthetic_examples.jsonl")
    parser.add_argument("--model", help="Administrator-approved local 32B HF checkpoint directory")
    parser.add_argument("--output", help="New run directory outside the Git repository")
    args = parser.parse_args()
    config, examples, summary = preflight(args.config, args.examples)
    if args.command == "preflight":
        print(json.dumps(summary, indent=2))
        return
    if not args.model or not args.output:
        parser.error("train/evaluate require --model and --output")
    if args.command == "train":
        train(args, config, examples, summary)
    else:
        evaluate(args, config, examples, summary)


if __name__ == "__main__":
    main()
