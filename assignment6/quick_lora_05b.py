"""Time-limited DGX LoRA fallback using the approved Qwen 2.5 0.5B checkpoint.

This is a separate experiment from the queued 32B run. It uses the same eight
hand-written synthetic examples and never opens course Development or held-out data.
"""

from __future__ import annotations

import gc
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from lora_experiment import ROOT, SYSTEM, digest, preflight

MODEL = Path("/mnt/nas/models/approved/qwen2.5-0.5b-instruct")
OUTPUT_ROOT = Path.home()
MAX_LENGTH = 192
MAX_NEW_TOKENS = 12
SEED = 2026


def model_config_hash() -> str:
    if MODEL.name != "qwen2.5-0.5b-instruct":
        raise ValueError("This fallback is restricted to the approved 0.5B directory")
    config_file = MODEL / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen2" or config.get("quantization_config"):
        raise ValueError("Expected an unquantized Qwen2 checkpoint")
    return digest(config_file)


def encode_training(tokenizer, row):
    prefix = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": row["narrative"]},
    ]
    prompt = tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        prefix + [{"role": "assistant", "content": row["decision"]}],
        tokenize=True, add_generation_prompt=False,
    )
    if full[:len(prompt)] != prompt or len(full) > MAX_LENGTH or len(full) == len(prompt):
        raise ValueError(f"Invalid chat-template training example: {row['id']}")
    return full, [-100] * len(prompt) + full[len(prompt):]


def main() -> None:
    import torch
    import transformers
    import peft
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    config, examples, summary = preflight(ROOT / "experiment.json", ROOT / "synthetic_examples.jsonl")
    model_hash = model_config_hash()
    job_id = __import__("os").environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("Run only inside an allocated Slurm job")
    output = OUTPUT_ROOT / f"a6-quick05b-run-{job_id}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load_base():
        base = AutoModelForCausalLM.from_pretrained(
            MODEL, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to("cuda")
        dtype = str(base.model.layers[0].self_attn.q_proj.weight.dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Unexpected base dtype: {dtype}")
        return base

    started = time.monotonic()
    model = get_peft_model(load_base(), LoraConfig(
        r=config["lora_r"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], target_modules=config["target_modules"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    ))
    trainable, total = model.get_nb_trainable_parameters()
    if not 0 < trainable < total:
        raise RuntimeError("Adapter parameter check failed")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config["learning_rate"])
    model.train()
    losses = []
    train_rows = [row for row in examples if row["split"] == "train"]
    for step, row in enumerate(train_rows[:config["max_steps"]], 1):
        ids, labels = encode_training(tokenizer, row)
        inputs = torch.tensor([ids], dtype=torch.long, device="cuda")
        targets = torch.tensor([labels], dtype=torch.long, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=inputs, labels=targets).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        losses.append(round(float(loss.detach().cpu()), 6))
        print(f"step={step} loss={losses[-1]}", flush=True)

    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    if not (adapter / "adapter_config.json").is_file() or not any(adapter.glob("adapter_model.*")):
        raise RuntimeError("Adapter files were not saved")
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = PeftModel.from_pretrained(load_base(), adapter, is_trainable=False)
    reloaded.eval()

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
            ids = reloaded.generate(
                **encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(ids[0, encoded["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    cases = []
    for row in examples:
        if row["split"] != "eval":
            continue
        with reloaded.disable_adapter():
            base_text = generate(row)
        adapted_text = generate(row)
        cases.append({
            "id": row["id"], "reference": row["decision"],
            "base_output": base_text, "adapted_output": adapted_text,
            "base_exact_match": base_text == row["decision"],
            "adapted_exact_match": adapted_text == row["decision"],
        })
    evidence = {
        **summary,
        "status": "dgx_quick_05b_lora_trained_reloaded_compared",
        "utc_completed": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "slurm_job_id": job_id,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "model_directory": str(MODEL),
        "model_config_sha256": model_hash,
        "deviation_from_32b_plan": "Time-limited fallback using approved unquantized Qwen 2.5 0.5B; 32B job remains separate",
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "train_losses": losses,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "peak_torch_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "adapter_files": {p.name: digest(p) for p in adapter.iterdir() if p.is_file()},
        "cases": cases,
        "base_exact_matches": sum(row["base_exact_match"] for row in cases),
        "adapted_exact_matches": sum(row["adapted_exact_match"] for row in cases),
    }
    (output / "quick_run_evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({"status": evidence["status"], "output": str(output),
                      "train_losses": losses, "cases": cases}, indent=2), flush=True)


if __name__ == "__main__":
    main()
