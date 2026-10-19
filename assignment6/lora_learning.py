"""Bounded DGX experiment: train a LoRA, save/reload it, compare raw responses.

Only a supplied Development CSV and an existing grouped split are used. Course
reference labels measure agreement, not correctness under official bank policy.
"""
import argparse
import collections
import csv
import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

FIELDS = ["urgency_level", "routing_destination", "escalation_required"]
CLASSES = ["Medium", "High", "Critical"]
SEED = 42


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def group_key(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def target(row):
    flag = row["escalation_required"]
    if flag not in ("Yes", "No"):
        raise ValueError("Unrecognized escalation reference")
    return {"urgency_level": row["urgency_level"],
            "routing_destination": row["routing_destination"],
            "escalation_required": flag == "Yes"}


def prompt(narrative, enhanced, routes):
    # Construct model input explicitly; never serialize complete source rows.
    basic = "Classify this complaint. Return a JSON object with exactly urgency_level, routing_destination, and escalation_required."
    extra = (" Return only JSON, without Markdown or explanation. urgency_level must be Low, Medium, High, Critical, or Unknown. "
             "routing_destination must be one of " + json.dumps(routes + ["Unknown"]) + ". "
             "escalation_required must be true, false, or null. Use only evidence in the complaint; "
             "use Unknown or null when the evidence does not support a decision. Never invent facts. "
             "These are candidate recommendations requiring employee confirmation.")
    return [{"role": "user", "content": basic + (extra if enhanced else "") + "\nComplaint:\n" + narrative}]


def prepare(csv_path, manifest_path, tokenizer, max_length):
    rows = read_csv(csv_path)
    if len(rows) != 8000 or any(r.get("dataset_split") != "Development" for r in rows):
        raise ValueError("Expected only the supplied 8000-row Development dataset")
    cohort = {r["course_record_id"]: r for r in rows
              if r["product"] == "Credit card" and r["issue"] == "Billing disputes"}
    manifest = read_csv(manifest_path)
    if len(cohort) != len(manifest) or set(cohort) != {r["course_record_id"] for r in manifest}:
        raise ValueError("Manifest and scoped Development IDs differ")
    assigned = {}
    for item in manifest:
        row = cohort[item["course_record_id"]]
        key = group_key(row["consumer_complaint_narrative"])
        if key != item["narrative_group_sha256"]:
            raise ValueError("Source narrative differs from the existing split manifest")
        if key in assigned and assigned[key] != item["internal_split"]:
            raise ValueError("Duplicate narrative crosses split boundaries")
        assigned[key] = item["internal_split"]
        row["internal_split"] = item["internal_split"]
        row["group"] = key
    routes = sorted({r["routing_destination"] for r in cohort.values() if r["internal_split"] == "train_candidate"})

    def encoded(row):
        messages = prompt(row["consumer_complaint_narrative"], True, routes)
        prefix_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        complete_text = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": json.dumps(target(row))}],
                                                       tokenize=False, add_generation_prompt=False)
        prefix = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        complete = tokenizer(complete_text, add_special_tokens=False)["input_ids"]
        if complete[:len(prefix)] != prefix:
            raise ValueError("Chat-template prefix mismatch; do not mask an assumed boundary")
        return prefix, complete

    ordered = sorted(cohort.values(), key=lambda r: hashlib.sha256(("ABI6-v1:" + r["course_record_id"]).encode()).hexdigest())
    seen = set()
    train, validation = [], []
    skipped = collections.Counter()
    for split, dest, per_class in [("train_candidate", train, 16), ("internal_validation", validation, 3)]:
        for urgency in CLASSES:
            chosen = 0
            for row in ordered:
                if row["internal_split"] != split or row["label_status"] != "Clear" or row["urgency_level"] != urgency or row["group"] in seen:
                    continue
                prefix, complete = encoded(row)
                if len(complete) > max_length:
                    skipped[split] += 1
                    continue
                seen.add(row["group"])
                dest.append(row)
                chosen += 1
                if chosen == per_class:
                    break
            if chosen != per_class:
                raise ValueError("Not enough eligible short clear cases for " + split + ":" + urgency)
    for row in ordered:
        if row["internal_split"] == "internal_validation" and row["label_status"] != "Clear" and row["group"] not in seen:
            prefix, complete = encoded(row)
            if len(complete) <= max_length:
                validation.append(row)
                seen.add(row["group"])
                if len(validation) == 12:
                    break
    if len(validation) != 12:
        raise ValueError("Not enough uncertain validation cases")
    assert not ({r["group"] for r in train} & {r["group"] for r in validation})
    examples = []
    for row in train:
        prefix, complete = encoded(row)
        labels = [-100] * len(prefix) + complete[len(prefix):]
        assert len(labels) == len(complete) and any(v != -100 for v in labels)
        examples.append((complete, labels))
    provenance = {"source_sha256": digest(csv_path), "manifest_sha256": digest(manifest_path),
                  "selection_rule": "ABI6-v1 hash order; unique narrative groups; full training sequence <=512 tokens; no truncation",
                  "train_per_class": 16, "validation_clear_per_class": 3, "validation_uncertain": 3,
                  "scope_count": len(cohort), "skipped_long_candidates": dict(skipped),
                  "no_shared_narrative_groups": True, "course_final_holdout_opened": False,
                  "training_ids": [r["course_record_id"] for r in train],
                  "validation_ids": [r["course_record_id"] for r in validation],
                  "low_coverage": 0, "references": "Synthetic course references; Clear is not independent expert adjudication"}
    return train, validation, examples, routes, provenance


def score(raw, row, routes):
    result = {"strict_json": False, "schema_valid": False, "reference_agreement": None}
    try:
        obj = json.loads(raw)
        result["strict_json"] = isinstance(obj, dict)
        if not isinstance(obj, dict) or set(obj) != set(FIELDS):
            return result
        if obj["urgency_level"] not in ["Low", "Medium", "High", "Critical", "Unknown"]:
            return result
        if obj["routing_destination"] not in routes + ["Unknown"]:
            return result
        if obj["escalation_required"] is not None and type(obj["escalation_required"]) is not bool:
            return result
        result["schema_valid"] = True
        result["parsed"] = obj
        if row["label_status"] == "Clear":
            result["reference_agreement"] = obj == target(row)
    except (ValueError, TypeError):
        pass
    return result


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    events = out / "events.jsonl"

    def event(stage, **values):
        record = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "stage": stage, **values}
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: request the DGX GPU resource through Slurm")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    config = {"base_model": args.model, "seed": SEED, "steps": args.steps, "max_sequence_tokens": 512,
              "optimizer": "AdamW", "learning_rate": 2e-4, "weight_decay": 0.0,
              "batch_size": 1, "gradient_accumulation": 1, "clip_grad_norm": 1.0,
              "dtype": "bfloat16", "quantization": None, "attention": "sdpa",
              "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"], "bias": "none", "task_type": "CAUSAL_LM"},
              "generation": {"do_sample": False, "max_new_tokens": 128},
              "decision_rule": "Recommend further team review only if C improves schema-valid outputs over B without a newly observed serious semantic error. Otherwise retain B as the current candidate. This exploratory sample cannot establish deployment readiness.",
              "semantic_review_required": True,
              "host": platform.node(), "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
              "packages": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "peft", "datasets", "trl", "accelerate"]},
              "script_sha256": digest(__file__), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    config["code_commit"] = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                                             capture_output=True, text=True, check=True).stdout.strip()
    write_json(out / "frozen_config.json", config)
    event("configuration_frozen", gpu=config["gpu"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    train, validation, examples, routes, provenance = prepare(args.csv, args.manifest, tokenizer, 512)
    write_json(out / "data_provenance.json", provenance)
    write_json(out / "validation_cases.json", [{k: r[k] for k in ["course_record_id", "consumer_complaint_narrative", "label_status", *FIELDS]} for r in validation])
    # Explicitly check original source/config identity; no Hub revision is inferred.
    model_files = []
    for file in sorted(Path(args.model).iterdir()):
        if file.suffix in (".json", ".safetensors"):
            model_files.append({"name": file.name, "bytes": file.stat().st_size, "sha256": digest(file)})
    write_json(out / "base_model_files.json", model_files)
    write_json(out / "prompts.json", {"A": prompt("<NARRATIVE>", False, routes), "B_and_C": prompt("<NARRATIVE>", True, routes)})
    event("data_validated", training=len(train), validation=len(validation), no_group_overlap=True)

    def load_base():
        return AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16,
                                                    attn_implementation="sdpa").to("cuda")

    responses = []

    def generate(model, row, enhanced):
        text = tokenizer.apply_chat_template(prompt(row["consumer_complaint_narrative"], enhanced, routes), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            result = model.generate(**inputs, do_sample=False, max_new_tokens=128,
                                    pad_token_id=tokenizer.eos_token_id, use_cache=True)
        torch.cuda.synchronize()
        raw = tokenizer.decode(result[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return raw, time.monotonic() - start, int(result.shape[1] - inputs["input_ids"].shape[1])

    def evaluate(model, condition, enhanced):
        model.eval()
        for row in validation:
            raw, seconds, tokens = generate(model, row, enhanced)
            rec = {"condition": condition, "course_record_id": row["course_record_id"], "label_status": row["label_status"],
                   "raw_response": raw, "seconds": seconds, "new_tokens": tokens, **score(raw, row, routes)}
            responses.append(rec)
            with (out / "responses.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
        event("evaluation_complete", condition=condition)

    model = load_base()
    evaluate(model, "A_base", False)
    evaluate(model, "B_prompted", True)
    model = get_peft_model(model, LoraConfig(**config["lora"]))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    event("training_started", trainable_parameters=sum(p.numel() for p in trainable), total_parameters=sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.0)
    model.train()
    started = time.monotonic()
    order = list(range(len(examples)))
    random.shuffle(order)
    for step in range(args.steps):
        ids, labels = examples[order[step % len(order)]]
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=torch.tensor([ids], device="cuda"), labels=torch.tensor([labels], device="cuda")).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; preserve the failed run")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        event("training_step", step=step + 1, loss=float(loss.detach()), tokens=len(ids))
    torch.cuda.synchronize()
    train_seconds = time.monotonic() - started
    adapter_path = out / "adapter"
    model.save_pretrained(adapter_path, safe_serialization=True)
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    model.eval()
    before_raw, _, _ = generate(model, validation[0], True)
    # Compare saved tensors as well as one decoded response after a fresh load.
    from peft import get_peft_model_state_dict
    saved_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    del optimizer, trainable, model
    gc.collect()
    torch.cuda.empty_cache()
    base = load_base()
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    loaded_state = get_peft_model_state_dict(model)
    state_equal = set(saved_state) == set(loaded_state) and all(torch.equal(saved_state[k], loaded_state[k].detach().cpu()) for k in saved_state)
    after_raw, _, _ = generate(model, validation[0], True)
    reload_check = {"adapter_state_exactly_equal": state_equal, "probe_case": validation[0]["course_record_id"],
                    "greedy_raw_response_equal": before_raw == after_raw, "before": before_raw, "after": after_raw,
                    "active_adapters": list(model.active_adapters), "training_seconds": train_seconds,
                    "peak_torch_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "adapter_bytes": sum(p.stat().st_size for p in adapter_path.iterdir() if p.is_file())}
    write_json(out / "reload_check.json", reload_check)
    if not state_equal:
        raise RuntimeError("Saved and reloaded adapter tensors differ")
    event("adapter_reloaded", state_equal=state_equal, probe_response_equal=before_raw == after_raw)
    evaluate(model, "C_adapted", True)
    summary = {}
    for condition in ("A_base", "B_prompted", "C_adapted"):
        subset = [r for r in responses if r["condition"] == condition]
        clear = [r for r in subset if r["label_status"] == "Clear"]
        summary[condition] = {"n": len(subset), "strict_json": sum(r["strict_json"] for r in subset),
                              "schema_valid": sum(r["schema_valid"] for r in subset),
                              "clear_reference_agreement": sum(r["reference_agreement"] is True for r in clear),
                              "clear_n": len(clear), "total_generation_seconds": sum(r["seconds"] for r in subset)}
    write_json(out / "summary.json", summary)
    event("experiment_complete", summary=summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=48)
    args = parser.parse_args()
    if not 1 <= args.steps <= 48:
        parser.error("This bounded learning experiment accepts 1 to 48 steps")
    run(args)
