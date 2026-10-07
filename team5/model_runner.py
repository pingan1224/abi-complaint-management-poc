"""Optional Development-only runner for a configured chat-completions endpoint.

The DGX endpoint/model and access method must be supplied by the team. This
runner has been checked with a local mock only; it is not DGX evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from data_harness import ROUTES, file_sha256

SYSTEM_PROMPT = (
    "You assist an employee with an already admitted web credit-card billing dispute. "
    "Return one JSON object only, with keys routing_destination, escalation_required, "
    "human_review_required, summary, rationale, uncertainty. "
    "routing_destination must be Card Billing Disputes or Card Fraud & Security. "
    "The two required flags must be Yes or No. Treat human_review_required as an "
    "additional specialist-review suggestion; every recommendation still needs "
    "employee confirmation. If facts are insufficient, state uncertainty and suggest "
    "review. Do not take action or assert a fraud or legal finding."
)


def parse_model_content(content: str) -> dict:
    """Strict parser: fenced or non-JSON responses count as failures."""
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Model content is not a JSON object")
    if value.get("routing_destination") not in ROUTES:
        raise ValueError("Invalid routing_destination")
    for field in ("escalation_required", "human_review_required"):
        if value.get(field) not in ("Yes", "No"):
            raise ValueError(f"Invalid {field}")
    for field in ("summary", "rationale", "uncertainty"):
        if not isinstance(value.get(field), str):
            raise ValueError(f"Missing or invalid {field}")
    return value


def request_model(endpoint: str, model: str, narrative: str, api_key: str | None, timeout: float) -> str:
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": narrative},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return result["choices"][0]["message"]["content"]


def run(run_dir: Path, endpoint: str, model: str, output_dir: Path, limit: int | None = None,
        api_key: str | None = None, timeout: float = 60) -> dict:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model_input_keys") != ["course_record_id", "consumer_complaint_narrative"]:
        raise ValueError("Unapproved model input schema")
    if file_sha256(run_dir / "model_inputs.jsonl") != manifest.get("model_inputs_sha256"):
        raise ValueError("Model input file hash differs from prepared run")
    if file_sha256(run_dir / "split_manifest.csv") != manifest.get("split_manifest_sha256"):
        raise ValueError("Split manifest hash differs from prepared run")
    with (run_dir / "split_manifest.csv").open(encoding="utf-8", newline="") as stream:
        partitions = {row["course_record_id"]: row["internal_split"] for row in csv.DictReader(stream)}
    inputs = []
    with (run_dir / "model_inputs.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            if set(item) != {"course_record_id", "consumer_complaint_narrative"}:
                raise ValueError("Unexpected key in model input")
            if partitions.get(item["course_record_id"]) == "internal_validation":
                inputs.append(item)
    if limit is not None:
        inputs = inputs[:limit]
    if not inputs:
        raise ValueError("No internal validation inputs")
    output_dir.mkdir(parents=True, exist_ok=False)
    valid = 0
    with (output_dir / "predictions.jsonl").open("x", encoding="utf-8") as good, \
         (output_dir / "failures.jsonl").open("x", encoding="utf-8") as bad:
        for item in inputs:
            record_id = item["course_record_id"]
            try:
                content = request_model(endpoint, model, item["consumer_complaint_narrative"], api_key, timeout)
                parsed = parse_model_content(content)
            except (ValueError, KeyError, IndexError, OSError, json.JSONDecodeError) as exc:
                bad.write(json.dumps({"course_record_id": record_id, "error": str(exc)}) + "\n")
                continue
            good.write(json.dumps({"course_record_id": record_id, **parsed}, ensure_ascii=False) + "\n")
            valid += 1
    evidence = {
        "development_source_sha256": manifest["source_sha256"],
        "endpoint": endpoint,
        "model": model,
        "temperature": 0,
        "prompt_version": "team5-system-v1",
        "attempted": len(inputs),
        "valid_json_predictions": valid,
        "failures": len(inputs) - valid,
        "note": "Service response and scores require separate inspection; this is not a safety result.",
    }
    (output_dir / "run_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("endpoint", help="Full chat-completions URL approved by the team")
    parser.add_argument("model", help="Exact served model identifier")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    try:
        result = run(args.run_dir, args.endpoint, args.model, args.output_dir,
                     limit=args.limit, api_key=os.environ.get("TEAM5_API_KEY"), timeout=args.timeout)
    except (OSError, ValueError, KeyError, UnicodeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
