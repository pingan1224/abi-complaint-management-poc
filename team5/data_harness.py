"""Development-only input boundary, duplicate-safe split, and reference scoring.

Only ``model_inputs.jsonl`` may be passed to a model. Source labels and
evaluator metadata are read here solely for scope selection and local scoring.
No held-out data access or model service is implemented in this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SCOPE = {"product": "Credit card", "issue": "Billing disputes", "submitted_via": "Web"}
ROUTES = ("Card Billing Disputes", "Card Fraud & Security")
INPUT_KEYS = ("course_record_id", "consumer_complaint_narrative")
REQUIRED = {
    "course_record_id", "dataset_split", "case_id", "product", "issue",
    "submitted_via", "consumer_complaint_narrative", "routing_destination",
    "escalation_required", "human_review_required", "label_status",
    "label_confidence", "fraud_indicator", "urgency_level",
}
SEED = "group2-team5-v1"


def load_development(path: Path) -> tuple[list[dict[str, str]], str]:
    """Reject held-out or malformed records before any scope filtering."""
    if "heldout" in path.name.casefold() or "held-out" in path.name.casefold():
        raise ValueError("Held-out file name is forbidden in development mode")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines(keepends=True))
    columns = reader.fieldnames or []
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate CSV headers")
    if missing := REQUIRED - set(columns):
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    rows = list(reader)
    if not rows:
        raise ValueError("Empty Development CSV")
    ids: set[str] = set()
    source_ids: set[str] = set()
    for number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV row {number}")
        record_id = row["course_record_id"]
        source_id = row["case_id"]
        if row["dataset_split"] != "Development" or not record_id.startswith("DEV-"):
            raise ValueError(f"Non-Development record at CSV row {number}")
        if not source_id or record_id in ids or source_id in source_ids:
            raise ValueError(f"Missing or duplicate identifier at CSV row {number}")
        if not row["consumer_complaint_narrative"].strip():
            raise ValueError(f"Blank narrative at CSV row {number}")
        ids.add(record_id)
        source_ids.add(source_id)
    return rows, digest


def narrative_group(text: str) -> str:
    """Hash whitespace-normalized text for grouping, without changing model text."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_map(rows: list[dict[str, str]], fraction: float = 0.2, seed: str = SEED) -> dict[str, str]:
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    groups = {narrative_group(row["consumer_complaint_narrative"]) for row in rows}
    if len(groups) < 2:
        raise ValueError("At least two distinct narrative groups required")
    ordered = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
    n = min(len(ordered) - 1, max(1, round(len(ordered) * fraction)))
    validation = set(ordered[:n])
    return {
        row["course_record_id"]: (
            "internal_validation" if narrative_group(row["consumer_complaint_narrative"]) in validation
            else "train_candidate"
        )
        for row in rows
    }


def cohort(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [row for row in rows if all(row[key] == value for key, value in SCOPE.items())]
    if not selected:
        raise ValueError("No records in the bounded cohort")
    for row in selected:
        if row["routing_destination"] not in ROUTES:
            raise ValueError("Unexpected route in bounded cohort: " + row["course_record_id"])
        if row["escalation_required"] not in ("Yes", "No") or row["human_review_required"] not in ("Yes", "No"):
            raise ValueError("Unexpected binary reference: " + row["course_record_id"])
    return selected


def model_input(row: dict[str, str]) -> dict[str, str]:
    """Construct the model-facing object from an explicit two-key allowlist."""
    return {
        "course_record_id": row["course_record_id"],
        "consumer_complaint_narrative": row["consumer_complaint_narrative"],
    }


def prepare(source: Path, output_dir: Path, fraction: float = 0.2, seed: str = SEED) -> dict:
    rows, digest = load_development(source)
    partitions = split_map(rows, fraction, seed)  # Group across all Development rows.
    selected = cohort(rows)
    inputs = [model_input(row) for row in selected]
    if any(tuple(item) != INPUT_KEYS for item in inputs):
        raise AssertionError("Model input projection changed")
    summary = {
        "source_sha256": digest,
        "source_rows": len(rows),
        "scope": SCOPE,
        "scope_rows": len(selected),
        "model_input_keys": list(INPUT_KEYS),
        "split_seed": seed,
        "validation_group_fraction": fraction,
        "split_counts": dict(sorted(Counter(partitions[row["course_record_id"]] for row in selected).items())),
        "routing_reference_counts": dict(sorted(Counter(row["routing_destination"] for row in selected).items())),
        "label_status_counts": dict(sorted(Counter(row["label_status"] for row in selected).items())),
        "limitations": [
            "Source issue is an offline cohort filter, not a model input or validated live intake gate.",
            "Whitespace-identical narratives are grouped; near duplicates are not detected.",
            "Synthetic labels remain provisional and require adjudication for consequential claims.",
            "Internal validation is Development evidence, not held-out evaluation.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "model_inputs.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for item in inputs:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (output_dir / "split_manifest.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["course_record_id", "internal_split", "narrative_group_sha256"])
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "course_record_id": row["course_record_id"],
                "internal_split": partitions[row["course_record_id"]],
                "narrative_group_sha256": narrative_group(row["consumer_complaint_narrative"]),
            })
    summary["model_inputs_sha256"] = file_sha256(output_dir / "model_inputs.jsonl")
    summary["split_manifest_sha256"] = file_sha256(output_dir / "split_manifest.csv")
    (output_dir / "run_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def read_predictions(path: Path) -> list[dict]:
    predictions = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prediction JSON line {number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Prediction line {number} is not an object")
            predictions.append(item)
    if not predictions:
        raise ValueError("No predictions")
    return predictions


def score(source: Path, run_dir: Path, predictions_path: Path) -> dict:
    rows, digest = load_development(source)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if digest != manifest["source_sha256"]:
        raise ValueError("Development file hash differs from prepared run")
    partitions = split_map(rows, manifest["validation_group_fraction"], manifest["split_seed"])
    selected = {row["course_record_id"]: row for row in cohort(rows)}
    outputs = read_predictions(predictions_path)
    seen: set[str] = set()
    route_pairs: Counter[tuple[str, str]] = Counter()
    escalation_pairs: Counter[tuple[str, str]] = Counter()
    review_pairs: Counter[tuple[str, str]] = Counter()
    status_counts: Counter[str] = Counter()
    for output in outputs:
        record_id = output.get("course_record_id")
        if record_id in seen or record_id not in selected:
            raise ValueError("Duplicate or out-of-scope prediction ID")
        if partitions[record_id] != "internal_validation":
            raise ValueError("Scoring accepts internal_validation IDs only")
        route = output.get("routing_destination")
        escalation = output.get("escalation_required")
        review = output.get("human_review_required")
        if route not in ROUTES or escalation not in ("Yes", "No") or review not in ("Yes", "No"):
            raise ValueError("Prediction has invalid route or flag values")
        seen.add(record_id)
        ref = selected[record_id]
        route_pairs[(ref["routing_destination"], route)] += 1
        escalation_pairs[(ref["escalation_required"], escalation)] += 1
        review_pairs[(ref["human_review_required"], review)] += 1
        status_counts[ref["label_status"]] += 1
    def agreement(pairs: Counter[tuple[str, str]]) -> dict:
        return {"matches": sum(n for (expected, actual), n in pairs.items() if expected == actual),
                "n": sum(pairs.values()),
                "confusion": [{"reference": a, "prediction": b, "n": n} for (a, b), n in sorted(pairs.items())]}
    return {
        "split": "internal_validation",
        "n_predictions": len(outputs),
        "route_reference_agreement": agreement(route_pairs),
        "escalation_reference_agreement": agreement(escalation_pairs),
        "human_review_reference_agreement_provisional": agreement(review_pairs),
        "label_status_counts": dict(sorted(status_counts.items())),
        "caution": "Agreement with course-created labels is not bank ground truth or a safety claim. Resolve disputed review labels before interpreting this metric.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("development_csv", type=Path)
    prep.add_argument("output_dir", type=Path)
    prep.add_argument("--validation-fraction", type=float, default=0.2)
    prep.add_argument("--seed", default=SEED)
    scoring = commands.add_parser("score")
    scoring.add_argument("development_csv", type=Path)
    scoring.add_argument("run_dir", type=Path)
    scoring.add_argument("predictions_jsonl", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.development_csv, args.output_dir, args.validation_fraction, args.seed)
        else:
            result = score(args.development_csv, args.run_dir, args.predictions_jsonl)
    except (OSError, ValueError, KeyError, UnicodeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
