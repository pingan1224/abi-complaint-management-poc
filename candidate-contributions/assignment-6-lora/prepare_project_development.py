import argparse
import csv
import json
from collections import Counter
from pathlib import Path


SCOPE = {
    "product": "Credit card",
    "issue": "Billing disputes",
    "submitted_via": "Web",
}
ROUTES = ("Card Billing Disputes", "Card Fraud & Security")
OUTPUT_FIELDS = ("course_record_id", "consumer_complaint_narrative", "routing_destination")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def prepare(development_csv, split_manifest, output_dir):
    dev_rows = read_csv(development_csv)
    scope_rows = [
        row
        for row in dev_rows
        if all(row.get(field) == expected for field, expected in SCOPE.items())
    ]
    if len(scope_rows) != 653:
        raise ValueError(f"Expected the documented 653-row project scope; found {len(scope_rows)}")

    manifest_rows = read_csv(split_manifest)
    manifest_by_id = {row["course_record_id"]: row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("The team split manifest contains duplicate course_record_id values")
    if set(manifest_by_id) != {row["course_record_id"] for row in scope_rows}:
        raise ValueError("Team split manifest IDs do not exactly match the scoped Development cohort")

    clear_rows = [row for row in scope_rows if row["label_status"] == "Clear"]
    if len(clear_rows) != 401:
        raise ValueError(f"Expected 401 Clear-label cases; found {len(clear_rows)}")
    for row in clear_rows:
        if not row["consumer_complaint_narrative"].strip():
            raise ValueError(f"Blank narrative in {row['course_record_id']}")
        if row["routing_destination"] not in ROUTES:
            raise ValueError(f"Unexpected routing label in {row['course_record_id']}")

    selected = {
        row["course_record_id"]: (row, manifest_by_id[row["course_record_id"]])
        for row in clear_rows
    }
    group_splits = {}
    split_counts = Counter()
    route_counts = Counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split_name, manifest_name in (
        ("train.csv", "train_candidate"),
        ("internal_validation.csv", "internal_validation"),
    ):
        written = []
        for record_id, (row, manifest_row) in selected.items():
            split = manifest_row["internal_split"]
            group_hash = manifest_row["narrative_group_sha256"]
            if split == "train_candidate":
                group_splits.setdefault(group_hash, split)
            elif split == "internal_validation":
                if group_hash in group_splits:
                    raise ValueError("A normalized narrative group crosses the internal split boundary")
            if split != manifest_name:
                continue
            written.append({field: row[field] for field in OUTPUT_FIELDS})
            split_counts[split] += 1
            route_counts[f"{split}|{row['routing_destination']}"] += 1
        with (out / split_name).open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(written)

    if split_counts != Counter({"train_candidate": 330, "internal_validation": 71}):
        raise ValueError(f"Unexpected Clear-label split counts: {dict(split_counts)}")
    train_groups = {
        manifest_by_id[record_id]["narrative_group_sha256"]
        for record_id, (row, manifest_row) in selected.items()
        if manifest_row["internal_split"] == "train_candidate"
    }
    validation_groups = {
        manifest_by_id[record_id]["narrative_group_sha256"]
        for record_id, (row, manifest_row) in selected.items()
        if manifest_row["internal_split"] == "internal_validation"
    }
    if train_groups & validation_groups:
        raise ValueError("Narrative-group leakage detected between train and internal validation")

    evidence = {
        "source_split": "Development only",
        "scope_filter": SCOPE,
        "scope_records": len(scope_rows),
        "clear_label_records": len(clear_rows),
        "internal_split_source": "Team Exercise 5 narrative-grouped internal Development manifest",
        "train_records": split_counts["train_candidate"],
        "internal_validation_records": split_counts["internal_validation"],
        "train_route_counts": {
            route: route_counts[f"train_candidate|{route}"] for route in ROUTES
        },
        "internal_validation_route_counts": {
            route: route_counts[f"internal_validation|{route}"] for route in ROUTES
        },
        "narrative_group_overlap": len(train_groups & validation_groups),
        "output_files": ["train.csv", "internal_validation.csv"],
        "model_input_columns": ["consumer_complaint_narrative"],
        "target_column": "routing_destination",
        "excluded_uncertain_status_counts": dict(
            Counter(row["label_status"] for row in scope_rows if row["label_status"] != "Clear")
        ),
        "heldout_rows_inadvertently_previewed_before_freeze": 2,
        "heldout_evaluation_used": False,
        "heldout_preview_note": "A prior schema-only check displayed two sample rows before configuration freeze; neither was used for training or configuration selection. The held-out set is not used in this experiment.",
        "raw_narratives_in_evidence": False,
    }
    encoded = json.dumps(evidence, indent=2, ensure_ascii=False) + "\n"
    (out / "prep_evidence.json").write_text(encoded, encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-csv", required=True)
    parser.add_argument("--split-manifest-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.development_csv, args.split_manifest_csv, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
