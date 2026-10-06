"""Build a development-only, duplicate-protected internal split manifest.

Standard library only. Preserves source rows and reference labels. No model calls.
"""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED = {
    "course_record_id", "case_id", "dataset_split", "product", "issue",
    "consumer_complaint_narrative", "urgency_level", "routing_destination",
    "escalation_required", "human_review_required", "label_status",
    "label_confidence", "special_case_flags", "word_count",
}
URGENCY = ("Low", "Medium", "High", "Critical")


def load_development(path):
    raw = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    columns = reader.fieldnames or []
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate CSV column names")
    if REQUIRED - set(columns):
        raise ValueError("Missing columns: " + ", ".join(sorted(REQUIRED - set(columns))))
    rows = list(reader)
    if not rows:
        raise ValueError("Empty dataset")
    record_ids, case_ids = set(), set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed CSV row")
        rid, cid = row["course_record_id"], row["case_id"]
        # Validate every row before filtering; even out-of-scope held-out rows fail.
        if row["dataset_split"] != "Development" or not rid.startswith("DEV-"):
            raise ValueError("Only Development / DEV- records are permitted")
        if not cid or rid in record_ids or cid in case_ids:
            raise ValueError("Missing or duplicate record/case ID")
        record_ids.add(rid)
        case_ids.add(cid)
        if not row["consumer_complaint_narrative"].strip():
            raise ValueError("Blank narrative: " + rid)
        if row["urgency_level"] not in URGENCY:
            raise ValueError("Unknown urgency label: " + rid)
        for field in ("human_review_required", "escalation_required"):
            if row[field] not in ("Yes", "No"):
                raise ValueError("Unknown binary label: " + rid)
    return rows, hashlib.sha256(raw).hexdigest()


def narrative_key(text):
    # Whitespace normalization applies only to grouping, never to source text.
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def make_manifest(rows, validation_fraction=0.20, seed="ABI5-v1"):
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be strictly between 0 and 1")
    groups = defaultdict(list)
    for row in rows:
        groups[narrative_key(row["consumer_complaint_narrative"])].append(row)
    if len(groups) < 2:
        raise ValueError("At least two distinct narrative groups are required")
    # Whole-group assignment. Fraction refers to groups; record counts can differ.
    ordered = sorted(groups, key=lambda key: hashlib.sha256((seed + ":" + key).encode()).hexdigest())
    n_validation = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
    validation_groups = set(ordered[:n_validation])
    manifest = []
    for key in sorted(groups):
        split = "internal_validation" if key in validation_groups else "train_candidate"
        for row in sorted(groups[key], key=lambda row: row["course_record_id"]):
            manifest.append({
                "course_record_id": row["course_record_id"], "case_id": row["case_id"],
                "narrative_group_sha256": key, "internal_split": split,
                "urgency_level": row["urgency_level"], "label_status": row["label_status"],
            })
    return manifest, groups


def audit(rows, source_sha256, validation_fraction=0.20, seed="ABI5-v1"):
    cohort = [row for row in rows if row["product"] == "Credit card" and row["issue"] == "Billing disputes"]
    if not cohort:
        raise ValueError("No credit-card Billing disputes records")
    manifest, groups = make_manifest(cohort, validation_fraction, seed)
    fields = ("urgency_level", "routing_destination", "escalation_required",
              "human_review_required", "label_status", "label_confidence")
    counts = {field: dict(sorted(Counter(row[field] for row in cohort).items())) for field in fields}
    urgency_counts = {value: counts["urgency_level"].get(value, 0) for value in URGENCY}
    per_split = {}
    for split in ("train_candidate", "internal_validation"):
        subset = [row for row in manifest if row["internal_split"] == split]
        per_split[split] = {
            "records": len(subset),
            "urgency": {value: sum(row["urgency_level"] == value for row in subset) for value in URGENCY},
            "label_status": dict(sorted(Counter(row["label_status"] for row in subset).items())),
        }
    duplicate_groups = [sorted(row["course_record_id"] for row in group) for group in groups.values() if len(group) > 1]
    review_rows = [row for row in cohort if row["label_status"] == "Insufficient Evidence"]
    profile = {
        "source_sha256": source_sha256, "development_records": len(rows),
        "scope": {"product": "Credit card", "issue": "Billing disputes", "records": len(cohort)},
        "counts": counts, "urgency_counts_including_zero": urgency_counts,
        "missing_urgency_classes": [value for value in URGENCY if not urgency_counts[value]],
        "all_development_low": sum(row["urgency_level"] == "Low" for row in rows),
        "credit_card_low": sum(row["product"] == "Credit card" and row["urgency_level"] == "Low" for row in rows),
        "missing_values": {field: sum(not row[field].strip() for row in cohort) for field in rows[0]
                           if any(not row[field].strip() for row in cohort)},
        "exact_duplicate_groups": [sorted(row["course_record_id"] for row in group)
                                   for group in _exact_groups(cohort).values() if len(group) > 1],
        "whitespace_normalized_duplicate_groups": sorted(duplicate_groups),
        "insufficient_evidence_with_no_human_review": sum(row["human_review_required"] == "No" for row in review_rows),
        "judgment_required_with_no_human_review": sum(row["label_status"] == "Judgment Required" and row["human_review_required"] == "No" for row in cohort),
        "split_configuration": {"seed": seed, "validation_group_fraction": validation_fraction,
                                "stratified": False, "grouping": "case-sensitive whitespace-normalized narrative SHA256"},
        "internal_split_counts": per_split,
        "warnings": ["Candidate internal split only; no training or model evaluation has occurred.",
                     "Labels are administrative references, not model inputs.",
                     "Exact/whitespace duplicates are protected; near duplicates are not detected.",
                     "This grouping within the selected cohort does not protect later scope expansion.",
                     "Coverage must be reviewed before adopting this unstratified split."],
    }
    return profile, manifest, review_rows


def _exact_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["consumer_complaint_narrative"]].append(row)
    return groups


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development_csv", type=Path)
    parser.add_argument("output_dir", type=Path, help="Must not already exist")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", default="ABI5-v1")
    args = parser.parse_args()
    try:
        rows, checksum = load_development(args.development_csv)
        profile, manifest, review_rows = audit(rows, checksum, args.validation_fraction, args.seed)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "audit_profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        write_csv(args.output_dir / "internal_split_manifest.csv", list(manifest[0]), manifest)
        review_fields = ["course_record_id", "case_id", "consumer_complaint_narrative", "urgency_level",
                         "routing_destination", "escalation_required", "human_review_required", "label_status",
                         "reviewer_decision", "reviewer_rationale"]
        write_csv(args.output_dir / "insufficient_evidence_review.csv", review_fields, review_rows)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.exit(2, "Error: " + str(exc) + "\n")
    print(json.dumps({"scope_records": profile["scope"]["records"],
                      "missing_urgency_classes": profile["missing_urgency_classes"],
                      "duplicate_groups": len(profile["whitespace_normalized_duplicate_groups"]),
                      "review_cases": len(review_rows), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
