"""Make an ID-only review queue for the team's Development billing-dispute cohort.

This utility does not infer correct labels or make model calls. It flags
instructional references that need a human taxonomy decision.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


REQUIRED = {
    "course_record_id", "case_id", "dataset_split", "product", "issue",
    "submitted_via", "consumer_complaint_narrative", "routing_destination",
    "fraud_indicator", "escalation_required", "human_review_required",
    "urgency_level", "label_status",
}
OUTPUT_FIELDS = [
    "course_record_id", "case_id", "review_reasons", "routing_destination",
    "fraud_indicator", "escalation_required", "human_review_required",
    "urgency_level", "label_status",
]
SCOPE = ("Credit card", "Billing disputes", "Web")
ROUTES = {"Card Billing Disputes", "Card Fraud & Security"}


def load_development(path):
    path = Path(path)
    if "heldout" in path.name.casefold() or "held-out" in path.name.casefold():
        raise ValueError("Held-out files are reserved for formal evaluation")
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines(keepends=True))
    headers = reader.fieldnames or []
    if len(headers) != len(set(headers)):
        raise ValueError("Duplicate CSV column names")
    missing = REQUIRED - set(headers)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    rows = list(reader)
    if not rows:
        raise ValueError("Empty Development file")
    record_ids, case_ids = set(), set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed CSV row")
        record_id, case_id = row["course_record_id"], row["case_id"]
        # Validate all source rows before filtering: a mixed file must fail.
        if row["dataset_split"] != "Development" or not record_id.startswith("DEV-"):
            raise ValueError("Only Development / DEV- records are allowed")
        if not case_id or record_id in record_ids or case_id in case_ids:
            raise ValueError("Missing or duplicate record/case ID")
        if not row["consumer_complaint_narrative"].strip():
            raise ValueError("Blank narrative: " + record_id)
        record_ids.add(record_id)
        case_ids.add(case_id)
    return rows, hashlib.sha256(raw).hexdigest()


def review_reasons(row):
    reasons = []
    if row["fraud_indicator"] == "Yes" and row["routing_destination"] == "Card Billing Disputes":
        reasons.append("FRAUD_SIGNAL_BILLING_ROUTE")
    if row["label_status"] == "Insufficient Evidence" and row["human_review_required"] == "No":
        reasons.append("INSUFFICIENT_NO_HUMAN_REVIEW")
    if row["label_status"] == "Judgment Required" and row["human_review_required"] == "No":
        reasons.append("JUDGMENT_NO_HUMAN_REVIEW")
    return reasons


def build_queue(rows):
    cohort = [row for row in rows if (row["product"], row["issue"], row["submitted_via"]) == SCOPE]
    if not cohort:
        raise ValueError("No in-scope Development records")
    queue = []
    for row in cohort:
        if row["routing_destination"] not in ROUTES:
            raise ValueError("Unrecognized in-scope route: " + row["course_record_id"])
        if any(row[field] not in {"Yes", "No"} for field in ("fraud_indicator", "human_review_required", "escalation_required")):
            raise ValueError("Unrecognized in-scope binary label: " + row["course_record_id"])
        if row["label_status"] not in {"Clear", "Judgment Required", "Insufficient Evidence"}:
            raise ValueError("Unrecognized in-scope label status: " + row["course_record_id"])
        if row["urgency_level"] not in {"Low", "Medium", "High", "Critical"}:
            raise ValueError("Unrecognized in-scope urgency label: " + row["course_record_id"])
        reasons = review_reasons(row)
        if reasons:
            queue.append({field: row[field] for field in OUTPUT_FIELDS if field != "review_reasons"} |
                         {"review_reasons": "|".join(reasons)})
    queue.sort(key=lambda row: row["course_record_id"])
    return cohort, queue


def write_queue(path, queue):
    path = Path(path)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(queue)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development_csv", type=Path)
    parser.add_argument("output_csv", type=Path, help="New local file; never checked in with complaint data")
    args = parser.parse_args()
    try:
        rows, source_sha256 = load_development(args.development_csv)
        cohort, queue = build_queue(rows)
        write_queue(args.output_csv, queue)
    except (ValueError, OSError, UnicodeError) as error:
        parser.exit(2, "Error: " + str(error) + "\n")
    print(json.dumps({
        "source_sha256": source_sha256,
        "development_rows": len(rows),
        "in_scope_rows": len(cohort),
        "queued_unique_cases": len(queue),
        "reason_counts": dict(sorted(Counter(reason for row in queue for reason in row["review_reasons"].split("|")).items())),
        "output_csv": str(args.output_csv),
    }, indent=2))


if __name__ == "__main__":
    main()
