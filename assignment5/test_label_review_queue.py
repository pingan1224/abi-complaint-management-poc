"""Small synthetic fixture checks for the Development-only review queue."""

import csv
import unittest
from pathlib import Path
from uuid import uuid4

from label_review_queue import build_queue, load_development, write_queue


FIELDS = [
    "course_record_id", "case_id", "dataset_split", "product", "issue",
    "submitted_via", "consumer_complaint_narrative", "routing_destination",
    "fraud_indicator", "escalation_required", "human_review_required",
    "urgency_level", "label_status",
]


def row(record_id, **changes):
    value = {
        "course_record_id": record_id, "case_id": record_id[4:],
        "dataset_split": "Development", "product": "Credit card",
        "issue": "Billing disputes", "submitted_via": "Web",
        "consumer_complaint_narrative": "Synthetic example; no customer data.",
        "routing_destination": "Card Billing Disputes", "fraud_indicator": "No",
        "escalation_required": "No", "human_review_required": "Yes",
        "urgency_level": "Medium", "label_status": "Clear",
    }
    value.update(changes)
    return value


def fixture(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class ReviewQueueTests(unittest.TestCase):
    def test_flags_without_rewriting_labels_or_exporting_narrative(self):
        token = uuid4().hex
        source = Path(__file__).resolve().parent / ("test_fixture_" + token + ".csv")
        output = Path(__file__).resolve().parent / ("test_queue_" + token + ".csv")
        try:
            fixture(source, [
                row("DEV-00001", fraud_indicator="Yes"),
                row("DEV-00002", label_status="Insufficient Evidence", human_review_required="No"),
                row("DEV-00003", label_status="Judgment Required", human_review_required="No"),
                row("DEV-00004"),
            ])
            rows, checksum = load_development(source)
            cohort, queue = build_queue(rows)
            write_queue(output, queue)
            self.assertEqual(len(checksum), 64)
            self.assertEqual(len(cohort), 4)
            self.assertEqual(len(queue), 3)
            self.assertEqual([item["review_reasons"] for item in queue], [
                "FRAUD_SIGNAL_BILLING_ROUTE", "INSUFFICIENT_NO_HUMAN_REVIEW",
                "JUDGMENT_NO_HUMAN_REVIEW",
            ])
            self.assertNotIn("consumer_complaint_narrative", output.read_text(encoding="utf-8-sig"))
            self.assertEqual(queue[0]["routing_destination"], "Card Billing Disputes")
        finally:
            source.unlink(missing_ok=True)
            output.unlink(missing_ok=True)

    def test_rejects_mixed_heldout_even_if_out_of_scope(self):
        source = Path(__file__).resolve().parent / ("test_fixture_" + uuid4().hex + ".csv")
        try:
            fixture(source, [row("DEV-00001"), row("HEL-00001", dataset_split="Heldout Evaluation", product="Prepaid card")])
            with self.assertRaisesRegex(ValueError, "Only Development"):
                load_development(source)
        finally:
            source.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
