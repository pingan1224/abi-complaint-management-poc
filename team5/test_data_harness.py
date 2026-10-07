"""Small controlled checks for the Team 5 Development data boundary."""

import csv
import io
import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from data_harness import INPUT_KEYS, load_development, prepare, score, split_map
from model_runner import parse_model_content, run

FIELDS = [
    "course_record_id", "dataset_split", "case_id", "product", "issue",
    "submitted_via", "consumer_complaint_narrative", "routing_destination",
    "escalation_required", "human_review_required", "label_status",
    "label_confidence", "fraud_indicator", "urgency_level",
    "company_response_to_consumer", "special_case_flags",
]


def row(record_id, case_id, text, **updates):
    value = {
        "course_record_id": record_id,
        "dataset_split": "Development",
        "case_id": case_id,
        "product": "Credit card",
        "issue": "Billing disputes",
        "submitted_via": "Web",
        "consumer_complaint_narrative": text,
        "routing_destination": "Card Billing Disputes",
        "escalation_required": "No",
        "human_review_required": "No",
        "label_status": "Clear",
        "label_confidence": "High",
        "fraud_indicator": "No",
        "urgency_level": "Medium",
        "company_response_to_consumer": "Closed",
        "special_case_flags": "none",
    }
    value.update(updates)
    return value


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class HarnessChecks(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).parent / "generated"
        scratch.mkdir(exist_ok=True)
        self.root = scratch / ("fixture_" + uuid.uuid4().hex)
        self.root.mkdir()
        self.source = self.root / "Development.csv"
        self.rows = [
            row("DEV-00001", "1", "a disputed charge"),
            row("DEV-00002", "2", "a  disputed\ncharge", label_status="Insufficient Evidence"),
            row("DEV-00003", "3", "card stolen", routing_destination="Card Fraud & Security", fraud_indicator="Yes"),
            row("DEV-00004", "4", "merchant refund"),
            row("DEV-00005", "5", "different product", product="Prepaid card"),
        ]
        write_csv(self.source, self.rows)

    def test_projection_and_duplicate_safe_split(self):
        summary = prepare(self.source, self.root / "run", fraction=0.5)
        self.assertEqual(summary["scope_rows"], 4)
        model_rows = [json.loads(line) for line in (self.root / "run" / "model_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(tuple(value) == INPUT_KEYS for value in model_rows))
        self.assertTrue(all("routing_destination" not in value and "company_response_to_consumer" not in value for value in model_rows))
        partitions = split_map(self.rows, 0.5)
        self.assertEqual(partitions["DEV-00001"], partitions["DEV-00002"])
        self.assertEqual(split_map(self.rows, 0.5), split_map(self.rows, 0.5))

    def test_reject_heldout_row_even_when_out_of_scope(self):
        self.rows[4].update(course_record_id="HEL-00005", dataset_split="Heldout Evaluation")
        write_csv(self.source, self.rows)
        with self.assertRaisesRegex(ValueError, "Non-Development"):
            load_development(self.source)

    def test_score_only_validation_and_check_source_hash(self):
        prepare(self.source, self.root / "run", fraction=0.5)
        model_rows = [json.loads(line) for line in (self.root / "run" / "model_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
        partitions = split_map(self.rows, 0.5)
        validation = next(value for value in model_rows if partitions[value["course_record_id"]] == "internal_validation")
        prediction = self.root / "predictions.jsonl"
        prediction.write_text(json.dumps({
            "course_record_id": validation["course_record_id"],
            "routing_destination": "Card Billing Disputes",
            "escalation_required": "No",
            "human_review_required": "No",
        }) + "\n", encoding="utf-8")
        result = score(self.source, self.root / "run", prediction)
        self.assertEqual(result["n_predictions"], 1)
        train = next(value for value in model_rows if partitions[value["course_record_id"]] == "train_candidate")
        prediction.write_text(json.dumps({
            "course_record_id": train["course_record_id"],
            "routing_destination": "Card Billing Disputes",
            "escalation_required": "No",
            "human_review_required": "No",
        }) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "internal_validation"):
            score(self.source, self.root / "run", prediction)
        self.rows[0]["consumer_complaint_narrative"] = "edited"
        write_csv(self.source, self.rows)
        with self.assertRaisesRegex(ValueError, "hash differs"):
            score(self.source, self.root / "run", prediction)

    def test_reject_heldout_filename(self):
        other = self.root / "Heldout_Evaluation.csv"
        write_csv(other, self.rows)
        with self.assertRaisesRegex(ValueError, "Held-out file name"):
            load_development(other)

    def test_mock_service_receives_narrative_only_and_strict_output(self):
        prepare(self.source, self.root / "run", fraction=0.5)
        payloads = []
        def fake_open(request, timeout):
            payloads.append(json.loads(request.data))
            content = json.dumps({
                "routing_destination": "Card Billing Disputes",
                "escalation_required": "No",
                "human_review_required": "Yes",
                "summary": "A disputed charge was reported.",
                "rationale": "Employee should inspect the record.",
                "uncertainty": "The narrative is incomplete.",
            })
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())
        with patch("model_runner.urlopen", side_effect=fake_open):
            evidence = run(self.root / "run", "http://127.0.0.1/mock", "mock-model",
                           self.root / "mock_output", limit=1)
        self.assertEqual(evidence["valid_json_predictions"], 1)
        self.assertEqual(set(payloads[0]["messages"][1]), {"role", "content"})
        self.assertEqual(payloads[0]["messages"][1]["role"], "user")
        self.assertNotIn("label_status", json.dumps(payloads[0]))
        with self.assertRaises(json.JSONDecodeError):
            parse_model_content("```json\n{}\n```")


if __name__ == "__main__":
    unittest.main()
