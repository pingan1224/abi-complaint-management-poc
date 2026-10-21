import json
import tempfile
import unittest
from pathlib import Path

from lora_experiment import ROOT, check_local_32b_model, load_config, load_examples, preflight


class BoundaryTests(unittest.TestCase):
    def test_frozen_synthetic_input_passes(self):
        _, rows, summary = preflight(ROOT / "experiment.json", ROOT / "synthetic_examples.jsonl")
        self.assertEqual((summary["train_count"], summary["eval_count"]), (4, 4))
        self.assertTrue(all(row["id"].startswith("SYN-") for row in rows))

    def test_target_or_evaluator_field_is_rejected(self):
        rows = [json.loads(s) for s in (ROOT / "synthetic_examples.jsonl").read_text(encoding="utf-8").splitlines()]
        rows[0]["label_status"] = "Insufficient Evidence"
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected or missing fields"):
                load_examples(path)

    def test_duplicate_across_splits_is_rejected(self):
        rows = [json.loads(s) for s in (ROOT / "synthetic_examples.jsonl").read_text(encoding="utf-8").splitlines()]
        rows[-1]["narrative"] = rows[0]["narrative"].upper()
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate narrative"):
                load_examples(path)

    def test_quantization_or_changed_target_is_not_silent(self):
        config = json.loads((ROOT / "experiment.json").read_text(encoding="utf-8"))
        config["load_in_4bit"] = True
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Configuration keys differ"):
                load_config(path)

    def test_eight_billion_parameter_model_config_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"model_type": "qwen2", "hidden_size": 3584,
                                        "num_hidden_layers": 28}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not the expected Qwen 2.5 32B"):
                check_local_32b_model(Path(directory))

    def test_unsupported_quantization_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"model_type": "qwen2", "hidden_size": 5120,
                                        "num_hidden_layers": 64,
                                        "quantization_config": {"load_in_4bit": True}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Expected the approved compressed-tensors FP8"):
                check_local_32b_model(Path(directory))

    def test_approved_fp8_architecture_is_accepted(self):
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "model_type": "qwen2", "hidden_size": 5120, "num_hidden_layers": 64,
                "quantization_config": {
                    "quant_method": "compressed-tensors", "format": "float-quantized",
                    "config_groups": {"group_0": {
                        "targets": ["Linear"], "weights": {"num_bits": 8, "type": "float"},
                    }},
                },
            }), encoding="utf-8")
            self.assertEqual(len(check_local_32b_model(Path(directory))), 64)


if __name__ == "__main__":
    unittest.main()
