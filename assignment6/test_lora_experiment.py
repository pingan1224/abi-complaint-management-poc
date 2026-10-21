import json
import tempfile
import unittest
from pathlib import Path

from lora_experiment import (ROOT, check_local_32b_model, encode_supervised_sample,
                             load_config, load_examples, preflight)


class BoundaryTests(unittest.TestCase):
    def test_chat_prompt_boundary_preserves_answer_only_loss(self):
        class BoundaryChangingTokenizer:
            eos_token_id = 99

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                self.assert_prompt = (len(messages) == 2 and tokenize and add_generation_prompt)
                if not self.assert_prompt:
                    raise AssertionError("Completed chat must not be tokenized separately")
                return [10, 11]

            def encode(self, text, *, add_special_tokens):
                if text != "REVIEW" or add_special_tokens:
                    raise AssertionError("Expected plain answer tokens without added specials")
                return [20, 21]

        row = {"id": "SYN-T01", "narrative": "A synthetic complaint.", "decision": "REVIEW"}
        item = encode_supervised_sample(BoundaryChangingTokenizer(), row, max_length=5)
        self.assertEqual(item["input_ids"], [10, 11, 20, 21, 99])
        self.assertEqual(item["labels"], [-100, -100, 20, 21, 99])
        self.assertEqual(item["attention_mask"], [1] * 5)
        with self.assertRaisesRegex(ValueError, "too long"):
            encode_supervised_sample(BoundaryChangingTokenizer(), row, max_length=4)

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
