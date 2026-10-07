import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Assignment5.src.data_quality_validator import (
    validate_input_target_boundary,
    validate_required_columns,
)


CONFIG_PATH = Path("Assignment5/config/schema_config.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_input_target_boundary_passes():
    config = load_config()

    result = validate_input_target_boundary(config)

    assert result["status"] == "PASS"
    assert result["overlap"] == []


def test_required_columns_are_detected():
    config = load_config()

    columns = set(
        config["model_inputs"]
        + config["target_fields"]
        + config["reference_fields"]
        + config["excluded_from_predictive_input"]
    )

    df = pd.DataFrame(
        {column: ["test"] for column in columns}
    )

    result = validate_required_columns(df, config)

    assert result["status"] == "PASS"
    assert result["missing_columns"] == []


def test_missing_required_column_is_detected():
    config = load_config()

    columns = set(
        config["model_inputs"]
        + config["target_fields"]
        + config["reference_fields"]
        + config["excluded_from_predictive_input"]
    )

    columns.remove("consumer_complaint_narrative")

    df = pd.DataFrame(
        {column: ["test"] for column in columns}
    )

    result = validate_required_columns(df, config)

    assert result["status"] == "FAIL"
    assert "consumer_complaint_narrative" in result["missing_columns"]


def test_boundary_detects_leakage_configuration():
    config = load_config()

    bad_config = dict(config)
    bad_config["model_inputs"] = [
        "consumer_complaint_narrative",
        "routing_destination",
    ]

    result = validate_input_target_boundary(bad_config)

    assert result["status"] == "FAIL"
    assert "routing_destination" in result["overlap"]


def run_tests():
    tests = [
        test_input_target_boundary_passes,
        test_required_columns_are_detected,
        test_missing_required_column_is_detected,
        test_boundary_detects_leakage_configuration,
    ]

    passed = 0

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
        passed += 1

    print()
    print(f"{passed}/{len(tests)} tests passed.")


if __name__ == "__main__":
    run_tests()
