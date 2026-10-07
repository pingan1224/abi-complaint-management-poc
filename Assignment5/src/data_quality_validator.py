import argparse
import json
from pathlib import Path

import pandas as pd


LEAKAGE_RULES = [
    (
        "fraud_indicator",
        "routing_destination",
        "Reference field may directly reveal routing destination."
    ),
    (
        "escalation_required",
        "human_review_required",
        "Escalation and human-review labels are directly aligned."
    ),
]


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_required_columns(df, config):
    required = set(
        config["model_inputs"]
        + config["target_fields"]
        + config["reference_fields"]
        + config["excluded_from_predictive_input"]
    )

    missing = sorted(required - set(df.columns))

    return {
        "status": "PASS" if not missing else "FAIL",
        "missing_columns": missing,
    }


def validate_input_target_boundary(config):
    inputs = set(config["model_inputs"])
    protected = set(
        config["target_fields"]
        + config["reference_fields"]
        + config["excluded_from_predictive_input"]
    )

    overlap = sorted(inputs & protected)

    return {
        "status": "PASS" if not overlap else "FAIL",
        "overlap": overlap,
    }


def profile_missingness(df, columns):
    results = {}

    for column in columns:
        if column not in df.columns:
            continue

        missing_count = int(df[column].isna().sum())
        missing_rate = float(df[column].isna().mean())

        results[column] = {
            "missing_count": missing_count,
            "missing_rate": round(missing_rate, 4),
        }

    return results


def check_duplicate_narratives(df):
    column = "consumer_complaint_narrative"

    if column not in df.columns:
        return {
            "status": "NOT_RUN",
            "reason": f"{column} not found"
        }

    duplicated = df[column].duplicated(keep=False)

    return {
        "status": "PASS" if not duplicated.any() else "FLAG",
        "duplicate_rows": int(duplicated.sum()),
        "duplicate_narratives": int(
            df.loc[duplicated, column].nunique()
        ),
    }


def profile_targets(df, target_fields):
    results = {}

    for column in target_fields:
        if column not in df.columns:
            continue

        results[column] = (
            df[column]
            .value_counts(dropna=False)
            .to_dict()
        )

    return results


def check_leakage(df):
    results = []

    for source, target, explanation in LEAKAGE_RULES:
        if source not in df.columns or target not in df.columns:
            continue

        contingency = pd.crosstab(
            df[source],
            df[target],
            dropna=False
        )

        results.append(
            {
                "source_field": source,
                "target_field": target,
                "status": "FLAG",
                "explanation": explanation,
                "contingency_table": contingency.to_dict(),
            }
        )

    return results


def filter_scoped_cohort(df, config):
    scope = config["scope"]

    required_scope_fields = [
        "product",
        "issue",
        "submitted_via",
    ]

    missing = [
        field for field in required_scope_fields
        if field not in df.columns
    ]

    if missing:
        return pd.DataFrame()

    return df[
        (df["product"] == scope["product"]) &
        (df["issue"] == scope["issue"]) &
        (df["submitted_via"] == scope["submitted_via"])
    ].copy()


def audit_scoped_cohort(df, config):
    if df.empty:
        return {
            "status": "NOT_RUN",
            "reason": "Scoped cohort could not be created."
        }

    result = {
        "status": "PASS",
        "rows": int(len(df)),
    }

    if "label_status" in df.columns:
        result["label_status_distribution"] = (
            df["label_status"]
            .value_counts(dropna=False)
            .to_dict()
        )

    if "urgency_level" in df.columns:
        result["urgency_distribution"] = (
            df["urgency_level"]
            .value_counts(dropna=False)
            .to_dict()
        )

    for field in ["sub_product", "sub_issue"]:
        if field in df.columns:
            result[f"{field}_missing"] = {
                "count": int(df[field].isna().sum()),
                "rate": round(
                    float(df[field].isna().mean()),
                    4
                ),
            }

    if (
        "escalation_required" in df.columns
        and "human_review_required" in df.columns
    ):
        result["escalation_human_review_crosstab"] = (
            pd.crosstab(
                df["escalation_required"],
                df["human_review_required"],
                dropna=False
            ).to_dict()
        )

    if (
        "fraud_indicator" in df.columns
        and "routing_destination" in df.columns
    ):
        result["fraud_routing_crosstab"] = (
            pd.crosstab(
                df["fraud_indicator"],
                df["routing_destination"],
                dropna=False
            ).to_dict()
        )

    return result


def build_audit(df, config):
    required_check = validate_required_columns(df, config)
    boundary_check = validate_input_target_boundary(config)

    profile_columns = list(
        dict.fromkeys(
            config["model_inputs"]
            + config["target_fields"]
            + config["reference_fields"]
            + [
                "label_status",
                "label_confidence",
                "urgency_level",
                "sub_product",
                "sub_issue",
            ]
        )
    )

    scoped = filter_scoped_cohort(df, config)

    return {
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "required_columns_check": required_check,
        "input_target_boundary_check": boundary_check,
        "missingness": profile_missingness(
            df,
            profile_columns
        ),
        "duplicate_narrative_check": check_duplicate_narratives(df),
        "target_distribution": profile_targets(
            df,
            config["target_fields"]
        ),
        "label_status_distribution": (
            df["label_status"]
            .value_counts(dropna=False)
            .to_dict()
            if "label_status" in df.columns
            else {}
        ),
        "leakage_checks": check_leakage(df),
        "scoped_cohort_audit": audit_scoped_cohort(
            scoped,
            config
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Assignment 5 Development Dataset quality validator."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to Development Dataset CSV."
    )

    parser.add_argument(
        "--config",
        default="Assignment5/config/schema_config.json",
        help="Path to schema configuration JSON."
    )

    parser.add_argument(
        "--output",
        default="Assignment5/evidence/audit_results.json",
        help="Path for JSON audit output."
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    config_path = Path(args.config)
    output_path = Path(args.output)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    df = pd.read_csv(data_path)
    config = load_config(config_path)

    audit = build_audit(df, config)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            audit,
            f,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    scoped = filter_scoped_cohort(df, config)

    print("Assignment 5 Data Quality Validator")
    print("-----------------------------------")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(
        "Required columns:",
        audit["required_columns_check"]["status"]
    )
    print(
        "Input/target boundary:",
        audit["input_target_boundary_check"]["status"]
    )
    print(
        "Duplicate narrative check:",
        audit["duplicate_narrative_check"]["status"]
    )

    print("\nScoped Cohort")
    print("-------------")
    print(f"Rows: {len(scoped)}")

    if not scoped.empty:
        print("\nLabel Status:")
        print(scoped["label_status"].value_counts())

        print("\nUrgency:")
        print(scoped["urgency_level"].value_counts())

        print("\nMissingness:")
        print(
            "sub_product:",
            scoped["sub_product"].isna().sum(),
            "/",
            len(scoped)
        )
        print(
            "sub_issue:",
            scoped["sub_issue"].isna().sum(),
            "/",
            len(scoped)
        )

    print(f"\nAudit saved to: {output_path}")


if __name__ == "__main__":
    main()
