"""Check recorded model text, not new inference or semantic correctness."""
import json
from pathlib import Path

FIELDS = {"summary", "issue", "urgency", "routing", "escalation", "human_review", "recommended_action"}

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key:" + key)
        result[key] = value
    return result

def reject_constant(value):
    raise ValueError("non_json_constant:" + value)

def check(case):
    result = {"id": case["id"], "prompt_version": case["prompt_version"], "errors": [], "observations": []}
    try:
        data = json.loads(case["raw_response"], object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, RecursionError) as error:
        result["errors"].append("not_strict_json: " + str(error))
        result["status"] = "REJECT"
        return result
    if type(data) is not dict or set(data) != FIELDS:
        result["errors"].append("wrong_field_set")
    else:
        for name in FIELDS - {"escalation", "human_review"}:
            if type(data[name]) is not str or not data[name].strip():
                result["errors"].append("invalid_text:" + name)
        if data["human_review"] is not True:
            result["errors"].append("employee_review_required")
        if case["prompt_version"] == 2:
            allowed = {
                "issue": {"Billing dispute", "Possible unauthorized transaction", "Insufficient information"},
                "urgency": {"Low", "Medium", "High", "Critical", "Unknown"},
                "routing": {"Card Billing Disputes", "Card Fraud and Security", "Manual Triage"},
            }
            for name, values in allowed.items():
                if type(data[name]) is str and data[name] not in values:
                    result["errors"].append("invalid_label:" + name)
            if type(data["escalation"]) is not bool:
                result["errors"].append("escalation_not_boolean")
        elif type(data["escalation"]) is not bool:
            result["observations"].append("String escalation; V1 prompt did not specify its type.")
    result["status"] = "REJECT" if result["errors"] else "FORMAT_PASS"
    return result

def main():
    root = Path(__file__).resolve().parent
    cases = json.loads((root / "responses.json").read_text(encoding="utf-8"))
    results = [check(case) for case in cases]
    for result in results:
        print(result["id"], result["status"])
        for note in result["errors"] + result["observations"]:
            print("  " + note)
    print("Format pass:", sum(r["status"] == "FORMAT_PASS" for r in results), "/", len(results))
    print("FORMAT_PASS does not establish factual accuracy, safe routing, or permission to execute an action.")
    # Keep run evidence separate from candidate source; no raw complaints in report.
    print("\nMachine-readable results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
