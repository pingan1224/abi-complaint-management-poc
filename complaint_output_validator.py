import json

STRING_FIELDS = (
    "summary",
    "issue",
    "urgency",
    "routing",
    "next_action",
)

BOOLEAN_FIELDS = (
    "escalation",
    "human_review",
)


def parse_complaint_output(raw_text):
    """Parse and validate one model-generated complaint result."""
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as error:
        return None, [
            f"Invalid JSON at line {error.lineno}, column {error.colno}: "
            f"{error.msg}"
        ]

    if not isinstance(result, dict):
        return None, ["The JSON value must be an object."]

    errors = []

    for field in STRING_FIELDS:
        if field not in result:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(result[field], str):
            errors.append(f"Field '{field}' must be a string.")
        elif not result[field].strip():
            errors.append(f"Field '{field}' must not be empty.")

    for field in BOOLEAN_FIELDS:
        if field not in result:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(result[field], bool):
            errors.append(f"Field '{field}' must be true or false.")

    if errors:
        return None, errors

    return result, []
