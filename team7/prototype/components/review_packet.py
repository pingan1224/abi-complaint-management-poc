"""Offline candidate review contract. No model calls, authentication or bank actions."""
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

CATALOG = json.loads(Path(__file__).with_name("action_catalog.json").read_text(encoding="utf-8"))
SCHEMA = json.loads(Path(__file__).with_name("proposal.schema.json").read_text(encoding="utf-8"))

class ContractError(ValueError):
    pass

def _check(value, schema, path="$ "):
    types = {"object": dict, "array": list, "string": str, "integer": int,
             "boolean": bool, "null": type(None)}
    allowed = schema.get("type", [])
    allowed = allowed if isinstance(allowed, list) else [allowed]
    if allowed and not any(type(value) is types[t] for t in allowed):
        raise ContractError(path + "invalid type")
    if "enum" in schema and not any(type(value) is type(v) and value == v for v in schema["enum"]):
        raise ContractError(path + "invalid enum")
    if type(value) is dict:
        props = schema.get("properties", {})
        if set(schema.get("required", [])) - value.keys():
            raise ContractError(path + "missing field")
        if schema.get("additionalProperties") is False and value.keys() - props.keys():
            raise ContractError(path + "unexpected field")
        for key, item in value.items():
            if key in props:
                _check(item, props[key], path + key + ".")
    if type(value) is list:
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 100000):
            raise ContractError(path + "array length")
        if schema.get("uniqueItems") and len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
            raise ContractError(path + "duplicate item")
        for item in value:
            _check(item, schema["items"], path + "[].")
    if type(value) is str and not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 100000):
        raise ContractError(path + "string length")
    if type(value) is int and value < schema.get("minimum", -100000000):
        raise ContractError(path + "below minimum")

def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ContractError("duplicate JSON key")
        result[key] = value
    return result

def strict_json(text):
    def bad_constant(value):
        raise ContractError("non-finite JSON value")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=bad_constant)

def _digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def validate_proposal(narrative, proposal):
    _check(proposal, SCHEMA)
    for claim in proposal["summary_claims"]:
        if not (0 <= claim["start"] < claim["end"] <= len(narrative)):
            raise ContractError("invalid source span")
        if narrative[claim["start"]:claim["end"]] != claim["quote"]:
            raise ContractError("source quotation mismatch")
    # A matching quote proves location, not that the paraphrase is semantically faithful.
    return copy.deepcopy(proposal)

def prepare_packet(case, model_json=None, service_status="ok"):
    """Accept intake-only fields; turn invalid/unavailable model results into manual review."""
    if type(case) is not dict or set(case) != {"case_id", "narrative"}:
        raise ContractError("intake fields must be exactly case_id and narrative")
    if any(type(case[k]) is not str or not case[k].strip() for k in case):
        raise ContractError("empty or invalid intake")
    if len(case["case_id"]) > 100 or len(case["narrative"]) > 20000:
        raise ContractError("intake size limit")
    if service_status not in {"ok", "timeout", "unavailable"}:
        raise ContractError("unknown service status")
    issues, proposal = [], None
    # Classroom screening examples only. Misses, negation and false positives remain possible.
    hits = [phrase for phrase in ("unauthorized", "not my transaction", "not my purchase")
            if re.search(r"\b" + re.escape(phrase) + r"\b", case["narrative"], re.I)]
    if service_status != "ok":
        issues.append("model_" + service_status)
    else:
        try:
            proposal = validate_proposal(case["narrative"], strict_json(model_json))
        except (ValueError, TypeError, RecursionError) as exc:
            issues.append("invalid_model_output")
    if hits:
        issues.append("raw_text_screening_hit")
    if proposal and (proposal["missing_information"] or proposal["routing_destination"] == "Unknown"
                     or proposal["urgency_level"] == "Unknown" or proposal["escalation_required"] is None):
        issues.append("uncertainty_requires_review")
    if proposal is None:
        allowed = ["MANUAL_REVIEW"]
    elif hits or proposal["escalation_required"] is True or proposal["urgency_level"] in {"High", "Critical"} or proposal["routing_destination"] == "Card Fraud & Security":
        allowed = ["PREPARE_SPECIALIST_REVIEW", "MANUAL_REVIEW"]
    elif "uncertainty_requires_review" in issues:
        allowed = ["REQUEST_INFORMATION", "MANUAL_REVIEW"]
    else:
        allowed = list(CATALOG["actions"])
    packet = {"contract_version": "ass7-v1", "case": copy.deepcopy(case),
              "proposal": proposal, "review_reasons": issues, "raw_screening_hits": hits,
              "catalog_version": CATALOG["version"], "allowed_action_ids": allowed,
              "state": "PENDING_EMPLOYEE_REVIEW", "operational_execution": False}
    packet["packet_sha256"] = _digest(packet)
    return packet

def record_review(packet, *, reviewer_id, decision, action_id, reason,
                  original_checked=False, semantic_support_checked=False):
    """Record a simulation decision; caller must provide real identity/authorization in integration."""
    body = {k: v for k, v in packet.items() if k != "packet_sha256"}
    if packet.get("packet_sha256") != _digest(body):
        raise ContractError("packet changed; prepare a new packet and review again")
    if packet["state"] != "PENDING_EMPLOYEE_REVIEW":
        raise ContractError("packet not pending review")
    if type(reviewer_id) is not str or not reviewer_id.strip():
        raise ContractError("reviewer required")
    if type(reason) is not str or not reason.strip():
        raise ContractError("review reason required")
    if decision not in {"accept", "modify", "reject"}:
        raise ContractError("invalid decision")
    if original_checked is not True or semantic_support_checked is not True:
        raise ContractError("explicit original-text and semantic checks required")
    if decision == "reject":
        if action_id != "MANUAL_REVIEW":
            raise ContractError("rejection must preserve manual handling")
    elif action_id not in packet["allowed_action_ids"]:
        raise ContractError("action prohibited by candidate control")
    if decision == "accept" and (packet["proposal"] is None or action_id != packet["proposal"]["next_action_id"]):
        raise ContractError("accept must match valid suggestion; use modify or reject")
    return {"record_type": "SIMULATED_REVIEW_RECORD", "case_id": packet["case"]["case_id"],
            "packet_sha256": packet["packet_sha256"], "catalog_version": packet["catalog_version"],
            "reviewer_id": reviewer_id.strip(), "decision": decision, "action_id": action_id,
            "reason": reason.strip(), "original_checked": True, "semantic_support_checked": True,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "state": "RECORDED_FOR_SIMULATED_HANDOFF", "operational_execution": False}
