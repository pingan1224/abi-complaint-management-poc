"""Candidate human-authority gate for a bounded complaint-management PoC.

This pure function accepts a parsed model suggestion. It never authorizes a
complaint disposition, customer communication, or case closure. The caller must
verify live intake scope independently and record the employee's actual action.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ROUTES = frozenset({"Card Billing Disputes", "Card Fraud & Security"})
MODEL_FIELDS = frozenset({
    "routing_destination", "escalation_required", "human_review_required",
    "summary", "rationale", "uncertainty",
})


def _decision(queue: str, route: str | None, *reasons: str) -> dict[str, Any]:
    return {
        "queue": queue,
        "suggested_route": route,
        "reasons": list(reasons),
        "final_action_authorized": False,
    }


def apply_authority_gate(
    *, scope_verified: bool, model_suggestion: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return a review queue, never a final case action.

    ``scope_verified`` must come from an independent intake check, not a model
    claim or a retrospective source label. ``None`` represents a failed or
    unavailable model call. No narrative or customer data is returned.
    """
    if type(scope_verified) is not bool:
        raise ValueError("scope_verified must be a boolean from the intake gate")
    if not scope_verified:
        return _decision("manual_intake", None, "scope_unverified")
    if model_suggestion is None:
        return _decision("manual_triage", None, "model_unavailable")
    if not isinstance(model_suggestion, Mapping) or set(model_suggestion) != MODEL_FIELDS:
        return _decision("manual_triage", None, "invalid_model_schema")

    route = model_suggestion["routing_destination"]
    if not isinstance(route, str) or route not in ROUTES:
        return _decision("manual_triage", None, "invalid_model_route")
    for field in ("escalation_required", "human_review_required"):
        if model_suggestion[field] not in ("Yes", "No"):
            return _decision("manual_triage", None, "invalid_model_flag")
    for field in ("summary", "rationale", "uncertainty"):
        if not isinstance(model_suggestion[field], str):
            return _decision("manual_triage", None, "invalid_model_text")
    if not model_suggestion["summary"].strip() or not model_suggestion["rationale"].strip():
        return _decision("manual_triage", None, "missing_model_explanation")

    reasons = []
    if route == "Card Fraud & Security":
        reasons.append("fraud_security_route_suggested")
    if model_suggestion["escalation_required"] == "Yes":
        reasons.append("escalation_suggested")
    if model_suggestion["human_review_required"] == "Yes":
        reasons.append("specialist_review_suggested")
    if model_suggestion["uncertainty"].strip():
        reasons.append("uncertainty_stated")
    if reasons:
        return _decision("specialist_review", route, *reasons)
    return _decision("employee_review", route, "employee_confirmation_required")
