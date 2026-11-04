"""Candidate control for a model's silent miss of an independently flagged risk.

The upstream intake service, not the model, must supply ``risk_status``. This
function cannot verify that provenance or discover risks absent from both
sources. It never authorizes a case disposition or customer communication.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from assignment7.authority_gate import apply_authority_gate


def apply_independent_risk_gate(
    *,
    scope_verified: bool,
    risk_status: str,
    model_suggestion: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Combine an intake risk signal with the A7 advisory authority gate.

    ``risk_status`` is ``suspected_fraud``, ``clear``, or ``unavailable``.
    Any other value is treated as unavailable. No narrative or identifiers
    enter the returned control result.
    """
    if type(scope_verified) is not bool:
        raise ValueError("scope_verified must be a boolean from the intake gate")
    if not scope_verified:
        return {
            "queue": "manual_intake",
            "suggested_route": None,
            "reasons": ["scope_unverified"],
            "final_action_authorized": False,
            "control_triggered": True,
        }
    if risk_status == "suspected_fraud":
        return {
            "queue": "specialist_review",
            "suggested_route": None,
            "reasons": ["independent_fraud_signal"],
            "final_action_authorized": False,
            "control_triggered": True,
        }
    if risk_status != "clear":
        return {
            "queue": "manual_triage",
            "suggested_route": None,
            "reasons": ["independent_risk_signal_unavailable"],
            "final_action_authorized": False,
            "control_triggered": True,
        }
    decision = apply_authority_gate(
        scope_verified=True, model_suggestion=model_suggestion
    )
    return {**decision, "control_triggered": False}
