"""Synthetic boundary tests for the Assignment 8 candidate control."""

import unittest

from assignment8.independent_risk_gate import apply_independent_risk_gate


def routine_model_suggestion():
    return {
        "routing_destination": "Card Billing Disputes",
        "escalation_required": "No",
        "human_review_required": "No",
        "summary": "Customer disputes a charge.",
        "rationale": "Billing dispute language.",
        "uncertainty": "",
    }


class IndependentRiskGateTests(unittest.TestCase):
    def check_control(self, decision, queue):
        self.assertEqual(decision["queue"], queue)
        self.assertIsNone(decision["suggested_route"])
        self.assertIs(decision["final_action_authorized"], False)
        self.assertIs(decision["control_triggered"], True)

    def test_silent_model_miss_cannot_enter_routine_queue(self):
        decision = apply_independent_risk_gate(
            scope_verified=True,
            risk_status="suspected_fraud",
            model_suggestion=routine_model_suggestion(),
        )
        self.check_control(decision, "specialist_review")
        self.assertIn("independent_fraud_signal", decision["reasons"])

    def test_model_outage_cannot_suppress_independent_signal(self):
        decision = apply_independent_risk_gate(
            scope_verified=True, risk_status="suspected_fraud", model_suggestion=None
        )
        self.check_control(decision, "specialist_review")

    def test_missing_or_invalid_signal_stops_ai_routing(self):
        for status in ("unavailable", "", "No", None):
            with self.subTest(status=status):
                decision = apply_independent_risk_gate(
                    scope_verified=True,
                    risk_status=status,
                    model_suggestion=routine_model_suggestion(),
                )
                self.check_control(decision, "manual_triage")

    def test_unverified_scope_stops_before_risk_or_model(self):
        decision = apply_independent_risk_gate(
            scope_verified=False,
            risk_status="suspected_fraud",
            model_suggestion=routine_model_suggestion(),
        )
        self.check_control(decision, "manual_intake")

    def test_verified_clear_signal_preserves_employee_review(self):
        decision = apply_independent_risk_gate(
            scope_verified=True,
            risk_status="clear",
            model_suggestion=routine_model_suggestion(),
        )
        self.assertEqual(decision["queue"], "employee_review")
        self.assertIs(decision["final_action_authorized"], False)
        self.assertIs(decision["control_triggered"], False)


if __name__ == "__main__":
    unittest.main()
