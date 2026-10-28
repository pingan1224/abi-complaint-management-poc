"""Boundary tests for the Assignment 7 candidate authority gate."""

import unittest

from authority_gate import apply_authority_gate


def suggestion(**changes):
    value = {
        "routing_destination": "Card Billing Disputes",
        "escalation_required": "No",
        "human_review_required": "No",
        "summary": "Customer disputes a merchant charge.",
        "rationale": "Narrative describes a billing dispute.",
        "uncertainty": "",
    }
    value.update(changes)
    return value


class AuthorityGateTests(unittest.TestCase):
    def assert_no_final_action(self, decision):
        self.assertIs(decision["final_action_authorized"], False)

    def test_unverified_scope_blocks_model_recommendation(self):
        decision = apply_authority_gate(scope_verified=False, model_suggestion=suggestion())
        self.assertEqual(decision["queue"], "manual_intake")
        self.assertIsNone(decision["suggested_route"])
        self.assert_no_final_action(decision)

    def test_service_failure_and_invalid_output_fail_to_manual_triage(self):
        for value in (None, {"routing_destination": "Card Billing Disputes"},
                      suggestion(routing_destination="Other"),
                      suggestion(routing_destination=["Card Billing Disputes"]),
                      suggestion(summary=" ")):
            with self.subTest(value=value):
                decision = apply_authority_gate(scope_verified=True, model_suggestion=value)
                self.assertEqual(decision["queue"], "manual_triage")
                self.assertIsNone(decision["suggested_route"])
                self.assert_no_final_action(decision)

    def test_routine_suggestion_still_needs_employee(self):
        decision = apply_authority_gate(scope_verified=True, model_suggestion=suggestion())
        self.assertEqual(decision["queue"], "employee_review")
        self.assert_no_final_action(decision)

    def test_consequential_signals_require_specialist(self):
        cases = (
            {"routing_destination": "Card Fraud & Security"},
            {"escalation_required": "Yes"},
            {"human_review_required": "Yes"},
            {"uncertainty": "Authorization is unclear."},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                decision = apply_authority_gate(
                    scope_verified=True, model_suggestion=suggestion(**changes)
                )
                self.assertEqual(decision["queue"], "specialist_review")
                self.assert_no_final_action(decision)


if __name__ == "__main__":
    unittest.main()
