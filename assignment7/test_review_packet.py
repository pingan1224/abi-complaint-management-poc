import copy
import json
import unittest
from review_packet import ContractError, prepare_packet, record_review

CASE = {"case_id": "SYN-ASS7-001", "narrative": "I disputed the same charge twice. I need a review."}

def proposal(case=CASE):
    quote = case["narrative"].split(".")[0] + "."
    return {"summary_claims": [{"text": "Customer reports repeated dispute attempts.",
            "start": 0, "end": len(quote), "quote": quote}], "missing_information": [],
            "urgency_level": "Medium", "routing_destination": "Card Billing Disputes",
            "escalation_required": False, "next_action_id": "PREPARE_BILLING_REVIEW"}

def packet():
    return prepare_packet(CASE, json.dumps(proposal()))

def review(p, **kwargs):
    args = dict(reviewer_id="SYNTHETIC-REVIEWER", decision="accept", action_id="PREPARE_BILLING_REVIEW",
                reason="Synthetic test fixture reviewed against source.", original_checked=True,
                semantic_support_checked=True)
    args.update(kwargs)
    return record_review(p, **args)

class ContractTests(unittest.TestCase):
    def test_valid_packet_waits_for_employee(self):
        p = packet()
        self.assertEqual(p["state"], "PENDING_EMPLOYEE_REVIEW")
        self.assertFalse(p["operational_execution"])

    def test_accept_is_only_a_simulated_record(self):
        r = review(packet())
        self.assertEqual(r["record_type"], "SIMULATED_REVIEW_RECORD")
        self.assertFalse(r["operational_execution"])

    def test_duplicate_keys_fail_closed(self):
        p = prepare_packet(CASE, '{"urgency_level":"High","urgency_level":"Low"}')
        self.assertEqual(p["allowed_action_ids"], ["MANUAL_REVIEW"])

    def test_timeout_preserves_original_and_manual_path(self):
        p = prepare_packet(CASE, service_status="timeout")
        self.assertEqual(p["case"], CASE)
        self.assertEqual(p["allowed_action_ids"], ["MANUAL_REVIEW"])
        self.assertFalse(review(p, decision="reject", action_id="MANUAL_REVIEW")["operational_execution"])

    def test_bad_quote_fails_closed(self):
        q = proposal(); q["summary_claims"][0]["quote"] = "Invented source"
        self.assertIsNone(prepare_packet(CASE, json.dumps(q))["proposal"])

    def test_boolean_cannot_be_a_span_offset(self):
        q = proposal(); q["summary_claims"][0]["start"] = False
        self.assertIsNone(prepare_packet(CASE, json.dumps(q))["proposal"])

    def test_forbidden_action_cannot_enter_proposal(self):
        q = proposal(); q["next_action_id"] = "REFUND_CUSTOMER"
        self.assertEqual(prepare_packet(CASE, json.dumps(q))["allowed_action_ids"], ["MANUAL_REVIEW"])

    def test_raw_screening_runs_independently_of_model_summary(self):
        case = dict(CASE, narrative=CASE["narrative"] + " This was unauthorized.")
        p = prepare_packet(case, json.dumps(proposal()))
        self.assertIn("raw_text_screening_hit", p["review_reasons"])
        with self.assertRaises(ContractError): review(p)
        self.assertEqual(review(p, decision="modify", action_id="PREPARE_SPECIALIST_REVIEW")["decision"], "modify")

    def test_security_route_cannot_use_routine_billing_action(self):
        q = proposal(); q["routing_destination"] = "Card Fraud & Security"
        p = prepare_packet(CASE, json.dumps(q))
        with self.assertRaises(ContractError): review(p)

    def test_unavailable_model_keeps_manual_path(self):
        p = prepare_packet(CASE, service_status="unavailable")
        self.assertIn("model_unavailable", p["review_reasons"])
        self.assertEqual(p["allowed_action_ids"], ["MANUAL_REVIEW"])

    def test_missing_information_blocks_routine_preparation(self):
        q = proposal(); q["missing_information"] = ["transaction_amount"]
        p = prepare_packet(CASE, json.dumps(q))
        self.assertEqual(p["allowed_action_ids"], ["REQUEST_INFORMATION", "MANUAL_REVIEW"])

    def test_high_risk_cannot_use_routine_action(self):
        q = proposal(); q["urgency_level"] = "High"
        with self.assertRaises(ContractError): review(prepare_packet(CASE, json.dumps(q)))

    def test_review_requires_identity_reason_and_checks(self):
        for args in ({"reviewer_id":""}, {"reason":" "}, {"original_checked":False},
                     {"semantic_support_checked":False}):
            with self.subTest(args=args), self.assertRaises(ContractError): review(packet(), **args)

    def test_modified_packet_cannot_reuse_approval_digest(self):
        p = packet(); p["proposal"]["urgency_level"] = "Low"
        with self.assertRaises(ContractError): review(p)

    def test_leakage_fields_rejected(self):
        with self.assertRaises(ContractError):
            prepare_packet(dict(CASE, company_response_to_consumer="Closed"), json.dumps(proposal()))

    def test_schema_validity_does_not_prove_semantic_support(self):
        q = proposal(); q["summary_claims"][0]["text"] = "The bank has already refunded the customer."
        p = prepare_packet(CASE, json.dumps(q))
        self.assertIsNotNone(p["proposal"])  # Semantic review remains a human duty.
        with self.assertRaises(ContractError): review(p, semantic_support_checked=False)

    def test_unknown_extra_model_fields_rejected(self):
        q = proposal(); q["customer_reply"] = "Your refund is approved."
        self.assertIsNone(prepare_packet(CASE, json.dumps(q))["proposal"])

    def test_non_finite_json_rejected(self):
        self.assertIsNone(prepare_packet(CASE, '{"x":NaN}')["proposal"])

    def test_rejection_cannot_preserve_an_operational_action(self):
        with self.assertRaises(ContractError): review(packet(), decision="reject")

if __name__ == "__main__":
    unittest.main(verbosity=2)
