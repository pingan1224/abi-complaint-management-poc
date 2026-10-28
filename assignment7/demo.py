"""Synthetic offline demonstration; no live inference or operational action."""
import json
from review_packet import prepare_packet, record_review
from test_review_packet import CASE, proposal

def run_demo():
    q = proposal(); q["missing_information"] = ["transaction_amount", "prior_case_reference"]
    q["next_action_id"] = "REQUEST_INFORMATION"
    packet = prepare_packet(CASE, json.dumps(q))
    decision = record_review(packet, reviewer_id="SYNTHETIC-REVIEWER", decision="accept",
        action_id="REQUEST_INFORMATION", reason="Synthetic fixture: collect missing context before assessment.",
        original_checked=True, semantic_support_checked=True)
    return {"evidence_kind":"SYNTHETIC_OFFLINE_FIXTURE", "packet":packet, "review_record":decision}

if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
