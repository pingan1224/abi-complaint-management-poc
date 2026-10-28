# Assignment 7 employee review packet candidate

This bounded contribution turns a model proposal into a source-linked employee review packet and records a simulated human decision. It is an individual proposal for Team Exercise 7, not a shared prototype or an operational approval system.

## Run

Python 3.10 or later, standard library only. From `assignment7/`:

```sh
python -B -m unittest -v test_review_packet
python -B demo.py
```

`prepare_packet(case, model_json=None, service_status="ok")` accepts exactly `case_id` and `narrative`. `model_json` must contain the proposal fields in `proposal.schema.json`. This module does NOT contact any model service. An integration adapter must supply the actual service response and report timeout/unavailability explicitly.

`record_review(packet, reviewer_id=..., decision=..., action_id=..., reason=..., original_checked=True, semantic_support_checked=True)` produces only a simulated review record. It has no routing, messaging, financial, account, refund or case-closure side effects. No free-text model action is executed.

## Boundaries that matter

- `action_catalog.json` contains four illustrative CLASSROOM procedures, version `classroom-actions-v1`; they are proposed for team review. They are not actual bank policy or instructor-approved procedures.
- Exact quotation/span checks establish source location, NOT semantic entailment. A matching quotation can accompany a false paraphrase; a test demonstrates this remaining human duty.
- Raw-text screening checks three illustrative phrases independently of the model summary. It is not a complete fraud detector, approved bank escalation policy or reliable negation handler. It can miss risk and can overflag negated phrases. Positive hits restrict the candidate to specialist/manual preparation; absence is not evidence of safety.
- High/Critical or model-escalation proposals also restrict candidate actions to specialist/manual preparation. Uncertainty restricts routine preparation. Employees can modify to a permitted action or reject to manual review. Overriding the restriction itself is outside this component.
- Reviewer identity and check acknowledgments are supplied by a trusted caller. There is no authentication, role authorization, persistent append-only audit store or signature. The digest detects accidental packet edits between preparation and review; it is NOT an adversarial tamper-proof control.
- Repeated calls may create repeated records; atomic state changes, idempotency and authenticated writes belong in the later workflow service. Nothing here claims an integrated employee interface.
- There is no new summarization model, verified handling quality, bank-policy retrieval, customer dialogue, actual task execution, production deployment or measured business improvement.

## Proposed architecture and handoff

Keep the current web-submitted credit-card billing-dispute boundary. Complete the originally scoped readout with summary, source excerpts, missing information and uncertainty. Present the employee with a bounded next-action recommendation and a named classroom procedure. Ask the team to explicitly decide whether to adopt this extension before integration. Retain employee confirmation of every proposal. Do not adopt the Assignment 6 adapter on the basis of this component.

The interface service must render the original complaint alongside the proposal, authenticate the reviewer, enforce permissions server-side, and persist the original proposal, employee decision, procedure version and timestamp. A later simulated task service may consume a confirmed record after team adoption; it must not infer that a case has been resolved.

## Verification and DGX status

The included local test log and verification manifest identify exactly what ran. Synthetic fixtures test contract behavior and fail-closed controls only, not LLM accuracy or business value. No course Development narrative or held-out record is included in this candidate. The official holdout was not opened for Assignment 7.

DGX execution and live model-service interface checks have NOT been performed for this contribution. No active DGX browser session was available in the inspected browser inventory. The existing Assignment 6 local checkpoint loader is a different interface from a verified serving endpoint. This candidate intentionally makes no model-service communication claim.

To close the target-environment check, pull the recorded branch/commit onto an authorized DGX worktree and run the two commands above with an available Python 3.10+ interpreter, preserving interpreter version, commit and log. Before any live integration, separately confirm service endpoint, request schema, actual model/version, response envelope, timeout behavior, token limits and access control. Then test a controlled authorized Development case; keep real course text and model replies out of this public repository. A successful offline DGX run would still not establish model-service interoperability.

## Three failure walkthroughs

1. A model omits “unauthorized” from its summary and suggests routine handling: the independent raw-text example screen restricts actions; an employee can modify to specialist preparation or reject to manual review. Phrase coverage remains incomplete.
2. A model times out or returns malformed JSON/unsupported fields: preserve the original complaint, discard the proposal and permit only manual review. Never invent a successful model result or silently retry with a new model.
3. A syntactically valid proposal asserts unsupported facts or a prohibited refund: unknown actions fail closed; a plausible but false paraphrase still needs source-based human review. Structural validation cannot certify truth.

## Sources

Course Assignment 7 and General Individual Instructions; Group 2 Team Exercise 2 scope; Team Exercise 6 Model Behavior and Adaptation Decision (October 22, 2026); course What You Are Building and Wells Fargo Patent and PoC briefings; US20250173329A1 paragraph 0023. The architecture is a proposed allocation, not a claim that the entire system is implemented.
