# Group 2 proposed labeling and taxonomy guide

**Status: proposed, pending team discussion and qualified reviewer adjudication.** This guide makes conflicting reference labels visible. It does not rewrite the course CSV or assert bank policy.

## Operational terms to settle

1. **Employee confirmation:** every PoC output is a suggestion and requires a person to confirm, override, reject, or defer. This is a workflow control, not the synthetic `human_review_required` target.
2. **Additional specialist review:** proposed interpretation of `human_review_required=Yes`, above ordinary employee confirmation. The team must approve this interpretation before using the field for formal scoring.
3. **Escalation:** a time-sensitive or consequential handoff beyond routine review. This can overlap with specialist review but is not identical to it.
4. **Insufficient Evidence:** do not force a confident destination from sparse facts. Show uncertainty and defer to a person; never let a synthetic `No` review label suppress ordinary employee approval or a justified safety handoff.

## Provisional two-route taxonomy

| Narrative signal | Candidate route / action | Evidence still needed |
|---|---|---|
| Merchant billing error, refund, service or charge dispute without clear unauthorized account/card use | Card Billing Disputes | Employee verifies merchant and transaction context. |
| Stolen card, unauthorized card or account use, identity takeover, or compromised credentials | Card Fraud & Security with appropriate escalation | Specialist validates whether the language actually describes unauthorized use. |
| Mixed merchant scam and unauthorized-use language, sparse narrative, or contradictory facts | Defer / specialist review; display uncertainty | Do not infer a confirmed fraud finding. Preserve both candidate interpretations. |

`fraud_indicator=Yes` is a course-generated screening signal, not a routing override. It may mark a merchant dispute. Disagreement with `routing_destination` is an adjudication candidate, not an automatic error.

## Adjudication record

For each consequential disputed Development case, retain the original ID and narrative locally; have two reviewers independently record proposed route, escalation, additional specialist review, confidence, and a short text-grounded reason. Record agreement and a third-party resolution for material disagreement. Preserve the unmodified course label, the adjudicated label, reviewer roles, date, and rationale as separate fields. If evidence remains insufficient, mark `defer/abstain` and exclude it from a forced-choice accuracy denominator while reporting the exclusion count and safety handling separately.

Priority review queue from Yuchen's Assignment 5 candidate: 8 `INSUFFICIENT_NO_HUMAN_REVIEW`, 90 `JUDGMENT_NO_HUMAN_REVIEW`, and 27 `FRAUD_SIGNAL_BILLING_ROUTE` IDs, 125 unique cases. The queue does not establish which course labels are wrong. The team should record which cases it actually inspected during the workshop and any dissent.
