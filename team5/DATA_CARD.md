# Group 2 data card candidate

**Decision status:** proposed synthesis for Team Exercise 5; the team has not yet ratified field boundaries, label rules, or final fitness judgment. Prepared 7 October 2026 from the course Data Dictionary, Student Data Briefing, Development CSV, and the three members' Assignment 5 branches.

## Purpose and provenance

This data supports a classroom, employee-facing, shadow-mode PoC for web-submitted credit-card billing disputes already admitted to that workflow. The consumer narratives and source descriptors come from a real consumer complaint corpus, **not** the case bank's own complaint data. Routing, escalation, review, urgency, fraud indicator, label status, and other operating labels were added for instruction. They are provisional references, not verified bank policy or ground truth. Every employee recommendation needs human confirmation, override, rejection, or deferral.

The course provides 8,000 Development rows and a separate 1,500-row held-out file. Only Development was opened for this package. The exact Development CSV SHA-256 is `edc95eb03b97681c8c7fce63163b29608751cd20260593e01a9b6a188c743a53`.

## Scope and approved input proposal

Offline selection: `product=Credit card`, `issue=Billing disputes`, `submitted_via=Web`, producing 653 Development cases. `issue` is a source classification used to select the research cohort; the PoC has not proved it can identify in-scope cases at live intake. Source `product` and `submitted_via` are scope checks, not model prompt features. `sub_product` and `sub_issue` are blank in all 653 scoped records.

| Modeled task | Model input | Withheld reference or assessment |
|---|---|---|
| Two-way route suggestion | Original `consumer_complaint_narrative` | `routing_destination`: Card Billing Disputes or Card Fraud & Security |
| Escalation suggestion | Same narrative | `escalation_required`; synthetic and not binding policy |
| Additional specialist-review suggestion | Same narrative | `human_review_required`; meaning disputed and must be adjudicated before authoritative scoring |
| Factual summary and rationale | Same narrative | Compare against original text for omissions or inventions; `reference_summary` is deterministic instructional text, not a human gold summary |
| Uncertainty communication | Same narrative | Qualitative review; `label_status` and `label_confidence` are evaluator metadata for stratification, not inputs or trusted uncertainty targets |

The model-facing JSONL contains only `course_record_id` for local matching and the original narrative. The runner sends **only the narrative** to the model. The split manifest remains separate. Do not feed `routing_destination`, `recommended_next_action`, `escalation_required`, `human_review_required`, `fraud_indicator`, `regulatory_indicator`, `urgency_level`, `severity_level`, `customer_harm_level`, `reference_summary`, or any other course-created answer/surrogate into the prompt. `label_status`, `label_confidence`, `special_case_flags`, `word_count`, and `label_provenance` are evaluator metadata. `case_id`, `course_record_id`, `dataset_split`, and `intended_use` are administration fields. `company_response_to_consumer`, `timely_response`, and `consumer_disputed` are post-outcome fields and would create temporal leakage at intake. `date_received` is contemporary context but excluded from this model input because its operational role has not been justified.

## Observed Development quality and coverage

| Finding | Evidence in 653 scoped cases | Consequence |
|---|---:|---|
| Route imbalance | Billing 513; Fraud & Security 140 | Report route-specific recall and consequential misses; overall accuracy alone is misleading. |
| Uncertain instructional labels | Clear 401; Judgment Required 237; Insufficient Evidence 15 | 252/653 are Medium or Low confidence. Reference agreement does not settle ambiguity. |
| Review-status contradiction | 8/15 Insufficient Evidence and 90/237 Judgment Required cases say `human_review_required=No` | Define employee approval versus extra specialist review before scoring; unclear cases should visibly defer. |
| Fraud/route tension | 167 have `fraud_indicator=Yes`; 27 of those still have Billing route | Inspect unauthorized card use separately from merchant disputes; these are review candidates, not automatically wrong labels. |
| Missingness and urgency coverage | `sub_product` and `sub_issue` blank in all 653; Low urgency 0; Medium 381, High 171, Critical 101 | Cannot test subtype logic or Low-urgency behavior in this cohort. |

The source records in scope are historical Web complaints dated 2015–2017. One whitespace-normalized narrative is duplicated in the scoped cohort (`DEV-03200`/`DEV-03315`); the split routine groups identical normalized text across **all** Development records before cohort selection. Yinuo's full Development audit flagged 7 exact duplicate narrative groups (14 rows). Near duplicates and cross-file similarity have not been assessed; held-out was not opened. The generated internal split has 541 candidate training and 112 validation records. Only 2 Insufficient Evidence cases landed in validation, so this split is thin for that issue.

## Inspected cases and interpretation

Yuchen read `DEV-01874`, `DEV-02020`, and `DEV-02074`: each is marked Insufficient Evidence while its synthetic human-review label is No. These examples expose a definition conflict that aggregate counts alone hide. `DEV-00807` describes a disputed purchase and alleged merchant scam; Billing could be defensible despite a fraud flag. `DEV-02283` describes a stolen card and use; `DEV-02550` denies charges and mentions identity theft. Both have Billing routes and warrant specialist adjudication. These descriptions are analyst summaries of directly inspected Development narratives. They are not label corrections or expert findings.

## Fitness and valid claims

**Candidate judgment: fit with controls for exploratory Development testing; insufficient as standalone safety, bank-routing, or business-impact evidence.** The most constraining issue is the unresolved semantics and reliability of synthetic review labels for insufficient-evidence cases. The team should define review layers, obtain two independent specialist labels on consequence-stratified cases, adjudicate disagreements without silently rewriting the source, and compare against a practical non-AI workflow. Current data can establish whether the code respects the Development boundary and how a PoC agrees with course references on this historical cohort. It cannot establish current bank prevalence, correct bank policy, actual handling time, employee behavior, security adequacy, or deployment readiness.

The team must ratify or revise this candidate judgment and record any dissent in the management briefing.
