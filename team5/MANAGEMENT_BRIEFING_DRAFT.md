# Group 2 Team Exercise 5 management briefing draft

**To:** Andres Fortino, Instructor acting as Management
**From:** Group 2 workshop draft prepared by Yuchen Yuan
**Date:** 7 October 2026
**Status:** discussion draft; Yinuo and Lester have not yet approved a joint decision, and no DGX execution of this commit has been recorded.

## Current conclusion and build increment

We propose to treat the Development data as **fit with controls for a bounded classroom PoC**. The new candidate harness can read and validate Development rows, select the 653-case cohort, export only the original narrative for model requests, keep labels and split metadata out of the model prompt, group duplicate narratives before internal splitting, and score only internal validation IDs against provisional references. Five local fixture tests passed. The code also includes an optional configurable chat-completions runner, tested with a local mock. It has not been run against the DGX-hosted model service, so the course requirement for named-commit DGX execution remains open.

## Evidence that affects the decision

- In-scope Development: 653 of 8,000; 513 Billing route and 140 Fraud & Security; 401 Clear, 237 Judgment Required, 15 Insufficient Evidence.
- Yuchen's direct inspection found that `DEV-01874`, `DEV-02020`, and `DEV-02074` are Insufficient Evidence yet say No for human review. Across the cohort, 8 of 15 such cases have this conflict.
- Yinuo's audit identified full-dataset duplicate narratives, target/surrogate leakage risks, missing subfields, and absent Low-urgency cases. Lester's candidate manifest demonstrates duplicate-group internal splitting. This branch combines those boundary lessons, but does not claim to merge or supersede their work.
- The local run produced 541 candidate training and 112 internal validation records; only 2 Insufficient Evidence cases are in validation. The source SHA-256 and split seed are recorded in the generated run manifest.

## Proposed team judgment and dissent to record

The highest-priority labeling decision is whether `human_review_required` means ordinary employee confirmation or **additional specialist review**. Yuchen proposes the latter, while maintaining mandatory employee approval for every output and deferring insufficient-evidence cases. This is a proposal, not a recorded dissent from either teammate. Ask Yinuo and Lester to accept, amend, or dissent, then record the evidence and decision. Also decide whether a merchant scam belongs in Billing while stolen-card or account misuse requires Fraud & Security. Do not silently relabel the source.

## Decision and evidence still required during the team workshop

1. Agree and record the input/target boundary, review semantics, route taxonomy, and how adjudicated labels are stored separately from course labels.
2. Have team members inspect consequential examples and sign off or preserve their disagreement.
3. Choose the shared code version and commit; have a teammate with DGX terminal or service access run that commit against the hosted model, record endpoint/model/configuration, inputs, outputs, parser failures, and timestamp.
4. Freeze the held-out rule in the shared project record and confirm who owns the later formal evaluation path.
5. Name the contributor/operator/submission roles and update this draft with the actual team decision. Do not report local mock execution as DGX evidence.

## AI use and verification

AI assisted with candidate code and draft language. Consequential counts were checked by executing the code on the original Development CSV and comparing with the individual audits; five synthetic fixture tests covered boundary and scoring behavior. Yuchen directly inspected the cited narratives. The course Data Dictionary and Student Data Briefing supplied provenance and leakage definitions. Label validity, SME adjudication, and DGX integration remain unverified.
