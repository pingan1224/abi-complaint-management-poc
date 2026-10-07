# Assignment 5 candidate: instructional label review queue

This is a bounded data-audit utility for the team's **already admitted** Web / Credit card / Billing disputes Development cohort. It complements the existing duplicate-protected split manifest candidate. It does not decide the correct route or change source labels.

Run from the repository root with Python 3.10 or newer:

```powershell
python -B assignment5/test_label_review_queue.py
python -B assignment5/label_review_queue.py "C:\path\to\ABI_Bank_Complaints_Development_8000.csv" "C:\local\review_queue.csv"
```

The output contains case IDs, existing reference labels, and reason codes only; it omits the complaint narrative. It refuses a Held-out file by name and checks **every** row is Development / DEV- before filtering. It will not overwrite an existing output file. Keep generated queues and the source dataset outside Git.

Reason codes ask for a human decision:

- `INSUFFICIENT_NO_HUMAN_REVIEW`: the course status says Insufficient Evidence while the reference human-review field says No. Resolve this first because an unclear case may appear safe for routine handling.
- `JUDGMENT_NO_HUMAN_REVIEW`: the course status says Judgment Required while the reference human-review field says No.
- `FRAUD_SIGNAL_BILLING_ROUTE`: the course fraud indicator is Yes but the reference destination is Billing. This can be a legitimate merchant dispute or a questionable security route.

The team's workflow requires an employee to confirm, override, reject, or defer **every** recommendation. A `human_review_required = No` course label must never bypass that final human authority. Before using this label as a target, the team must define whether it means final employee confirmation or an additional specialist review.

**Boundary:** The source `issue` column is used only to select this retrospective cohort and is never a model input. A live intake workflow would need a separately validated upstream scope gate. Intended model input is the original `consumer_complaint_narrative`, with contemporaneous product/channel context only if the workflow actually has it. Routing, escalation, human review, fraud indicator, urgency, label status, outcome fields, and course metadata are withheld as references or audit metadata. The two routes are provisional course labels, not approved bank policy. A reviewer should preserve the original narrative and record any label disagreement with a rationale.

This standard-library script performs no model inference and does not require DGX. Its synthetic fixture checks cover the review flags, held-out rejection, and narrative-free output. They do not validate the course labels or prove the PoC safe.
