# Held-out evaluation protection rule

The held-out CSV is reserved for formal evaluation **after** the team freezes the named code commit, model and adapter version, prompt, output schema, deterministic rules, metrics, case definitions, thresholds, and adjudication rubric. This package does not open or score held-out data.

Before the freeze, no team member may use held-out records to train, prompt-tune, select a model or hyperparameter, rewrite a rule, choose examples, or repair particular failures. Development validation is the only iterative evidence here. The `data_harness.py` loader rejects held-out file names and any row whose split/ID is not Development / `DEV-`, even if that row is outside the scoped cohort.

At the later formal evaluation, record the frozen version and dataset hash, run the evaluation once under the agreed plan, and report every failure and denominator. If the team changes the system after viewing held-out outcomes, version the change and keep the original result; the same held-out cases cannot be presented as an untouched independent test of the revised system. Faculty-held unseen cases remain a separate future challenge.
