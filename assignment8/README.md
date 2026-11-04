# Assignment 8 candidate independent risk gate

High-consequence failure: an independently recorded suspected fraud/security risk is silently missed by the model, which returns a routine billing route and `No` for both review flags. The Assignment 7 gate only reacts to risk present in the model suggestion, so that false negative would reach ordinary employee review.

`apply_independent_risk_gate` accepts a **trusted, separately obtained** intake risk status. A `suspected_fraud` status suppresses the model route and requires specialist review even when the model says routine or is unavailable. Missing or invalid status stops AI routing and requires manual triage. Unverified scope remains manual intake. A verified `clear` status passes through the A7 authority gate. Every result keeps `final_action_authorized=False`.

The caller must establish the risk signal independently of the model and define its owner, provenance, freshness, and audit record. This function cannot validate provenance, classify complaint narratives, detect risk missed by both intake and model, enforce that a specialist queue exists, or persist an audit log. Those are production integration requirements. It is candidate code for team review, not a merged team control.

Run synthetic tests from the repository root:

```bash
python -B -m unittest discover -s assignment8 -p 'test_*.py' -v
```

No customer data or live model service is needed for this deterministic control test. DGX execution status is documented separately when available.
