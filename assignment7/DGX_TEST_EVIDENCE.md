# Assignment 7 DGX execution and interface smoke test

**Observed:** 28 October 2026, 06:27–06:29 UTC, through the student's NYU AI Lab JupyterLab terminal on `spark-1338` (Python 3.12.3).

**Fixed sources:**

- Assignment 7 candidate: `7aa6ffc9d99b33a77f3993eac086afc97493dbcd` (`candidate/assignment7-architecture`).
- Team 5 candidate parser: `3eb3cbf5d0d0ea2eec65b422ef7bb3979a75217` (`candidate/team5-data-harness`).

Both branches were freshly cloned into separate directories in the student's DGX home directory. The terminal printed each full commit hash before the interface result was recorded.

## Checks and observed results

1. From the Assignment 7 checkout, `python3 -B -m unittest discover -s assignment7 -p 'test_*.py' -v` completed with `Ran 4 tests` and `OK`. These cover unverified scope, unavailable or malformed model output, routine advice, and consequential signals.
2. A Python process imported `parse_model_content` from the fixed Team 5 candidate and `apply_authority_gate` from the fixed Assignment 7 candidate. The input was a hand-written, six-field synthetic JSON object. The parser accepted it. The gate returned `employee_review` for the routine suggestion, `specialist_review` after an uncertainty statement was added, and `manual_triage` when the model suggestion was `None`. Assertions confirmed `final_action_authorized` was `false` in all three results. The process printed `synthetic_parser_gate_pass`.

No real complaint, course held-out record, customer identifier, API credential, or model output was used in this smoke test.

## Evidence boundary

This establishes that the fixed candidate code executes in the DGX JupyterLab environment and that a synthetic object accepted by the Team 5 parser passes into the authority gate. It does **not** establish a live model-service connection, model response quality, bank policy approval, a functioning specialist queue, or an end-to-end employee workflow. The course chat shortcut identifies a browser chat entry point but does not supply the approved full API endpoint, exact served model ID, or credential method needed for a controlled live request. Those service details must be confirmed before a separate synthetic live-call test.
