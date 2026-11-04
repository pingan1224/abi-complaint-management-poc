# Assignment 8 DGX deterministic control test

**Observed:** 3 November 2026, approximately 19:53 Eastern / 4 November 2026, 00:53 UTC, in the student's NYU AI Lab JupyterLab terminal on `spark-1338`.

**Fixed source:** `candidate/assignment8-ai-control`, commit `cce5f048d5f119c1f6089359ecc6015ebe554881`.

The branch was freshly cloned from GitHub into `~/a8-control-check`. The terminal printed the full commit hash, then ran:

```bash
python3 -B -m unittest discover -s assignment8 -p 'test_*.py' -v
```

Observed result: five tests reported `ok`; `Ran 5 tests in 0.000s`; `OK`. The tests cover a silent model miss with an independent suspected-fraud flag, a model outage with that flag, absent or invalid risk signal, unverified scope, and a clear signal passed through the existing employee-review gate. The run used only synthetic data and the Python standard library.

**Evidence limit:** This demonstrates execution of the fixed candidate control on DGX and the tested queue decisions. It does not establish that a bank-owned independent risk signal exists, is correctly populated, or has valid provenance; that the specialist queue and audit log are wired; or that live model output and real complaint cases are safe. No live model call, customer data, or end-to-end workflow test was performed.
