# Team 5 local verification record

Date: 7 October 2026. Operator: Yuchen Yuan with AI-assisted code preparation. Source: original 8,000-row Development CSV, SHA-256 `edc95eb03b97681c8c7fce63163b29608751cd20260593e01a9b6a188c743a53`.

Commands executed from repository root with bundled Python:

```text
python -B team5/test_data_harness.py
python -B team5/data_harness.py prepare <Development CSV> <local run folder>
```

Results: five fixture tests passed. The real Development run validated 8,000 rows and selected 653 records. It exported 653 model input objects, each with exactly `course_record_id` and `consumer_complaint_narrative`; every narrative was nonblank. The separate manifest assigned 541 train candidate and 112 internal validation records. No whitespace-normalized duplicate narrative group crossed those partitions. The SHA-256 values of the exported inputs and manifest are recorded in `development_run_manifest.json`.

Route counts were 513 Billing and 140 Fraud & Security. Label status counts were 401 Clear, 237 Judgment Required, and 15 Insufficient Evidence. In internal validation, route counts were 88 Billing and 24 Fraud & Security; only 2 cases were Insufficient Evidence. This is an unstratified candidate split, so thin rare-case coverage remains a material limitation.

The model runner was exercised with an isolated local mock endpoint: the captured request contained only the narrative as the user message; one strict JSON response was accepted; a fenced JSON example was rejected by the parser. **No DGX execution, real model output, or formal score is claimed.** The held-out CSV was not opened.
