# Team Exercise 5 candidate data and test package

**Status: candidate for Group 2 workshop review, not a recorded team decision.** The branch supplies executable Development loading, a model input allowlist, duplicate protected internal splitting, a configurable chat completions runner, scoring, a proposed labeling guide, a data card, and a management briefing draft. Yuchen prepared this integration candidate from the three members' posted Assignment 5 evidence. The team must reconcile and approve it during its workshop before treating it as the shared design.

## Scope and boundary

The retrospective cohort is `product=Credit card`, `issue=Billing disputes`, `submitted_via=Web`. The **only text sent to a model** is the original `consumer_complaint_narrative`; `course_record_id` is retained outside the prompt for matching. Source `issue` selects an already admitted offline cohort. A future live scope gate needs separate validation. The model proposes a route, escalation, additional specialist review, summary, rationale, and uncertainty. Employees retain final confirmation or deferral on every case.

See [DATA_CARD.md](DATA_CARD.md) for field roles, evidence limits, and provenance; [LABELING_GUIDE.md](LABELING_GUIDE.md) for proposed adjudication rules; and [MANAGEMENT_BRIEFING_DRAFT.md](MANAGEMENT_BRIEFING_DRAFT.md) for the workshop decision record.

## Reproduce locally

Use Python 3.10+ and the standard library. Keep the CSV and generated files **outside Git**. From the repository root:

```powershell
python -B team5/test_data_harness.py
python -B team5/data_harness.py prepare "C:\path\to\ABI_Bank_Complaints_Development_8000.csv" "C:\local\team5_run"
```

`prepare` checks every row is Development with `DEV-` IDs before filtering, checks unique IDs and nonblank narratives, groups whitespace-identical narratives across all 8,000 Development rows, and produces:

- `model_inputs.jsonl`: `course_record_id` and original narrative only. The ID is never put in the model prompt.
- `split_manifest.csv`: ID, candidate internal split, and narrative-group hash, held locally and separate from model text.
- `run_manifest.json`: Development file digest, field boundary, counts, split seed, and generated-file hashes.

The default split is deterministic but not stratified. Check rare-case coverage before adopting it. A local run on the original Development file gave 8,000 validated rows, 653 in scope, 541 candidate training records, and 112 internal validation records. This is **Development evidence**, not held-out evaluation. The 112 validation records include 88 Billing routes, 24 Fraud routes, and only 2 `Insufficient Evidence` cases; do not make a strong safety claim from that small stratum.

## Model path and scoring

When a teammate has an approved DGX service endpoint and terminal/API access, run a small smoke test on the **named commit**. Supply the exact served model ID and the full approved chat-completions endpoint. If the service needs a key, set `TEAM5_API_KEY` in that terminal without putting it in a file or chat.

```powershell
python -B team5/model_runner.py "C:\local\team5_run" "<approved-chat-completions-endpoint>" "<served-model-id>" "C:\local\team5_model_smoke" --limit 3
python -B team5/data_harness.py score "C:\path\to\ABI_Bank_Complaints_Development_8000.csv" "C:\local\team5_run" "C:\local\team5_model_smoke\predictions.jsonl"
```

The runner sends only the narrative as the user message. It requires JSON-only outputs, logs malformed responses as failures, and writes the requested model/version/endpoint and attempted count to `run_evidence.json`. The scorer accepts only internal validation IDs and the original Development CSV digest. Agreement with synthetic references is provisional. Preserve the local evidence and record timestamp, DGX host, model/adapter version, commit, and observed errors in the team experiment log.

The runner was verified only through an isolated local mock response. **This commit has not been executed on DGX**; the user's available access is the Open WebUI page, not a terminal or API credential. A browser conversation cannot prove that this Git commit ran on DGX.

## Held-out rule

The code deliberately has no held-out loader or scorer. Do not inspect or use held-out cases to select prompts, adapters, rules, thresholds, or hyperparameters. Before a later formal evaluation, freeze the model, prompt, code commit, rubric, metric, sample plan, and decision thresholds. The team must create a separate evaluation-only path after that freeze. See [HELDOUT_RULES.md](HELDOUT_RULES.md).

## Verification completed on this branch

- Five synthetic fixture tests passed: input allowlist, duplicate grouping, held-out rejection even outside scope, validation-only scoring and source-hash check, and a mock endpoint request/strict parser.
- The real Development file was read and projected locally; no held-out CSV was opened.
- The source SHA-256 was `edc95eb03b97681c8c7fce63163b29608751cd20260593e01a9b6a188c743a53`.
- The generated `model_inputs.jsonl` was inspected for exact two-key objects. No labels, outcome fields, or evaluator metadata were exported to the model-facing file.

This work can support repeatable classroom exploration. It cannot establish bank correctness, live intake performance, DGX integration, or production safety.
