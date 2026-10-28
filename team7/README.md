# Team Exercise 7 integrated complaint prototype

Group 2: Lester Wang, Yinuo Lou, Yuchen Yuan. Version: team7-v1.

This classroom shadow workflow composes all three Assignment 7 candidates with durable intake, an explicit DGX model service, a terminal employee interface, deterministic review restrictions, transactional SQLite audit records and a stop control. It has no bank-action or customer-message implementation.

## Run on DGX

From the repository's `team7/prototype` folder, run `sbatch run_dgx.sh` on the course Slurm environment. The script uses the existing `/opt/sps-ai-lab/venv/bin/python`, the approved local Qwen2.5 32B FP8 checkpoint reconstructed in BF16, and one GB10 GPU. No LoRA adapter is loaded; no training or datasets are used. The model service listens only on `127.0.0.1:18707` inside the allocated node and exits with the job. This is a new team service, not the course chat API or a claim that an existing serving endpoint was verified.

The controlled run executes three synthetic complaints through HTTP -> model -> rules -> review packet -> scripted employee decision -> SQLite and injects timeout/malformed output failures. `SCRIPTED_TEST_REVIEWER` is a test actor, not a person who assessed summary accuracy. Code hashes, commit, versions, raw replies and audit records identify the tested build. Inspect all failures; do not turn an incomplete run into a success claim.

## Interactive employee workflow

While the local model service is running on the allocated node, use another terminal there. Save a synthetic JSON intake file containing exactly `case_id` and `narrative`. Only pass `--scope-verified` after independent intake review of an already-admitted web credit-card billing dispute. Historical issue labels and the LLM do not establish live scope.

```sh
python -B workflow.py --db demo.sqlite submit --case case.json --scope-verified --endpoint http://127.0.0.1:18707/v1/chat/completions --model qwen2.5-32b-team7-base
python -B workflow.py --db demo.sqlite show CASE_ID
python -B workflow.py --db demo.sqlite review CASE_ID
python -B workflow.py --db demo.sqlite pause 'Operator stopped the shadow workflow'
```

The review command displays the original and suggestion and requires operator ID, accept/modify/reject, a permitted action, reason, and explicit source and meaning checks. Defer by leaving the case pending. A pause prevents new submissions and simulated handoffs; use separate manual handling. For this version, restart with a new demonstration database after investigating a pause. No automatic resume is implemented.

Unknown scope skips the model and enters manual intake. Timeout, invalid JSON/types, unavailable model, or abstention preserves manual triage. Independent raw-text flags survive failure and remain visible as a separate specialist-attention requirement. For a valid suggestion, a fraud route, High/Critical urgency, escalation, specialist flag, or any stated uncertainty restricts preparation to specialist/manual review. Employees cannot dismiss those restrictions by ordinary acceptance. Every case requires confirmation. All outcomes remain simulation records.

## Contract and provenance

The eight-field live contract is Lou's six fields (`summary`, `suggested_route`, `urgency`, `escalation`, `abstain`, `rationale`) plus Yuan's `human_review_required` (Yes/No for additional specialist review) and `uncertainty` text. Strict JSON and the original gates validate these fields. Software maps them into Yuan's six-field gate. It attaches the original narrative excerpt to Lester's source-linked packet and selects a bounded classroom preparation action; it does not attribute that quotation or action to the model. Source anchoring does not prove the generated paraphrase. The first 1,000 source characters anchor the packet; the full original remains available. The packet readout is capped at 500 characters while the full raw model reply is retained.

`components/triage_guard.py` is unchanged from Lou commit 2896314a7ff3892175f99490b0352f066f9405b6; `components/authority_gate.py` is unchanged from Yuan 7aa6ffc9d99b33a77f3993eac086afc97493dbcd; `components/review_packet.py`, schema and catalog are unchanged from Lester ea910930463deaf53f511b6a7660af5234335966. Integration is a separate increment. The catalog remains illustrative classroom material, not bank policy. REQUEST_INFORMATION is retained in the catalog but not automatically generated in the initial live path; no customer reply generator is included.

## Checks and limitations

```sh
cd prototype
python -B -m unittest -v test_workflow
```

The integrated tests exercise original-before-model persistence, HTTP envelope, scope exclusion, malformed/duplicate JSON, strict Boolean validation, raw risk precedence, uncertainty, abstention, source checks, duplicate case/review rejection, pause and audit rollback. SQLite contains originals and model replies: use synthetic examples for shared evidence. The digest detects accidental packet edits, not adversarial tampering. SQLite audit is transactional, not immutable. Operator identity is self-declared, not authentication; an operator can edit the local database. A phrase detector is incomplete and can overflag negation. Scope admission remains an external human decision. No real queue delivery, calibrated risk score, real employee study, bank-policy approval, production security or financial operation is claimed.

The management briefing and project records accompany the submission. Workshop attendance, member assent and time allocation must come from actual human records; AI consolidation does not establish them. The old individual DGX job 273 is supporting component evidence and must not be relabeled as this team's new integrated run.
