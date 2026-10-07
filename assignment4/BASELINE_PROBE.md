# Assignment 4 candidate baseline probe

This candidate runs three fixed synthetic credit-card complaints through an Open WebUI chat-completions API and records the exact request and response. It complements the existing candidate output checkers. It does not validate facts, routing, urgency, escalation policy, or employee decisions.

## Relationship to Group 2's agreed scope

Team Exercises 2 and 3 narrowed the classroom PoC to **web-submitted credit-card billing disputes**. Its employee-facing readout should distinguish Card Billing Disputes from Card Fraud and Security, show escalation or human-review status, rationale, and uncertainty, and allow human confirmation, override, rejection, or deferral. The cases in this probe are synthetic examples within that problem area; the current prompt is an exploratory, unadapted-model format probe, not the team's approved schema or routing policy. It does not explicitly encode the Web channel, constrain destinations to the two team categories, or require a rationale and uncertainty indicator. These are known gaps for team review, not performance failures proven by this script.

Do not use course held-out evaluation cases for this learning exercise. The team's threshold proposals and instructional labels are not observed bank results. The probe cannot establish business materiality, safety-gate performance, or incremental value over rules, forms, interface changes, or training.

Use Python 3.10+ with no third-party packages. The model ID must be confirmed from the authenticated Open WebUI model list. A display name alone does not prove the exact model build or serving configuration.

To reproduce the local transport check without contacting DGX, run `python -B assignment4/test_baseline_probe.py`. It starts a mock HTTP server on 127.0.0.1, exercises all three cases, and checks request structure, raw-text preservation, JSON syntax signaling, duplicate-key rejection, and the absence of the mock key in the saved evidence. A passing mock test does not prove the real Open WebUI deployment works.

```bash
python -B assignment4/baseline_probe.py --model ACTUAL_MODEL_ID --dry-run
export OPEN_WEBUI_API_KEY='YOUR_KEY'  # set this privately; do not commit it
python -B assignment4/baseline_probe.py --base-url http://127.0.0.1:8080 --model ACTUAL_MODEL_ID --output baseline_run.jsonl
```

The local URL above assumes the script runs on the DGX host. For remote use, obtain an instructor-approved HTTPS endpoint rather than transmitting an API key over plain HTTP. On PowerShell, use `$env:OPEN_WEBUI_API_KEY = 'YOUR_KEY'` instead of `export`. The live run refuses to overwrite an existing evidence file. Keep API keys, real customer data, and raw private server responses out of Git. The committed cases are synthetic. Review the JSONL locally and record the branch, commit, DGX hostname, actual model ID and revision if available, generation settings, results, and limitations in the individual assignment record.

The fixed request sets `temperature=0`, `max_tokens=500`, and `stream=false`; these are request parameters, not proof of the server's effective settings. The response envelope and model text are preserved even when the text is not valid JSON. `strict_json` is a syntax signal only; it rejects duplicate keys and non-JSON constants but does not judge the content.

Technical reference: Open WebUI's official API documentation describes `POST /api/chat/completions`, bearer authentication, and the `model` and `messages` request fields: https://docs.openwebui.com/reference/api-endpoints/
