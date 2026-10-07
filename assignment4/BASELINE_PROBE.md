# Assignment 4 candidate baseline probe

This candidate runs three fixed synthetic credit-card complaints through an Open WebUI chat-completions API and records the exact request and response. It complements the existing candidate output checkers. It does not validate facts, routing, urgency, escalation policy, or employee decisions.

Use Python 3.10+ with no third-party packages. The model ID must be confirmed from the authenticated Open WebUI model list. A display name alone does not prove the exact model build or serving configuration.

```bash
python -B assignment4/baseline_probe.py --model ACTUAL_MODEL_ID --dry-run
export OPEN_WEBUI_API_KEY='YOUR_KEY'  # set this privately; do not commit it
python -B assignment4/baseline_probe.py --base-url http://127.0.0.1:8080 --model ACTUAL_MODEL_ID --output baseline_run.jsonl
```

The local URL above assumes the script runs on the DGX host. For remote use, obtain an instructor-approved HTTPS endpoint rather than transmitting an API key over plain HTTP. On PowerShell, use `$env:OPEN_WEBUI_API_KEY = 'YOUR_KEY'` instead of `export`. The live run refuses to overwrite an existing evidence file. Keep API keys, real customer data, and raw private server responses out of Git. The committed cases are synthetic. Review the JSONL locally and record the branch, commit, DGX hostname, actual model ID and revision if available, generation settings, results, and limitations in the individual assignment record.

The fixed request sets `temperature=0`, `max_tokens=500`, and `stream=false`; these are request parameters, not proof of the server's effective settings. The response envelope and model text are preserved even when the text is not valid JSON. `strict_json` is a syntax signal only; it rejects duplicate keys and non-JSON constants but does not judge the content.

Technical reference: Open WebUI's official API documentation describes `POST /api/chat/completions`, bearer authentication, and the `model` and `messages` request fields: https://docs.openwebui.com/reference/api-endpoints/
