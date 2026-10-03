# Assignment 4 candidate output check

Candidate branch: candidate/lester-output-check

Run on DGX with Python 3.10+ (standard library only):

    python -B assignment4/output_check.py

This component checks six AI Chat responses pasted by the student into the learning conversation. It performs no model inference. responses.json preserves supplied response text, including the V2-C2 Markdown fences and explanatory paragraph. The UI display name was llama-3.1-8b-instruct; exact revision, additional adaptation, serving settings, and generation parameters were not verified.

V1 did not specify escalation type or allowed label values. String escalation is therefore an observation, not a V1 schema failure. V2 explicitly required Boolean escalation and fixed labels. Strict JSON parsing is applied to both prompts because both requested JSON only. Expected format checks: V1 three passes; V2 two passes and one rejection, V2-C2. Six examples are not a benchmark or general performance claim.

A successful format check says nothing about invented facts, omitted facts, urgency evidence, escalation policy, or the suitability of a next action. Those require human examination. Do not silently strip Markdown or explanations: doing so would hide the observed contract failure.

Technical reference: https://docs.python.org/3/library/json.html
The official documentation defines JSONDecodeError and object_pairs_hook; the latter rejects duplicate keys here. The observed extra-text case independently demonstrates the parsing limitation.

No integrated prototype, automatic action, production integration, main-branch merge, or Gate 2 authorization is implied. This package is an individual learning candidate for workshop review. Record the actual DGX terminal result, commit, and run environment before describing it as tested there.
