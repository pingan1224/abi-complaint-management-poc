"""Run fixed synthetic complaints through Open WebUI and preserve raw evidence.

This runner records output; it does not judge the correctness or safety of advice.
Python 3.10+ standard library only. Never writes the API key to an output file.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parent


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Non-JSON constant: " + value)


def load_cases(path):
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a nonempty JSON list")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "complaint"}:
            raise ValueError("Each case needs exactly id and complaint")
        if not all(isinstance(case[key], str) and case[key].strip() for key in case):
            raise ValueError("Case id and complaint must be nonempty strings")
        if case["id"] in ids:
            raise ValueError("Duplicate case id: " + case["id"])
        ids.add(case["id"])
    return cases


def build_request(model, prompt, complaint, temperature, max_tokens):
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt.replace("{complaint}", complaint)}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }


def call_webui(url, api_key, request_body, timeout):
    payload = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        return {"http_status": error.code, "error": "HTTP error", "response_text": error.read().decode("utf-8", errors="replace")}
    except (urllib.error.URLError, TimeoutError) as error:
        return {"error": type(error).__name__ + ": " + str(error)}

    result = {"http_status": status, "response_text": raw}
    try:
        envelope = json.loads(raw)
        content = envelope["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("message content is not a string")
    except (ValueError, KeyError, IndexError, TypeError) as error:
        result["error"] = "Cannot extract model text: " + str(error)
        return result
    result["model_text"] = content
    result["response_model"] = envelope.get("model")
    result["response_id"] = envelope.get("id")
    try:
        json.loads(content, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result["strict_json"] = True
    except ValueError as error:
        result["strict_json"] = False
        result["json_parse_error"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPEN_WEBUI_URL"))
    parser.add_argument("--model", default=os.environ.get("OPEN_WEBUI_MODEL"))
    parser.add_argument("--cases", type=Path, default=ROOT / "baseline_cases.json")
    parser.add_argument("--prompt-file", type=Path, default=ROOT / "baseline_prompt.txt")
    parser.add_argument("--output", type=Path, help="New JSONL evidence file; existing files are not overwritten")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true", help="Show request bodies without calling the model")
    args = parser.parse_args()
    if not args.model:
        parser.error("Provide --model or OPEN_WEBUI_MODEL")
    if not 0 <= args.temperature <= 2 or args.max_tokens < 1 or args.timeout < 1:
        parser.error("Invalid temperature, max-tokens, or timeout")
    cases = load_cases(args.cases)
    prompt = args.prompt_file.read_text(encoding="utf-8")
    if prompt.count("{complaint}") != 1:
        parser.error("Prompt must contain {complaint} exactly once")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    requests = [(case, build_request(args.model, prompt, case["complaint"], args.temperature, args.max_tokens)) for case in cases]
    if args.dry_run:
        print(json.dumps({"prompt_sha256": prompt_sha256, "requests": [body for _, body in requests]}, indent=2))
        return
    api_key = os.environ.get("OPEN_WEBUI_API_KEY")
    if not args.base_url or not api_key or not args.output:
        parser.error("Live runs require --base-url, OPEN_WEBUI_API_KEY, and --output")
    url = args.base_url.rstrip("/") + "/api/chat/completions"
    with args.output.open("x", encoding="utf-8") as stream:
        for case, request_body in requests:
            record = {
                "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "case_id": case["id"],
                "complaint": case["complaint"],
                "endpoint": url,
                "prompt_sha256": prompt_sha256,
                "request": request_body,
                "result": call_webui(url, api_key, request_body, args.timeout),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            outcome = "ERROR" if "error" in record["result"] else ("STRICT_JSON" if record["result"].get("strict_json") else "NON_JSON_TEXT")
            print(case["id"], outcome)
    print("Evidence saved to", args.output)
    print("Output format does not establish factual accuracy or safe routing.")


if __name__ == "__main__":
    main()
