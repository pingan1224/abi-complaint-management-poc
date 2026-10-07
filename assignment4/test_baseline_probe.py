"""Controlled local API test for the Assignment 4 candidate runner.

Runs only against an in-process mock server on 127.0.0.1; no model/API key needed.
"""

import importlib.util
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
from threading import Thread

module_path = Path(__file__).parent / "baseline_probe.py"
spec = importlib.util.spec_from_file_location("baseline_probe", module_path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Handler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        assert self.path == "/api/chat/completions"
        assert self.headers["Authorization"] == "Bearer test-key"
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append(body)
        content = '{"summary":"ok"}' if len(self.requests) == 1 else "```json\n{}\n```"
        data = json.dumps({"id": "mock-id", "model": "mock-model", "choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


server = HTTPServer(("127.0.0.1", 0), Handler)
thread = Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    url = f"http://127.0.0.1:{server.server_port}/api/chat/completions"
    request = probe.build_request("mock-model", "Complaint: {complaint}", "sample", 0, 500)
    first = probe.call_webui(url, "test-key", request, 5)
    second = probe.call_webui(url, "test-key", request, 5)
    assert first["strict_json"] is True
    assert second["strict_json"] is False
    assert second["model_text"].startswith("```json")
    assert first["response_model"] == "mock-model"
    assert len(Handler.requests) == 2
    assert "test-key" not in json.dumps([first, second, Handler.requests])
    try:
        json.loads('{"x":1,"x":2}', object_pairs_hook=probe.unique_object, parse_constant=probe.reject_constant)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate keys must be rejected")
    Handler.requests.clear()
    output = Path(__file__).parent / "probe_mock_run.jsonl"
    output.unlink(missing_ok=True)
    old_argv = sys.argv
    old_key = os.environ.get("OPEN_WEBUI_API_KEY")
    try:
        sys.argv = ["baseline_probe.py", "--base-url", url.rsplit("/api/", 1)[0], "--model", "mock-model", "--output", str(output)]
        os.environ["OPEN_WEBUI_API_KEY"] = "test-key"
        probe.main()
    finally:
        sys.argv = old_argv
        if old_key is None:
            os.environ.pop("OPEN_WEBUI_API_KEY", None)
        else:
            os.environ["OPEN_WEBUI_API_KEY"] = old_key
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert records[0]["result"]["strict_json"] is True
    assert records[1]["result"]["strict_json"] is False
    assert "test-key" not in output.read_text(encoding="utf-8")
    output.unlink()
    print("PASS: authenticated request, raw text preservation, strict JSON signal, no key in evidence")
finally:
    server.shutdown()
