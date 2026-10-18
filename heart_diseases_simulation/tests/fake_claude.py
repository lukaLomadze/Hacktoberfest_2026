"""A stand-in for the Claude Messages API, for testing the agent loop without a key.

It answers POST /v1/messages like the real API (thinking block + tool_use, usage), choosing
the tool call from the request's tools and conversation, and records every request so the
tests can check what the SDK actually sent (model, thinking, effort, fallbacks, tools,
the append-only history). Point the SDK at it with ANTHROPIC_BASE_URL.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONTRACT = {
    "features": [{"name": n, "encoding": e, "missing": "impute_mean"} for n, e in [
        ("age", "numeric"), ("sex", "binary"), ("cp", "categorical"), ("trestbps", "numeric"), ("chol", "numeric"),
        ("fbs", "binary"), ("restecg", "categorical"), ("thalach", "numeric"), ("exang", "binary"),
        ("oldpeak", "numeric")]],
    "quality_rules": [{"column": "chol", "rule": "zero_means_missing"},
                      {"column": "trestbps", "rule": "zero_means_missing"}],
    "reasoning": "fake: the ten columns most hospitals record",
}


def _last_user_text(messages):
    for m in reversed(messages):
        if m["role"] == "user" and isinstance(m["content"], str):
            return m["content"]
    return ""


def _tool_calls_since_last_text(messages):
    names = []
    for m in reversed(messages):
        if m["role"] == "user" and isinstance(m["content"], str):
            break
        if m["role"] == "assistant":
            names += [b["name"] for b in m["content"] if b.get("type") == "tool_use"]
    return names


def decide(body, rounds_before_finish=4):
    tools = {t["name"] for t in body.get("tools", [])}
    msgs = body["messages"]
    if "run_round" in tools:  # the coordinator
        done = [b["name"] for m in msgs if m["role"] == "assistant" for b in m["content"]
                if b.get("type") == "tool_use"]
        seq = (["collect_data_descriptions", "propose_contract", "evaluate_candidate_models", "select_model"]
               + ["run_round"] * rounds_before_finish + ["finish_training"])
        name = seq[min(len(done), len(seq) - 1)]
        args = {"collect_data_descriptions": {"note": "describe broken values"},
                "propose_contract": CONTRACT,
                "evaluate_candidate_models": {"candidates": [{"type": "logistic", "l2": 0.01},
                                                             {"type": "mlp", "hidden": 8, "l2": 0.01}]},
                "select_model": {"type": "logistic", "l2": 0.01, "reasoning": "fake: fewer parameters under DP"},
                "run_round": {"strategy": "fedprox", "mu": 0.05, "reasoning": "fake: different disease rates"},
                "finish_training": {"checkpoint": "best_loss", "reasoning": "fake: validation loss plateaued"}}[name]
        return name, args
    text = _last_user_text(msgs)
    called = _tool_calls_since_last_text(msgs)
    if text.startswith("STEP: describe"):
        if "inspect_local_data" not in called:
            return "inspect_local_data", {"view": "overview"}
        return "submit_data_description", {
            "columns": [{"name": "chol", "usable": True, "note": "zeros are missing"}],
            "quality_rules": [{"column": "chol", "rule": "zero_means_missing"}], "summary": "fake description"}
    if text.startswith("STEP: review"):
        if "inspect_local_data" not in called:
            return "inspect_local_data", {"view": "contract_impact"}
        return "review_contract", {"accept": True, "objections": [], "reasoning": "fake: we can provide these"}
    if text.startswith("STEP: plan"):
        return "submit_training_plan", {"participate": True, "epochs": 4, "lr": 0.5, "reasoning": "fake plan"}
    return None, None


class FakeClaude:
    def __init__(self, port=0):
        self.requests = []
        self.errors = []  # tool results the agents got back with is_error
        self.lock = threading.Lock()
        owner = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.startswith("/v1/models/"):
                    mid = self.path.split("/")[3].split("?")[0]
                    self._send(200, {"type": "model", "id": mid, "display_name": f"Fake {mid}",
                                     "created_at": "2026-01-01T00:00:00Z"})
                else:
                    self._send(404, {"type": "error", "error": {"type": "not_found_error", "message": self.path}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with owner.lock:
                    owner.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                    for m in body["messages"]:
                        if m["role"] == "user" and isinstance(m["content"], list):
                            for b in m["content"]:
                                if b.get("type") == "tool_result" and b.get("is_error"):
                                    if b not in owner.errors:
                                        owner.errors.append(b)
                name, args = decide(body)
                n = len(owner.requests)
                content = [{"type": "thinking", "thinking": f"fake thinking {n}", "signature": f"sig{n}"}]
                if name:
                    content += [{"type": "text", "text": "Acting."},
                                {"type": "tool_use", "id": f"toolu_{n:05d}", "name": name, "input": args}]
                    stop = "tool_use"
                else:
                    content += [{"type": "text", "text": "Nothing to do."}]
                    stop = "end_turn"
                self._send(200, {"id": f"msg_{n:05d}", "type": "message", "role": "assistant",
                                 "model": body["model"], "content": content, "stop_reason": stop,
                                 "stop_sequence": None,
                                 "usage": {"input_tokens": 100, "output_tokens": 20,
                                           "cache_read_input_tokens": 50, "cache_creation_input_tokens": 10}})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()


if __name__ == "__main__":  # run standalone: python tests/fake_claude.py 8999
    import sys

    with FakeClaude(int(sys.argv[1]) if len(sys.argv) > 1 else 8999) as f:
        print(f"fake Claude API on http://127.0.0.1:{f.port}")
        threading.Event().wait()
