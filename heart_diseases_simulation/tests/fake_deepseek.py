"""A stand-in for DeepSeek's chat-completions API, for testing the agent loop without a key.

It answers POST /chat/completions like the real API in thinking mode (reasoning_content +
tool_calls, usage with cache hit/miss), choosing the tool call from the request's tools and
conversation, and records every request so the tests can check what the agents actually
sent (model, reasoning_effort, tools, the append-only history). Like the real API it
answers 400 when an earlier assistant message comes back without its reasoning_content.
Point the agents at it with DEEPSEEK_BASE_URL.
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
MODELS = ("deepseek-v4-pro", "deepseek-flash")


def _tool_names(m):
    return [c["function"]["name"] for c in m.get("tool_calls") or []]


def _last_user_text(messages):
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


def _tool_calls_since_last_user(messages):
    names = []
    for m in reversed(messages):
        if m["role"] == "user":
            break
        if m["role"] == "assistant":
            names += _tool_names(m)
    return names


def decide(body, rounds_before_finish=4):
    tools = {t["function"]["name"] for t in body.get("tools", [])}
    msgs = body["messages"]
    if "run_round" in tools:  # the coordinator
        done = [n for m in msgs if m["role"] == "assistant" for n in _tool_names(m)]
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
    called = _tool_calls_since_last_user(msgs)
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


class FakeDeepSeek:
    def __init__(self, port=0):
        self.requests = []
        self.errors = []  # tool results the agents got back as errors
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

            def _error(self, code, message):
                self._send(code, {"error": {"message": message, "type": "invalid_request_error"}})

            def do_GET(self):
                if self.path.split("?")[0] == "/models":
                    self._send(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "deepseek"}
                                                                 for m in MODELS]})
                else:
                    self._error(404, self.path)

            def do_POST(self):
                if self.path != "/chat/completions":
                    return self._error(404, self.path)
                if not self.headers.get("Authorization", "").startswith("Bearer "):
                    return self._error(401, "missing bearer token")
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with owner.lock:
                    owner.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                    for m in body["messages"]:
                        if m["role"] == "tool" and '"error"' in m["content"][:12] and m not in owner.errors:
                            owner.errors.append(m)
                    n = len(owner.requests)
                for m in body["messages"]:
                    if m["role"] == "assistant" and not m.get("reasoning_content"):
                        return self._error(400, "The reasoning_content in the thinking mode must be passed back "
                                                "to the API.")
                name, args = decide(body)
                msg = {"role": "assistant", "content": "Acting." if name else "Nothing to do.",
                       "reasoning_content": f"fake reasoning {n}"}
                if name:
                    msg["tool_calls"] = [{"id": f"call_{n:05d}", "type": "function",
                                          "function": {"name": name, "arguments": json.dumps(args)}}]
                self._send(200, {"id": f"chatcmpl-{n:05d}", "object": "chat.completion", "created": 0,
                                 "model": body["model"],
                                 "choices": [{"index": 0, "message": msg,
                                              "finish_reason": "tool_calls" if name else "stop"}],
                                 "usage": {"prompt_tokens": 150, "completion_tokens": 20, "total_tokens": 170,
                                           "prompt_cache_hit_tokens": 50, "prompt_cache_miss_tokens": 100,
                                           "completion_tokens_details": {"reasoning_tokens": 12}}})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()


if __name__ == "__main__":  # run standalone: python tests/fake_deepseek.py 8999
    import sys

    with FakeDeepSeek(int(sys.argv[1]) if len(sys.argv) > 1 else 8999) as f:
        print(f"fake DeepSeek API on http://127.0.0.1:{f.port}")
        threading.Event().wait()
