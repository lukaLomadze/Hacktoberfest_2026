"""JSON over HTTP between separate processes (stdlib only).

Each hospital is an HTTP server on 127.0.0.1; the coordinator is a client. Every payload
passes a tripwire that refuses keys that would mean patient-level data is crossing a
process boundary, and every message's size is counted for the run report.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FORBIDDEN_KEYS = frozenset({"rows", "records", "patients", "X", "y", "raw", "table", "dataframe"})


class RemoteError(RuntimeError):
    """The other process answered with an error (status, message)."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


def check_no_patient_data(obj, path="payload"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_KEYS and isinstance(v, (list, dict)):  # a count is fine, a collection is not
                raise ValueError(f"refusing to send {path}.{k}: looks like patient-level data")
            check_no_patient_data(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:50]):
            check_no_patient_data(v, f"{path}[{i}]")


def serve(port: int, handlers: dict, host: str = "127.0.0.1"):
    """Start a threaded JSON server; handlers map "/path" → fn(payload) -> dict.

    A handler raises `RemoteError(status, msg)` to answer with that status. Returns the
    server; call serve_forever() on it.
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console for the agents' story
            pass

        def _send(self, status, body):
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.do_POST()

        def do_POST(self):
            fn = handlers.get(self.path.split("?")[0])
            if fn is None:
                self._send(404, {"error": f"no route {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                body = fn(payload)
                check_no_patient_data(body, "response")
                self._send(200, body)
            except RemoteError as e:
                self._send(e.status, {"error": e.message})
            except Exception as e:  # noqa: BLE001 - report every failure to the caller
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


class Traffic:
    """Counts messages and bytes per peer, for the 'what crossed the wire' report."""

    def __init__(self):
        self.lock = threading.Lock()
        self.by_peer = {}

    def add(self, peer, sent, received):
        with self.lock:
            e = self.by_peer.setdefault(peer, {"messages": 0, "bytes_sent": 0, "bytes_received": 0})
            e["messages"] += 1
            e["bytes_sent"] += sent
            e["bytes_received"] += received


def post(base_url: str, path: str, payload: dict | None = None, timeout: float = 900.0,
         traffic: Traffic | None = None, peer: str | None = None) -> dict:
    payload = payload or {}
    check_no_patient_data(payload)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read() or b"{}").get("error", str(e))
        except ValueError:
            msg = str(e)
        raise RemoteError(e.code, msg) from None
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        raise RemoteError(0, f"unreachable: {e}") from None
    if traffic is not None:
        traffic.add(peer or base_url, len(data), len(raw))
    return json.loads(raw or b"{}")
