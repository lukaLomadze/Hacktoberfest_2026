"""The agent runtime: a DeepSeek tool-use loop, plus a scripted stand-in for offline tests.

A `DeepSeekAgent` is one long conversation with DeepSeek (one per hospital, one for the
coordinator) that lasts the whole job, so each agent remembers what it decided and what
happened. Its only way to act is to call the tools its process gives it; the tools do the
arithmetic (training, masking, aggregation), the agent makes the decisions and explains
them. It speaks DeepSeek's OpenAI-compatible chat-completions API over stdlib HTTP (no SDK).

The loop is append-only: every assistant message is sent back unchanged, including its
`reasoning_content`. In thinking mode with tools DeepSeek requires that (it answers 400
otherwise), and it keeps each request a prefix of the next, which is what DeepSeek's
automatic context caching needs to hit. Every request/response goes to a transcript file.

`ScriptedAgent` drives the same tools with a fixed policy. It exists so the test suite and
CI run without an API key; a run made with it is labelled made_by="rule" everywhere.
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
import time
import urllib.error
import urllib.request

from . import config


class AgentError(RuntimeError):
    pass


class ToolError(Exception):
    """Raised by a tool function: the message goes back to the agent as an error result."""


class ToolBox:
    def __init__(self):
        self._tools = {}

    def add(self, name: str, description: str, properties: dict, required: list, fn):
        self._tools[name] = ({"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}}}, fn)

    def api_tools(self) -> list:
        return [spec for spec, _ in self._tools.values()]

    def call(self, name: str, args) -> tuple[str, bool]:
        """→ (JSON text for the tool message, is_error)."""
        if name not in self._tools:
            return json.dumps({"error": f"unknown tool {name!r}"}), True
        if not isinstance(args, dict):
            return json.dumps({"error": "tool input must be a JSON object"}), True
        try:
            result = self._tools[name][1](args)
            return json.dumps(result, separators=(",", ":"), default=float), False
        except ToolError as e:
            return json.dumps({"error": str(e)}), True
        except (ValueError, TypeError, KeyError) as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"}), True


class Usage:
    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.input = self.output = self.cache_read = self.reasoning = 0
        self.served_by = {}

    def add(self, resp: dict):
        u = resp.get("usage") or {}
        self.calls += 1
        hit = u.get("prompt_cache_hit_tokens") or 0
        self.cache_read += hit
        self.input += u.get("prompt_cache_miss_tokens", (u.get("prompt_tokens") or 0) - hit) or 0
        self.output += u.get("completion_tokens") or 0
        self.reasoning += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        m = resp.get("model") or self.model
        self.served_by[m] = self.served_by.get(m, 0) + 1

    def cost_usd(self) -> float:
        """At DeepSeek's peak-hour rates; an off-peak run costs about half of this."""
        miss, hit, out = config.PRICES.get(self.model, (0.0, 0.0, 0.0))
        return round((self.input * miss + self.cache_read * hit + self.output * out) / 1e6, 4)

    def to_dict(self):
        return {"model": self.model, "api_calls": self.calls, "input_tokens": self.input,
                "output_tokens": self.output, "reasoning_tokens": self.reasoning,
                "cache_read_tokens": self.cache_read, "served_by": self.served_by,
                "estimated_cost_usd": self.cost_usd()}


class _Transcript:
    def __init__(self, path):
        self.path = pathlib.Path(path) if path else None
        self.lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind, **data):
        if not self.path:
            return
        line = json.dumps({"t": round(time.time(), 3), "kind": kind, **data}, default=str)
        with self.lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


NUDGE = ("You have not finished this step yet. Act by calling the appropriate tool "
         "(see the instructions above); do not reply with text only.")

RETRY_STATUS = (429, 500, 502, 503, 504)  # rate limit and server trouble: wait and retry


def base_url() -> str:
    return (os.environ.get("DEEPSEEK_BASE_URL") or config.DEEPSEEK_BASE_URL).rstrip("/")


def api_key() -> str:
    return os.environ.get("DEEPSEEK_API_KEY", "")


def _http(method: str, path: str, body=None, timeout: float = 600.0) -> dict:
    """One JSON request to the DeepSeek API. urllib errors go to the caller."""
    req = urllib.request.Request(
        base_url() + path, method=method,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "Authorization": f"Bearer {api_key()}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _error_text(e: urllib.error.HTTPError) -> str:
    try:
        err = json.loads(e.read().decode("utf-8", "replace")).get("error", {})
        return err.get("message", str(err)) if isinstance(err, dict) else str(err)
    except (ValueError, OSError):
        return str(e)


class DeepSeekAgent:
    made_by = "llm"

    def __init__(self, name: str, system: str, toolbox: ToolBox, model: str = config.DEFAULT_MODEL,
                 effort: str = "high", transcript=None, max_tokens: int = 32000):
        self.name = name
        self.system = system
        self.toolbox = toolbox
        self.model = model
        self.effort = effort  # DeepSeek reasoning_effort: none (no thinking), low, high or max
        self.max_tokens = max_tokens
        self.messages = [{"role": "system", "content": system}]
        self.usage = Usage(model)
        self.transcript = _Transcript(transcript)
        self.last_thinking = ""
        self.transcript.write("system", agent=name, model=model, effort=effort, system=system,
                              tools=toolbox.api_tools())

    def _create(self) -> dict:
        # no tool_choice: "required" or a named tool is a 400 in thinking mode
        body = {"model": self.model, "messages": self.messages, "tools": self.toolbox.api_tools(),
                "reasoning_effort": self.effort, "max_tokens": self.max_tokens}
        for attempt in range(7):
            try:
                resp = _http("POST", "/chat/completions", body)
                if (resp.get("choices") or [{}])[0].get("finish_reason") != "insufficient_system_resource":
                    return resp
                why = "insufficient_system_resource"
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_STATUS:
                    raise AgentError(f"{self.name}: DeepSeek API error {e.code}: {_error_text(e)}") from None
                why = f"HTTP {e.code}: {_error_text(e)}"
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                why = f"{type(e).__name__}: {e}"
            if attempt == 6:
                raise AgentError(f"{self.name}: DeepSeek API unavailable after retries ({why})")
            self.transcript.write("api_retry", error=why)
            time.sleep(min(60, 2 * 2 ** attempt))

    def run(self, text: str, done, max_turns: int = 12, context=None) -> str:
        """Send `text`, let the agent call tools until done() is true. → its last text."""
        self.messages.append({"role": "user", "content": text})
        self.transcript.write("user", content=text)
        nudges = 0
        last_text = ""
        for _ in range(max_turns):
            resp = self._create()
            self.usage.add(resp)
            choice = resp["choices"][0]
            msg = choice["message"]
            finish = choice.get("finish_reason")
            self.transcript.write("assistant", finish_reason=finish, model=resp.get("model"), message=msg,
                                  usage=resp.get("usage"))
            if finish == "content_filter":
                raise AgentError(f"{self.name}: the reply was blocked by DeepSeek's content filter")
            # keep the message as received, reasoning_content included: DeepSeek needs it back
            # on every later request of a conversation that uses tools
            kept = {k: msg[k] for k in ("content", "reasoning_content", "tool_calls") if k in msg}
            self.messages.append({"role": "assistant", "content": None, **kept})
            if msg.get("reasoning_content"):
                self.last_thinking = msg["reasoning_content"]
            if msg.get("content"):
                last_text = msg["content"]
            calls = msg.get("tool_calls") or []
            if not calls:
                if done():
                    return last_text
                if finish == "length":
                    self.messages.append({"role": "user", "content": "Your reply was cut off. Continue, briefly."})
                    continue
                nudges += 1
                if nudges > 3:
                    raise AgentError(f"{self.name}: stopped without calling a tool ({last_text[:200]!r})")
                self.messages.append({"role": "user", "content": NUDGE})
                self.transcript.write("user", content=NUDGE)
                continue
            for c in calls:
                fn = c.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:  # the API warns arguments may not be valid JSON
                    content, is_error = json.dumps({"error": "the tool arguments were not valid JSON"}), True
                else:
                    content, is_error = self.toolbox.call(fn.get("name", ""), args)
                self.transcript.write("tool", name=fn.get("name"), input=fn.get("arguments"), result=content,
                                      is_error=is_error)
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": content})
            if done():
                return last_text
        raise AgentError(f"{self.name}: did not finish within {max_turns} turns")


class ScriptedAgent:
    """Calls the same tools through a fixed policy(agent, context). No API, no key."""

    made_by = "rule"

    def __init__(self, name: str, toolbox: ToolBox, policy, transcript=None):
        self.name = name
        self.toolbox = toolbox
        self.policy = policy
        self.usage = Usage("rule")
        self.transcript = _Transcript(transcript)
        self.last_thinking = ""
        self.memory = {}

    def call(self, name, args):
        content, is_error = self.toolbox.call(name, args)
        self.transcript.write("tool", name=name, input=args, result=content, is_error=is_error)
        if is_error:
            raise AgentError(f"{self.name}: tool {name} failed: {content}")
        return json.loads(content)

    def run(self, text: str, done, max_turns: int = 12, context=None) -> str:
        self.transcript.write("user", content=text)
        self.policy(self, context or {})
        if not done():
            raise AgentError(f"{self.name}: scripted policy did not finish the step")
        return ""


def check_credentials() -> str:
    """Fail fast with a helpful message when no DeepSeek key is configured."""
    if api_key():
        return "environment"
    raise AgentError(
        "No DeepSeek API key found. Set it for this terminal first, e.g.\n"
        "  PowerShell:  $env:DEEPSEEK_API_KEY = \"sk-...\"\n"
        "  bash:        export DEEPSEEK_API_KEY=sk-...\n"
        "(or run with --agents rule for the offline scripted stand-in).")


def preflight(model: str) -> str:
    """One tiny authenticated call, so a bad key or model fails before any process starts."""
    try:
        listed = _http("GET", "/models", timeout=60.0)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise AgentError("the API key was rejected (401). Check DEEPSEEK_API_KEY.") from None
        if e.code == 402:
            raise AgentError("the DeepSeek account has no balance left (402). Top it up first.") from None
        raise AgentError(f"DeepSeek API error {e.code}: {_error_text(e)}") from None
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        raise AgentError(f"cannot reach the DeepSeek API at {base_url()}: {e}") from None
    ids = [m.get("id") for m in listed.get("data", [])]
    if ids and model not in ids:
        raise AgentError(f"model {model!r} is not offered to this key; available: {', '.join(ids)}")
    return model
