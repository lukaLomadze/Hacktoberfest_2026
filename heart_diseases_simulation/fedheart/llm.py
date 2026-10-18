"""The agent runtime: a Claude tool-use loop, plus a scripted stand-in for offline tests.

A `ClaudeAgent` is one long conversation with Claude (one per hospital, one for the
coordinator) that lasts the whole job, so each agent remembers what it decided and what
happened. Its only way to act is to call the tools its process gives it; the tools do the
arithmetic (training, masking, aggregation), the agent makes the decisions and explains
them. The loop is append-only (every assistant turn is sent back unchanged, thinking
blocks included), uses prompt caching, adaptive thinking and the server-side refusal
fallback, and writes every request/response to a transcript file.

`ScriptedAgent` drives the same tools with a fixed policy. It exists so the test suite and
CI run without an API key; a run made with it is labelled made_by="rule" everywhere.
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
import time

from . import config


class AgentError(RuntimeError):
    pass


class ToolError(Exception):
    """Raised by a tool function: the message goes back to the agent as an error result."""


class ToolBox:
    def __init__(self):
        self._tools = {}

    def add(self, name: str, description: str, properties: dict, required: list, fn):
        self._tools[name] = ({"name": name, "description": description,
                              "input_schema": {"type": "object", "properties": properties,
                                               "required": required, "additionalProperties": False}}, fn)

    def api_tools(self) -> list:
        return [spec for spec, _ in self._tools.values()]

    def call(self, name: str, args) -> tuple[str, bool]:
        """→ (JSON text for the tool_result, is_error)."""
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
        self.input = self.output = self.cache_read = self.cache_write = 0
        self.served_by = {}

    def add(self, resp):
        u = resp.usage
        self.calls += 1
        self.input += getattr(u, "input_tokens", 0) or 0
        self.output += getattr(u, "output_tokens", 0) or 0
        self.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0
        m = getattr(resp, "model", self.model)
        self.served_by[m] = self.served_by.get(m, 0) + 1

    def cost_usd(self) -> float:
        pin, pout = config.PRICES.get(self.model, (0.0, 0.0))
        return round((self.input * pin + self.cache_read * pin * 0.1 + self.cache_write * pin * 1.25
                      + self.output * pout) / 1e6, 4)

    def to_dict(self):
        return {"model": self.model, "api_calls": self.calls, "input_tokens": self.input,
                "output_tokens": self.output, "cache_read_tokens": self.cache_read,
                "cache_write_tokens": self.cache_write, "served_by": self.served_by,
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


class ClaudeAgent:
    made_by = "llm"

    def __init__(self, name: str, system: str, toolbox: ToolBox, model: str = config.DEFAULT_MODEL,
                 effort: str = "medium", transcript=None, max_tokens: int = 16000):
        import anthropic  # imported here so offline tests never need the package

        self.name = name
        self.system = system
        self.toolbox = toolbox
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.messages = []
        self.usage = Usage(model)
        self.transcript = _Transcript(transcript)
        self.last_thinking = ""
        self.client = anthropic.Anthropic(max_retries=6, timeout=600.0)
        self._anthropic = anthropic
        self.transcript.write("system", agent=name, model=model, effort=effort, system=system,
                              tools=toolbox.api_tools())

    def _create(self):
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=self.system,
                      tools=self.toolbox.api_tools(), messages=self.messages,
                      thinking={"type": "adaptive", "display": "summarized"},
                      output_config={"effort": self.effort}, cache_control={"type": "ephemeral"})
        if self.model in config.FALLBACK_MODELS:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        for attempt in range(4):
            try:
                return self.client.beta.messages.create(**kwargs)
            except (self._anthropic.RateLimitError, self._anthropic.InternalServerError,
                    self._anthropic.APIConnectionError) as e:
                # the SDK already retried with backoff; wait longer before giving up
                self.transcript.write("api_retry", error=f"{type(e).__name__}: {e}")
                time.sleep(20 * (attempt + 1))
        return self.client.beta.messages.create(**kwargs)

    def run(self, text: str, done, max_turns: int = 12, context=None) -> str:
        """Send `text`, let the agent call tools until done() is true. → its last text."""
        self.messages.append({"role": "user", "content": text})
        self.transcript.write("user", content=text)
        nudges = 0
        last_text = ""
        for _ in range(max_turns):
            resp = self._create()
            self.usage.add(resp)
            blocks = [b.model_dump(mode="json", exclude_none=True) for b in resp.content]
            self.transcript.write("assistant", stop_reason=resp.stop_reason, model=resp.model, content=blocks,
                                  usage=resp.usage.model_dump(mode="json", exclude_none=True))
            if resp.stop_reason == "refusal":
                details = getattr(resp, "stop_details", None)
                raise AgentError(f"{self.name}: the model declined ({details})")
            self.messages.append({"role": "assistant", "content": resp.content})
            thinking = [b.thinking for b in resp.content if b.type == "thinking" and getattr(b, "thinking", "")]
            if thinking:
                self.last_thinking = "\n".join(thinking)
            texts = [b.text for b in resp.content if b.type == "text"]
            if texts:
                last_text = "\n".join(texts)
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                if done():
                    return last_text
                if resp.stop_reason == "max_tokens":
                    self.messages.append({"role": "user", "content": "Your reply was cut off. Continue, briefly."})
                    continue
                nudges += 1
                if nudges > 3:
                    raise AgentError(f"{self.name}: stopped without calling a tool ({last_text[:200]!r})")
                self.messages.append({"role": "user", "content": NUDGE})
                self.transcript.write("user", content=NUDGE)
                continue
            results = []
            for b in uses:
                content, is_error = self.toolbox.call(b.name, b.input)
                self.transcript.write("tool", name=b.name, input=b.input, result=content, is_error=is_error)
                r = {"type": "tool_result", "tool_use_id": b.id, "content": content}
                if is_error:
                    r["is_error"] = True
                results.append(r)
            self.messages.append({"role": "user", "content": results})
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
    """Fail fast with a helpful message when no Claude credential is configured."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "environment"
    raise AgentError(
        "No Claude API key found. Set it for this terminal first, e.g.\n"
        "  PowerShell:  $env:ANTHROPIC_API_KEY = \"sk-ant-...\"\n"
        "  bash:        export ANTHROPIC_API_KEY=sk-ant-...\n"
        "(or run with --agents rule for the offline scripted stand-in).")


def preflight(model: str):
    """One tiny authenticated call, so a bad key or model fails before any process starts."""
    import anthropic

    client = anthropic.Anthropic(max_retries=2, timeout=60.0)
    try:
        info = client.models.retrieve(model)
    except anthropic.AuthenticationError:
        raise AgentError("the API key was rejected (401). Check ANTHROPIC_API_KEY.") from None
    except anthropic.PermissionDeniedError:
        raise AgentError(f"this API key may not use {model} (403).") from None
    except anthropic.NotFoundError:
        raise AgentError(f"model {model!r} was not found for this key (404).") from None
    except anthropic.APIConnectionError as e:
        raise AgentError(f"cannot reach the Claude API: {e}") from None
    return getattr(info, "display_name", model)
