"""Agent runtime and message bus for PrivateLearn (T6). Stdlib only, headless.

Every agent is a plain object with ``step(messages) -> messages``. The bus delivers
messages in synchronous ticks: whatever an agent sends during tick t is delivered at
tick t+1, in a fixed order, so a run is fully deterministic and replayable.

The bus is also the privacy boundary between agents:

- every payload must be JSON-serialisable (no numpy arrays, no DataFrames) and is
  deep-copied through JSON on the way in, so agents never share mutable state;
- ``update`` payloads are checked with ``schemas.validate_update`` (CONTRACTS.md §3.1)
  and ``broadcast`` payloads with ``schemas.validate_broadcast`` (§3.2);
- payloads may not carry raw-data keys (``rows``, ``X``, ``y``, ...) at any depth.

Other kinds (for example ``schema`` or ``plan``) are free-form but still pass the JSON
and raw-data checks. Every delivered message is kept in ``bus.log`` for the trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from shared import schemas

BROADCAST = "*"

VALIDATORS = {
    "update": schemas.validate_update,
    "broadcast": schemas.validate_broadcast,
}

# Keys that would mean patient rows are crossing a hospital boundary.
FORBIDDEN_KEYS = frozenset({"rows", "data", "X", "y", "features_raw", "records", "patients"})


@dataclass(frozen=True)
class Message:
    sender: str
    recipient: str  # an agent id, or BROADCAST for every agent except the sender
    kind: str
    payload: Any
    tick: int = -1  # set by the bus on delivery
    seq: int = -1  # global delivery order, set by the bus

    def to_dict(self) -> dict:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "kind": self.kind,
            "payload": self.payload,
            "tick": self.tick,
            "seq": self.seq,
        }


def _check_no_raw_data(obj, path="payload"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_KEYS:
                raise ValueError(f"{path}.{k}: raw-data field may not cross the bus")
            _check_no_raw_data(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_no_raw_data(v, f"{path}[{i}]")


def sanitize(kind: str, payload: Any) -> Any:
    """Return a JSON deep copy of ``payload`` after contract and privacy checks.

    Raises ValueError naming the field on any violation.
    """
    if not isinstance(kind, str) or not kind:
        raise ValueError("message.kind: must be a non-empty string")
    try:
        copy = json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{kind}.payload: must be JSON-serialisable and finite ({exc})") from None
    _check_no_raw_data(copy, f"{kind}.payload")
    validator = VALIDATORS.get(kind)
    if validator is not None:
        validator(copy)
    return copy


class Agent:
    """Base class. Subclasses override ``step``."""

    def __init__(self, agent_id: str):
        if not isinstance(agent_id, str) or not agent_id or agent_id == BROADCAST:
            raise ValueError(f"agent id must be a non-empty string other than {BROADCAST!r}")
        self.id = agent_id

    def send(self, recipient: str, kind: str, payload: Any) -> Message:
        return Message(sender=self.id, recipient=recipient, kind=kind, payload=payload)

    @property
    def waiting(self) -> bool:
        """True while the agent expects to act on a later tick even with an empty inbox
        (a coordinator waiting out a reply deadline). Keeps `MessageBus.run` going."""
        return False

    def step(self, messages: list[Message]) -> list[Message]:
        """Handle this tick's inbox and return the messages to send. Default: do nothing."""
        return []


class MessageBus:
    """Synchronous, deterministic tick-based bus."""

    def __init__(self):
        self._agents: dict[str, Agent] = {}
        self._pending: list[Message] = []
        self.log: list[Message] = []
        self.tick = 0
        self._seq = 0

    @property
    def agents(self) -> list[str]:
        return list(self._agents)

    def register(self, agent: Agent) -> Agent:
        if agent.id in self._agents:
            raise ValueError(f"agent id {agent.id!r} already registered")
        self._agents[agent.id] = agent
        return agent

    def post(self, msg: Message) -> None:
        """Queue a message for delivery on the next tick (validated now, at the sender)."""
        if msg.sender not in self._agents:
            raise ValueError(f"message.sender: unknown agent {msg.sender!r}")
        if msg.recipient != BROADCAST and msg.recipient not in self._agents:
            raise ValueError(f"message.recipient: unknown agent {msg.recipient!r}")
        payload = sanitize(msg.kind, msg.payload)
        self._pending.append(Message(msg.sender, msg.recipient, msg.kind, payload))

    def _deliver(self) -> dict[str, list[Message]]:
        inboxes: dict[str, list[Message]] = {aid: [] for aid in self._agents}
        for msg in self._pending:
            targets = (
                [aid for aid in self._agents if aid != msg.sender]
                if msg.recipient == BROADCAST
                else [msg.recipient]
            )
            for aid in targets:
                delivered = Message(
                    msg.sender, aid, msg.kind, json.loads(json.dumps(msg.payload)),
                    tick=self.tick, seq=self._seq,
                )
                self._seq += 1
                inboxes[aid].append(delivered)
                self.log.append(delivered)
        self._pending = []
        return inboxes

    def run_tick(self) -> int:
        """Deliver pending messages, step every agent in registration order.

        Returns the number of messages the agents sent this tick.
        """
        inboxes = self._deliver()
        sent = 0
        for aid, agent in self._agents.items():
            out = agent.step(inboxes[aid]) or []
            for msg in out:
                if msg.sender != aid:
                    raise ValueError(f"agent {aid!r} tried to send as {msg.sender!r}")
                self.post(msg)
                sent += 1
        self.tick += 1
        return sent

    def run(self, max_ticks: int = 100) -> int:
        """Run until a tick sends nothing, nothing is pending and no agent is waiting.

        Without the `waiting` check a round in which nobody answers (every hospital offline,
        or every participant silent after masking) would stop the bus before the
        coordinator reaches its deadline and handles it. Returns ticks run."""
        start = self.tick
        while self.tick - start < max_ticks:
            sent = self.run_tick()
            waiting = any(getattr(a, "waiting", False) for a in self._agents.values())
            if sent == 0 and not self._pending and not waiting:
                return self.tick - start
        raise RuntimeError(f"bus did not go quiet within {max_ticks} ticks")
