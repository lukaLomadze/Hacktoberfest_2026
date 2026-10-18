"""Contract validators for PrivateLearn messages and the trace.

Pure Python (stdlib only) so it has no dependency on numpy and can run anywhere.
Raise ValueError with a field-specific message on any violation.
"""

from __future__ import annotations

import math
import re

ALLOWED_STRATEGIES = {"fedavg", "fedprox", "trimmed_mean"}
VERSION = "1.0"
COORDINATOR_ID = "coord"
# A participant id is any short opaque token ("h0", "node_9f3a", a company slug). It is never
# parsed for meaning: mask indices come from its position in the sorted participant list.
PARTICIPANT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def _fail(path, why):
    raise ValueError(f"{path}: {why}")


def _num(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        _fail(path, "must be finite")


def _vec(value, length, path):
    if not isinstance(value, list) or len(value) != length:
        _fail(path, f"must be a list of {length} numbers")
    for i, x in enumerate(value):
        _num(x, f"{path}[{i}]")


def _participant(value, path):
    if not isinstance(value, str) or not PARTICIPANT_ID.fullmatch(value) or value == COORDINATOR_ID:
        _fail(path, "must be a participant id (1-64 chars of letters, digits, '_', '.', '-'; "
                    f"not {COORDINATOR_ID!r})")


def is_participant_id(value) -> bool:
    """True for any string the validators accept as a participant id."""
    try:
        _participant(value, "id")
    except ValueError:
        return False
    return True


def _vec_min(value, minlen, path):
    """A weight vector whose length is set by the negotiated contract, not a constant."""
    if not isinstance(value, list) or len(value) < minlen:
        _fail(path, f"must be a list of at least {minlen} numbers")
    for i, x in enumerate(value):
        _num(x, f"{path}[{i}]")


def validate_update(msg):
    if not isinstance(msg, dict):
        _fail("update", "must be an object")
    if msg.get("v") != VERSION:
        _fail("update.v", f"must be {VERSION!r}")
    if not isinstance(msg.get("round"), int):
        _fail("update.round", "must be an int")
    _participant(msg.get("hospital"), "update.hospital")
    _vec_min(msg.get("delta_w"), 2, "update.delta_w")
    _num(msg.get("loss"), "update.loss")
    if not isinstance(msg.get("n"), int) or msg["n"] <= 0:
        _fail("update.n", "must be a positive int")
    p = msg.get("privacy")
    if not isinstance(p, dict):
        _fail("update.privacy", "must be an object")
    if not isinstance(p.get("masked"), bool):
        _fail("update.privacy.masked", "must be a bool")
    _num(p.get("dp_eps"), "update.privacy.dp_eps")
    if p.get("dp_mechanism") not in (None, "gaussian"):
        _fail("update.privacy.dp_mechanism", "must be 'gaussian'")
    _num(p.get("dp_sigma"), "update.privacy.dp_sigma")
    return msg


def validate_broadcast(msg):
    if not isinstance(msg, dict):
        _fail("broadcast", "must be an object")
    if msg.get("v") != VERSION:
        _fail("broadcast.v", f"must be {VERSION!r}")
    if not isinstance(msg.get("round"), int):
        _fail("broadcast.round", "must be an int")
    _vec_min(msg.get("w"), 2, "broadcast.w")
    if msg.get("strategy") not in ALLOWED_STRATEGIES:
        _fail("broadcast.strategy", f"must be one of {sorted(ALLOWED_STRATEGIES)}")
    _num(msg.get("eps_total"), "broadcast.eps_total")
    _num(msg.get("auc"), "broadcast.auc")
    if not (0.0 <= msg.get("auc", -1) <= 1.0):
        _fail("broadcast.auc", "must be in [0,1]")
    if not isinstance(msg.get("converged"), bool):
        _fail("broadcast.converged", "must be a bool")
    return msg


def validate_trace_update(u, participants=None):
    """Validate a trace round's summary update (player shape, not the wire message).

    `participants` (the trace's hospital node ids), when given, is the set the update's
    hospital must belong to."""
    if not isinstance(u, dict):
        _fail("trace update", "must be an object")
    _participant(u.get("hospital"), "trace update.hospital")
    if participants is not None and u["hospital"] not in participants:
        _fail("trace update.hospital", f"{u['hospital']!r} is not a hospital node of this trace")
    if not isinstance(u.get("n"), int) or u["n"] <= 0:
        _fail("trace update.n", "must be a positive int")
    if not isinstance(u.get("sent"), bool):
        _fail("trace update.sent", "must be a bool")
    if u.get("loss") is not None:
        _num(u["loss"], "trace update.loss")
    if not isinstance(u.get("masked"), bool):
        _fail("trace update.masked", "must be a bool")
    _num(u.get("dp_eps"), "trace update.dp_eps")
    if "rejected" in u and not isinstance(u["rejected"], bool):
        _fail("trace update.rejected", "must be a bool")
    if "reason" in u and not isinstance(u["reason"], str):
        _fail("trace update.reason", "must be a string")
    return u


def validate_trace(trace):
    if not isinstance(trace, dict):
        _fail("trace", "must be an object")
    if trace.get("v") != VERSION:
        _fail("trace.v", f"must be {VERSION!r}")

    meta = trace.get("meta")
    if not isinstance(meta, dict):
        _fail("trace.meta", "must be an object")
    if not isinstance(meta.get("hospitals"), int) or meta["hospitals"] < 1:
        _fail("trace.meta.hospitals", "must be a positive int")
    if not isinstance(meta.get("features"), list) or len(meta["features"]) < 1:
        _fail("trace.meta.features", "must be a non-empty list of feature names")
    if not isinstance(meta.get("rounds"), int) or meta["rounds"] < 1:
        _fail("trace.meta.rounds", "must be a positive int")

    nodes = trace.get("nodes")
    if not isinstance(nodes, list):
        _fail("trace.nodes", "must be a list")
    if any(not isinstance(n, dict) or n.get("role") not in ("hospital", "coordinator") for n in nodes):
        _fail("trace.nodes", "role must be hospital or coordinator")
    hospitals = [n.get("id") for n in nodes if n["role"] == "hospital"]
    for i, h in enumerate(hospitals):
        _participant(h, f"trace.nodes[{i}].id")
    if len(set(hospitals)) != len(hospitals):
        _fail("trace.nodes", "hospital ids must be unique")
    if len(hospitals) != meta["hospitals"]:
        _fail("trace.nodes", f"must list meta.hospitals={meta['hospitals']} hospital nodes")
    if [n.get("id") for n in nodes if n["role"] == "coordinator"] != [COORDINATOR_ID]:
        _fail("trace.nodes", f"must include exactly one coordinator node {COORDINATOR_ID!r}")
    participants = set(hospitals)

    rounds = trace.get("rounds")
    if not isinstance(rounds, list):
        _fail("trace.rounds", "must be a list")
    if len(rounds) != meta["rounds"]:
        _fail("trace.rounds", "length must equal meta.rounds")
    for r in rounds:
        validate_broadcast(r.get("broadcast"))
        for u in r.get("updates", []):
            validate_trace_update(u, participants)

    metrics = trace.get("metrics")
    if not isinstance(metrics, dict):
        _fail("trace.metrics", "must be an object")
    auc = metrics.get("auc_series")
    if not isinstance(auc, list) or len(auc) != meta["rounds"] + 1:
        _fail("trace.metrics.auc_series", "length must be rounds+1")

    attack = trace.get("attack")
    if not isinstance(attack, dict):
        _fail("trace.attack", "must be an object")
    if attack.get("target") not in participants:
        _fail("trace.attack.target", "must be a hospital node id of this trace")
    for k in ("without_privacy", "with_privacy"):
        v = attack.get(k)
        if not isinstance(v, dict) or not isinstance(v.get("recovered"), bool):
            _fail(f"trace.attack.{k}", "must have boolean 'recovered'")

    return trace
