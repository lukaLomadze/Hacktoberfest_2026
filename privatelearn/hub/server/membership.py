"""Membership rules shared by the company routes, the node routes, jobs and the seed.

The state machine is PLATFORM-API §3. Everything that moves a membership goes through
`move()`, which checks the current state inside the caller's transaction, so two
requests racing on the same membership cannot both win.
"""

from __future__ import annotations

from .db import now_iso
from .errors import invalid_transition

# Where a node's data work has got to. A later stage never moves back on its own
# (a node reconnecting must not undo a confirmation).
DATA_STAGE = {"approved": 0, "node_connected": 1, "schema_proposed": 2, "schema_confirmed": 3,
              "ready": 4, "active": 4}
WIRED = tuple(DATA_STAGE)  # states in which a node may talk to the server


def move(conn, membership_id, allowed_from, to, reason=None):
    """Transition `membership_id` to `to` if it is currently in `allowed_from`."""
    row = conn.execute("SELECT status FROM memberships WHERE id = ?", (membership_id,)).fetchone()
    if row is None or row["status"] not in allowed_from:
        current = row["status"] if row else "missing"
        raise invalid_transition(f"membership {membership_id} is {current!r}; "
                                 f"this needs one of: {', '.join(allowed_from)}")
    conn.execute("UPDATE memberships SET status = ?, status_reason = ?, updated_at = ? WHERE id = ?",
                 (to, reason, now_iso(), membership_id))


def settled_status(activated_at) -> str:
    """Where a wired-up membership rests: `active` once it has ever received a model."""
    return "active" if activated_at else "ready"


def missing_required(task_spec, mapping) -> list:
    mapped = {m["feature"] for m in mapping or []}
    return [i["name"] for i in task_spec["input_schema"] if i["required"] and i["name"] not in mapped]
