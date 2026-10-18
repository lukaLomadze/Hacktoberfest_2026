"""Node liveness (PLATFORM-API §7: polling is the heartbeat).

A node is online while one of its long-polls is open, or within `node_offline_s` of the
last request it made. Open polls are tracked in memory: they only matter while this
server process is running, and after a restart every node simply polls again.
"""

from __future__ import annotations

import threading
import time


class NodeRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._open = {}

    def poll_started(self, node_id: str) -> None:
        with self._lock:
            self._open[node_id] = self._open.get(node_id, 0) + 1

    def poll_ended(self, node_id: str) -> None:
        with self._lock:
            n = self._open.get(node_id, 0) - 1
            if n > 0:
                self._open[node_id] = n
            else:
                self._open.pop(node_id, None)

    def polling(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._open


def node_status(hub, node_row, membership_status: str):
    """offline | online | busy, or None when no node ever said hello."""
    if node_row is None:
        return None
    recent = time.time() - node_row["last_seen_epoch"] <= hub.settings.node_offline_s
    if not (recent or hub.nodes.polling(node_row["id"])):
        return "offline"
    return "busy" if membership_status == "training" else "online"
