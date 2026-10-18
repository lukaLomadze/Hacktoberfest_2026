"""The job's event log, in the platform's §6 event format (privatelearn/docs/PLATFORM-API.md).

The coordinator writes one event per thing that happened (a packet, an agent's message, a
round decision, ...). The log is saved after every event to runs/<id>/events.json and,
together with the Job and Group objects, to a bundle the hub's mock serves
(privatelearn/hub/web/mock/real/job_heart.json), so #/jobs/job_heart shows the run live and
can replay it afterwards.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import threading
import time

JOB_ID = "job_heart"
GROUP_ID = "grp_heart"


def now_iso():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json_atomic(path: pathlib.Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    for attempt in range(20):  # Windows: a reader (the static server) may hold the file briefly
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


class EventLog:
    def __init__(self, run_dir: pathlib.Path, job: dict, group: dict, mirror: pathlib.Path | None = None,
                 source: dict | None = None):
        self.run_dir = run_dir
        self.job = job
        self.group = group
        self.mirror = mirror
        self.source = source or {}
        self.version = None
        self.items = []
        self.lock = threading.Lock()

    def emit(self, type_: str, round_: int = 0, **fields) -> dict:
        with self.lock:
            e = {"seq": len(self.items) + 1, "ts": now_iso(), "job_id": self.job["id"], "round": int(round_),
                 "type": type_, **fields}
            self.items.append(e)
            self._flush()
            return e

    def update_job(self, **fields):
        with self.lock:
            self.job.update(fields)
            self._flush()

    def set_version(self, version: dict):
        with self.lock:
            self.version = version
            self._flush()

    def _flush(self):
        write_json_atomic(self.run_dir / "events.json", {"items": self.items})
        write_json_atomic(self.run_dir / "job.json", self.job)
        if self.mirror is not None:
            bundle = {"source": self.source, "job": self.job, "group": self.group, "version": self.version,
                      "events": self.items}
            try:
                write_json_atomic(self.mirror, bundle)
                index = self.mirror.parent / "index.json"
                try:
                    jobs = json.loads(index.read_text(encoding="utf-8")).get("jobs", [])
                except (OSError, ValueError):
                    jobs = []
                if self.job["id"] not in jobs:
                    write_json_atomic(index, {"jobs": jobs + [self.job["id"]]})
            except OSError as e:  # the hub mirror is a convenience; never fail the run over it
                print(f"[events] could not update {self.mirror}: {e}", flush=True)
