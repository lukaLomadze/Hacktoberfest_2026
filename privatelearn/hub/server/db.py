"""SQLite storage for the platform server. Stdlib `sqlite3` only.

One connection is shared by every thread (FastAPI runs sync routes on a thread pool,
and the federation emits events from its own threads), guarded by a re-entrant lock.
`tx()` opens `BEGIN IMMEDIATE`, so a read-check-write sequence inside it (for example
"is this membership still `ready`? then move it to `training`") cannot interleave with
another thread's. Nested `tx()` calls join the outer transaction.

The whole schema is created up front so later tasks never need a migration. JSON
columns are TEXT written with `dumps` (NaN/Infinity rejected, so the file stays valid
JSON for the frontend). Partial unique indexes enforce two invariants in the database
itself, not only in the routes:
  - a company has at most one open membership per group (rejected/left do not count);
  - a group has at most one queued/running job.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import weakref
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    name_key        TEXT NOT NULL UNIQUE,          -- casefolded name, for uniqueness
    field           TEXT NOT NULL,
    contact_email   TEXT NOT NULL,
    status          TEXT NOT NULL,
    status_reason   TEXT,
    wallet_address  TEXT,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    email       TEXT NOT NULL UNIQUE,              -- stored lower-cased
    role        TEXT NOT NULL CHECK (role IN ('admin', 'company')),
    company_id  TEXT REFERENCES companies(id),
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash      TEXT PRIMARY KEY,              -- sha256 of the bearer token
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at      TEXT NOT NULL,
    expires_epoch   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ml_groups (            -- "groups" is an SQL keyword
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL,
    status      TEXT NOT NULL,
    task_spec   TEXT NOT NULL,
    pricing     TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memberships (
    id                  TEXT PRIMARY KEY,
    company_id          TEXT NOT NULL REFERENCES companies(id),
    group_id            TEXT NOT NULL REFERENCES ml_groups(id),
    status              TEXT NOT NULL,
    status_reason       TEXT,
    node_token_hash     TEXT UNIQUE,
    data                TEXT,                      -- {row_count, columns, profile}
    schema_             TEXT,                      -- {label, mapping, reasoning, made_by, confirmed, ...}
    validation          TEXT,                      -- {ok, checks}
    contribution_rows   INTEGER NOT NULL DEFAULT 0,
    contribution_rounds INTEGER NOT NULL DEFAULT 0,
    activated_at        TEXT,                      -- first time it received a model
    requested_at        TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS one_open_membership
    ON memberships (company_id, group_id) WHERE status NOT IN ('rejected', 'left');
CREATE TABLE IF NOT EXISTS nodes (
    id              TEXT PRIMARY KEY,
    membership_id   TEXT NOT NULL UNIQUE REFERENCES memberships(id),
    node_version    TEXT NOT NULL,
    agent           TEXT NOT NULL,
    connector       TEXT NOT NULL,
    last_seen_epoch REAL NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_versions (
    id                  TEXT PRIMARY KEY,
    group_id            TEXT NOT NULL REFERENCES ml_groups(id),
    version             INTEGER NOT NULL,
    parent_version_id   TEXT REFERENCES model_versions(id),
    job_id              TEXT NOT NULL,
    model_type          TEXT NOT NULL,
    features            TEXT NOT NULL,
    norm                TEXT NOT NULL,
    weights             TEXT NOT NULL,
    metrics             TEXT NOT NULL,
    contributors        TEXT NOT NULL,
    sha256              TEXT NOT NULL,
    synthetic           INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    UNIQUE (group_id, version)
);
CREATE TABLE IF NOT EXISTS jobs (
    id                  TEXT PRIMARY KEY,
    group_id            TEXT NOT NULL REFERENCES ml_groups(id),
    kind                TEXT NOT NULL,
    status              TEXT NOT NULL,
    step                TEXT NOT NULL,
    round               INTEGER NOT NULL DEFAULT 0,
    rounds              INTEGER NOT NULL,
    parent_version_id   TEXT REFERENCES model_versions(id),
    metrics             TEXT NOT NULL,
    error               TEXT,
    created_at          TEXT NOT NULL,
    started_at          TEXT,
    finished_at         TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS one_live_job
    ON jobs (group_id) WHERE status IN ('queued', 'running');
CREATE TABLE IF NOT EXISTS job_participants (
    job_id          TEXT NOT NULL REFERENCES jobs(id),
    membership_id   TEXT NOT NULL REFERENCES memberships(id),
    node_id         TEXT,
    company_id      TEXT NOT NULL,
    company_name    TEXT NOT NULL,
    color           TEXT NOT NULL,
    position        INTEGER NOT NULL,
    state           TEXT NOT NULL,
    rows            INTEGER NOT NULL DEFAULT 0,
    rounds          INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (job_id, membership_id)
);
CREATE TABLE IF NOT EXISTS events (
    job_id  TEXT NOT NULL REFERENCES jobs(id),
    seq     INTEGER NOT NULL,
    ts      TEXT NOT NULL,
    type    TEXT NOT NULL,
    body    TEXT NOT NULL,                         -- the full event object
    PRIMARY KEY (job_id, seq)
);
CREATE TABLE IF NOT EXISTS usage_events (
    id          TEXT PRIMARY KEY,
    company_id  TEXT NOT NULL REFERENCES companies(id),
    group_id    TEXT REFERENCES ml_groups(id),
    kind        TEXT NOT NULL,
    units       REAL NOT NULL,
    amount      REAL,
    currency    TEXT,
    payment_ref TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_by_company ON usage_events (company_id, created_at);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


def dumps(obj) -> str:
    return json.dumps(obj, allow_nan=False, separators=(",", ":"), ensure_ascii=False)


def loads(text):
    return None if text is None else json.loads(text)


class Database:
    def __init__(self, path):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._local = threading.local()
        # close the connection when the Database is dropped without close() (tests, app reloads)
        self._finalizer = weakref.finalize(self, self._conn.close)
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(SCHEMA)

    @contextmanager
    def tx(self):
        """A write transaction. Re-entrant: an inner tx() joins the outer one.

        An exception that escapes the outermost tx() rolls everything back.
        """
        with self._lock:
            depth = getattr(self._local, "depth", 0)
            if depth == 0:
                self._conn.execute("BEGIN IMMEDIATE")
            self._local.depth = depth + 1
            try:
                yield self._conn
            except BaseException:
                self._local.depth = depth
                if depth == 0:
                    self._conn.execute("ROLLBACK")
                raise
            self._local.depth = depth
            if depth == 0:
                try:
                    self._conn.execute("COMMIT")
                except BaseException:
                    self._conn.execute("ROLLBACK")  # never leave a transaction open
                    raise

    def one(self, sql, args=()):
        with self._lock:
            row = self._conn.execute(sql, args).fetchone()
            return dict(row) if row is not None else None

    def all(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def execute(self, sql, args=()):
        """One statement in its own transaction (or the caller's, inside tx())."""
        with self.tx() as conn:
            return conn.execute(sql, args).rowcount

    def close(self):
        with self._lock:
            self._finalizer()
