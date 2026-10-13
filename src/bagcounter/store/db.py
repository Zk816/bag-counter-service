from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from bagcounter.errors import StorageError

BUSY_TIMEOUT_S = 5.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
  id TEXT PRIMARY KEY, created_at REAL NOT NULL, filename TEXT NOT NULL, bytes INTEGER NOT NULL,
  info_json TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, created_at REAL NOT NULL, upload_id TEXT NOT NULL, filename TEXT NOT NULL,
  line_json TEXT NOT NULL, annotate INTEGER NOT NULL, status TEXT NOT NULL,
  progress REAL NOT NULL DEFAULT 0, count INTEGER, attempts INTEGER NOT NULL DEFAULT 0,
  started_at REAL, finished_at REAL, error_code TEXT, error_message TEXT, model_version TEXT,
  summary_json TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0, outputs_expired INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at);
CREATE TABLE IF NOT EXISTS cameras (
  id TEXT PRIMARY KEY, created_at REAL NOT NULL, name TEXT NOT NULL, url TEXT NOT NULL, line_json TEXT,
  enabled INTEGER NOT NULL DEFAULT 1, revision INTEGER NOT NULL DEFAULT 1, state TEXT NOT NULL DEFAULT 'pending',
  count_total INTEGER NOT NULL DEFAULT 0, status_json TEXT, updated_at REAL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL, ts REAL NOT NULL, track_id INTEGER NOT NULL,
  direction INTEGER NOT NULL, confidence REAL NOT NULL, late INTEGER NOT NULL, snapshot TEXT,
  model_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_camera_ts ON events(camera_id, ts);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL NOT NULL);
"""


def new_id() -> str:
    return uuid.uuid4().hex[:16]


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_S, isolation_level=None)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(SCHEMA)
            finally:
                conn.close()
        except sqlite3.Error as e:
            raise StorageError("cannot initialise the database", context={"path": str(path)}) from e

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_S, isolation_level=None)
        except sqlite3.Error as e:
            raise StorageError("cannot open the database") from e
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except sqlite3.Error as e:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise StorageError("database operation failed", context={"error": str(e)}) from e
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def ping(self) -> None:
        with self._conn() as c:
            c.execute("SELECT 1")

    def set_meta(self, key: str, value: object) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO meta VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                      "updated_at=excluded.updated_at", (key, json.dumps(value), time.time()))

    def get_meta(self, key: str) -> tuple[Any, float] | None:
        with self._conn() as c:
            r = c.execute("SELECT value, updated_at FROM meta WHERE key=?", (key,)).fetchone()
        return (json.loads(r["value"]), r["updated_at"]) if r else None
