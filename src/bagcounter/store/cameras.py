from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from bagcounter.errors import CameraLimitError, NotFoundError
from bagcounter.store.db import Database, new_id


@dataclass(frozen=True)
class Camera:
    id: str
    created_at: float
    name: str
    url: str
    line: dict[str, Any] | None
    enabled: bool
    revision: int
    state: str
    count_total: int
    status: dict[str, Any] | None
    updated_at: float | None


def _camera(r: sqlite3.Row) -> Camera:
    return Camera(r["id"], r["created_at"], r["name"], r["url"],
                  json.loads(r["line_json"]) if r["line_json"] else None, bool(r["enabled"]), r["revision"],
                  r["state"], r["count_total"], json.loads(r["status_json"]) if r["status_json"] else None,
                  r["updated_at"])


class CameraStore(Database):
    def add_camera(self, name: str, url: str, line: dict[str, object] | None, max_cameras: int) -> Camera:
        cid = new_id()
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
            if n >= max_cameras:
                raise CameraLimitError("camera limit reached for this box", context={"max_cameras": max_cameras})
            c.execute("INSERT INTO cameras (id, created_at, name, url, line_json) VALUES (?, ?, ?, ?, ?)",
                      (cid, time.time(), name, url, json.dumps(line) if line else None))
        return self.get_camera(cid)

    def get_camera(self, camera_id: str) -> Camera:
        with self._conn() as c:
            r = c.execute("SELECT * FROM cameras WHERE id=?", (camera_id,)).fetchone()
        if r is None:
            raise NotFoundError("camera not found", context={"camera_id": camera_id})
        return _camera(r)

    def list_cameras(self) -> list[Camera]:
        with self._conn() as c:
            return [_camera(r) for r in c.execute("SELECT * FROM cameras ORDER BY created_at")]

    def update_camera(self, camera_id: str, *, name: str | None = None, url: str | None = None,
                      line: dict[str, object] | None = None, enabled: bool | None = None) -> Camera:
        sets, args = [], []
        for col, val in (("name", name), ("url", url), ("enabled", None if enabled is None else int(enabled)),
                         ("line_json", None if line is None else json.dumps(line))):
            if val is not None:
                sets.append(f"{col}=?")
                args.append(val)
        if sets:
            with self._conn() as c:
                n = c.execute(f"UPDATE cameras SET {', '.join(sets)}, revision=revision+1 WHERE id=?",  # noqa: S608
                              (*args, camera_id)).rowcount
            if n == 0:
                raise NotFoundError("camera not found", context={"camera_id": camera_id})
        return self.get_camera(camera_id)

    def delete_camera(self, camera_id: str) -> None:
        with self._conn() as c:
            if c.execute("DELETE FROM cameras WHERE id=?", (camera_id,)).rowcount == 0:
                raise NotFoundError("camera not found", context={"camera_id": camera_id})

    def delete_camera_events(self, camera_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM events WHERE camera_id=?", (camera_id,))

    def set_camera_status(self, camera_id: str, state: str, status: dict[str, object]) -> None:
        with self._conn() as c:
            c.execute("UPDATE cameras SET state=?, status_json=?, updated_at=? WHERE id=?",
                      (state, json.dumps(status), time.time(), camera_id))

    def add_events(self, camera_id: str, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        with self._conn() as c:
            c.executemany("INSERT INTO events (camera_id, ts, track_id, direction, confidence, late, snapshot, "
                          "model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                          [(camera_id, e["ts"], e["track_id"], e["direction"], e["confidence"], int(e["late"]),
                            e["snapshot"], e["model_version"]) for e in events])
            c.execute("UPDATE cameras SET count_total=count_total+? WHERE id=?", (len(events), camera_id))

    def list_events(self, camera_id: str, since: float, until: float, limit: int) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM events WHERE camera_id=? AND ts>=? AND ts<? ORDER BY ts DESC LIMIT ?",
                             (camera_id, since, until, limit))
            return [dict(r) for r in rows]

    def count_buckets(self, camera_id: str, since: float, until: float, bucket_s: int) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT CAST(ts / ? AS INTEGER) * ? AS start, COUNT(*) AS count FROM events "
                             "WHERE camera_id=? AND ts>=? AND ts<? GROUP BY start ORDER BY start",
                             (bucket_s, bucket_s, camera_id, since, until))
            return [dict(r) for r in rows]

    def delete_events_before(self, cutoff: float) -> list[str]:
        with self._conn() as c:
            snaps = [r["snapshot"] for r in c.execute("SELECT snapshot FROM events WHERE ts<? AND snapshot IS NOT "
                                                      "NULL", (cutoff,))]
            c.execute("DELETE FROM events WHERE ts<?", (cutoff,))
        return snaps
