from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from bagcounter.errors import ConflictError, NotFoundError, QueueFullError
from bagcounter.store.db import Database, new_id

MAX_JOB_ATTEMPTS = 2


@dataclass(frozen=True)
class Job:
    id: str
    created_at: float
    upload_id: str
    filename: str
    line: dict[str, Any]
    annotate: bool
    status: str
    progress: float
    count: int | None
    attempts: int
    started_at: float | None
    finished_at: float | None
    error_code: str | None
    error_message: str | None
    model_version: str | None
    summary: dict[str, Any] | None
    cancel_requested: bool
    outputs_expired: bool


def _job(r: sqlite3.Row) -> Job:
    return Job(r["id"], r["created_at"], r["upload_id"], r["filename"], json.loads(r["line_json"]),
               bool(r["annotate"]), r["status"], r["progress"], r["count"], r["attempts"], r["started_at"],
               r["finished_at"], r["error_code"], r["error_message"], r["model_version"],
               json.loads(r["summary_json"]) if r["summary_json"] else None, bool(r["cancel_requested"]),
               bool(r["outputs_expired"]))


class JobStore(Database):
    def add_upload(self, upload_id: str, filename: str, size: int, info: dict[str, object]) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO uploads VALUES (?, ?, ?, ?, ?, 0)",
                      (upload_id, time.time(), filename, size, json.dumps(info)))

    def get_upload(self, upload_id: str) -> dict[str, Any]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone()
        if r is None:
            raise NotFoundError("upload not found", context={"upload_id": upload_id})
        return {**dict(r), "info": json.loads(r["info_json"])}

    def expired_uploads(self, older_than: float) -> list[str]:
        with self._conn() as c:
            rows = c.execute("SELECT u.id FROM uploads u WHERE u.created_at < ? AND NOT EXISTS (SELECT 1 FROM jobs j "
                             "WHERE j.upload_id = u.id AND j.status IN ('queued', 'running'))", (older_than,))
            return [r["id"] for r in rows]

    def delete_upload(self, upload_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM uploads WHERE id=?", (upload_id,))

    def create_job(self, upload_id: str, line: dict[str, object], annotate: bool, max_queued: int) -> Job:
        job_id = new_id()
        with self._conn() as c:
            up = c.execute("SELECT filename, used FROM uploads WHERE id=?", (upload_id,)).fetchone()
            if up is None:
                raise NotFoundError("upload not found", context={"upload_id": upload_id})
            if up["used"]:
                raise ConflictError("upload already has a job", context={"upload_id": upload_id})
            waiting = c.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0]
            if waiting >= max_queued:
                raise QueueFullError("too many jobs waiting", context={"queued": waiting})
            c.execute("UPDATE uploads SET used=1 WHERE id=?", (upload_id,))
            c.execute("INSERT INTO jobs (id, created_at, upload_id, filename, line_json, annotate, status) "
                      "VALUES (?, ?, ?, ?, ?, ?, 'queued')",
                      (job_id, time.time(), upload_id, up["filename"], json.dumps(line), int(annotate)))
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Job:
        with self._conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if r is None:
            raise NotFoundError("job not found", context={"job_id": job_id})
        return _job(r)

    def list_jobs(self, limit: int) -> list[Job]:
        with self._conn() as c:
            return [_job(r) for r in c.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))]

    def queued_count(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0])

    def claim_job(self) -> Job | None:
        with self._conn() as c:
            r = c.execute("UPDATE jobs SET status='running', started_at=?, attempts=attempts+1 WHERE id = "
                          "(SELECT id FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1) RETURNING *",
                          (time.time(),)).fetchone()
        return _job(r) if r else None

    def recover_interrupted(self) -> tuple[int, int]:
        with self._conn() as c:
            q = c.execute("UPDATE jobs SET status='failed', finished_at=?, error_code='WORKER_CRASHED', "
                          "error_message='the worker stopped twice while processing this video; not retried' "
                          "WHERE status='running' AND attempts >= ?", (time.time(), MAX_JOB_ATTEMPTS)).rowcount
            r = c.execute("UPDATE jobs SET status='queued', progress=0 WHERE status='running'").rowcount
        return r, q

    def update_progress(self, job_id: str, progress: float, count: int) -> bool:
        with self._conn() as c:
            c.execute("UPDATE jobs SET progress=?, count=? WHERE id=? AND status='running'", (progress, count, job_id))
            r = c.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        return bool(r and r["cancel_requested"])

    def finish_job(self, job_id: str, status: str, *, count: int | None = None, summary: dict[str, object] | None
                   = None, model_version: str | None = None, error_code: str | None = None,
                   error_message: str | None = None) -> None:
        with self._conn() as c:
            c.execute("UPDATE jobs SET status=?, finished_at=?, progress=CASE WHEN ?='succeeded' THEN 1 ELSE progress "
                      "END, count=COALESCE(?, count), summary_json=?, model_version=?, error_code=?, error_message=? "
                      "WHERE id=? AND status='running'",
                      (status, time.time(), status, count, json.dumps(summary) if summary else None, model_version,
                       error_code, error_message, job_id))

    def request_cancel(self, job_id: str) -> Job:
        with self._conn() as c:
            c.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE id=? AND status='queued'",
                      (time.time(), job_id))
            c.execute("UPDATE jobs SET cancel_requested=1 WHERE id=? AND status='running'", (job_id,))
        return self.get_job(job_id)

    def expired_job_outputs(self, older_than: float) -> list[str]:
        with self._conn() as c:
            rows = c.execute("SELECT id FROM jobs WHERE finished_at < ? AND outputs_expired=0", (older_than,))
            return [r["id"] for r in rows]

    def mark_outputs_expired(self, job_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE jobs SET outputs_expired=1 WHERE id=?", (job_id,))
