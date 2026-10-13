from __future__ import annotations

import logging
import os
import shutil
import signal
import threading
import time
from pathlib import Path
from types import FrameType

from prometheus_client import start_http_server

from bagcounter.cameras import CameraManager
from bagcounter.config import Settings, get_settings
from bagcounter.detect.openvino import Detector
from bagcounter.errors import AppError, JobCancelledError, ModelError
from bagcounter.geometry import line_from_dict
from bagcounter.jobs import run_video
from bagcounter.logging import job_id as job_ctx
from bagcounter.logging import setup_logging
from bagcounter.metrics import (
    DATA_BYTES,
    DISK_FREE,
    JOB_SECONDS,
    JOBS,
    MODEL,
    QUEUE_DEPTH,
    REAPED,
    SELFTEST_OK,
    SELFTEST_TS,
    WORKER_HEARTBEAT,
)
from bagcounter.preflight import run_preflight, selftest
from bagcounter.store import Job, Store
from bagcounter.video.sources import FileSource

log = logging.getLogger("bagcounter.worker")

CAMERA_SYNC_S = 2.0
HEARTBEAT_S = 5.0
SHUTDOWN_GRACE_S = 20.0


def upload_path(settings: Settings, upload_id: str) -> Path:
    return settings.uploads_dir / f"{upload_id}.video"


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


class Worker:
    def __init__(self, settings: Settings, store: Store, detector: Detector) -> None:
        self.s, self.store, self.det = settings, store, detector
        self.stop = threading.Event()
        self.exit_code = 0
        self.loop_alive_at = time.monotonic()
        self.cameras = CameraManager(detector, store, settings)

    def run(self) -> int:
        WORKER_HEARTBEAT.set(time.time())
        DISK_FREE.set(shutil.disk_usage(self.s.data_dir).free)
        requeued, quarantined = self.store.recover_interrupted()
        if requeued or quarantined:
            log.warning("recovered interrupted jobs", extra={"requeued": requeued, "quarantined": quarantined})
        threading.Thread(target=self._housekeeping, name="housekeeping", daemon=True).start()
        threading.Thread(target=self._camera_loop, name="cameras", daemon=True).start()
        while not self.stop.is_set():
            self.loop_alive_at = time.monotonic()
            job = self.store.claim_job()
            if job is None:
                self.stop.wait(self.s.worker_poll_s)
                continue
            self._run_job(job)
        self.cameras.stop_all()
        return self.exit_code

    def _run_job(self, job: Job) -> None:
        token = job_ctx.set(job.id)
        started = time.monotonic()
        out_dir = self.s.outputs_dir / job.id
        cancel = False

        def progress(frac: float, count: int) -> None:
            nonlocal cancel
            self.loop_alive_at = time.monotonic()
            cancel = self.store.update_progress(job.id, frac, count)

        log.info("job started", extra={"attempt": job.attempts})
        try:
            line = line_from_dict(job.line)
            src = FileSource(upload_path(self.s, job.upload_id), max_seconds=self.s.max_video_seconds,
                             max_pixels=self.s.max_video_pixels)
            summary = run_video(src, line, self.s, self.det, out_dir, annotate=job.annotate, progress=progress,
                                cancelled=lambda: cancel or self.stop.is_set())
            self.store.finish_job(job.id, "succeeded", count=int(summary["count"]),  # type: ignore[call-overload]
                                  summary=summary, model_version=self.det.version)
            JOBS.labels(outcome="succeeded").inc()
            log.info("job succeeded", extra={"count": summary["count"], "seconds": round(time.monotonic() - started)})
        except JobCancelledError:
            if self.stop.is_set():
                log.warning("job interrupted by shutdown; will be retried")
            else:
                self.store.finish_job(job.id, "cancelled")
                JOBS.labels(outcome="cancelled").inc()
                log.info("job cancelled")
        except AppError as e:
            self.store.finish_job(job.id, "failed", error_code=e.code, error_message=e.message,
                                  model_version=self.det.version)
            JOBS.labels(outcome="failed").inc()
            log.error("job failed", exc_info=True, extra={"code": e.code})
            if isinstance(e, ModelError):
                self._fatal("inference failed during a job")
        except Exception as e:
            self.store.finish_job(job.id, "failed", error_code="INTERNAL", error_message=type(e).__name__,
                                  model_version=self.det.version)
            JOBS.labels(outcome="failed").inc()
            log.error("job failed with an unexpected error", exc_info=True)
        finally:
            JOB_SECONDS.observe(time.monotonic() - started)
            if not self.stop.is_set():
                upload_path(self.s, job.upload_id).unlink(missing_ok=True)
            job_ctx.reset(token)

    def _fatal(self, reason: str) -> None:
        log.critical("worker stopping", extra={"reason": reason})
        SELFTEST_OK.set(0)
        self.store.set_meta("selftest", {"ok": False, "reason": reason})
        self.exit_code = 1
        self.stop.set()

    def _camera_loop(self) -> None:
        while not self.stop.wait(CAMERA_SYNC_S):
            try:
                self.cameras.sync()
            except AppError:
                log.exception("camera sync failed")

    def _housekeeping(self) -> None:
        last_reap = last_selftest = 0.0
        while not self.stop.wait(HEARTBEAT_S):
            now = time.monotonic()
            try:
                if now - self.loop_alive_at < self.s.job_stall_s:
                    self.store.set_meta("worker_heartbeat", {"ts": time.time()})
                    WORKER_HEARTBEAT.set(time.time())
                else:
                    log.critical("job loop stalled", extra={"silent_s": round(now - self.loop_alive_at)})
                QUEUE_DEPTH.set(self.store.queued_count())
                DISK_FREE.set(shutil.disk_usage(self.s.data_dir).free)
                if now - last_selftest >= self.s.selftest_interval_s:
                    last_selftest = now
                    self._selftest()
                if now - last_reap >= self.s.reap_interval_s:
                    last_reap = now
                    self.reap()
            except ModelError:
                self._fatal("periodic self-test failed")
            except (AppError, OSError):
                log.exception("housekeeping step failed")

    def _selftest(self) -> None:
        detail = selftest(self.det, self.s)
        SELFTEST_OK.set(1)
        SELFTEST_TS.set(time.time())
        self.store.set_meta("selftest", {"ok": True, "detail": detail})

    def reap(self) -> None:
        now = time.time()
        for uid in self.store.expired_uploads(now - self.s.retain_uploads_s):
            upload_path(self.s, uid).unlink(missing_ok=True)
            (self.s.uploads_dir / f"{uid}.jpg").unlink(missing_ok=True)
            self.store.delete_upload(uid)
            REAPED.labels(kind="upload").inc()
        for jid in self.store.expired_job_outputs(now - self.s.retain_outputs_s):
            if (self.s.outputs_dir / jid).exists():
                shutil.rmtree(self.s.outputs_dir / jid)
            self.store.mark_outputs_expired(jid)
            REAPED.labels(kind="job_output").inc()
        for snap in self.store.delete_events_before(now - self.s.retain_events_s):
            (self.s.data_dir / snap).unlink(missing_ok=True)
            REAPED.labels(kind="snapshot").inc()
        cams = self.s.data_dir / "cameras"
        if cams.exists():
            for d in cams.glob("*/snapshots/*"):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
        for kind, path in (("uploads", self.s.uploads_dir), ("outputs", self.s.outputs_dir), ("cameras", cams)):
            DATA_BYTES.labels(kind=kind).set(_dir_bytes(path))

    def sweep_orphans(self) -> None:
        known = set()
        for f in self.s.uploads_dir.glob("*"):
            uid = f.name.split(".")[0]
            if uid in known:
                continue
            try:
                self.store.get_upload(uid)
                known.add(uid)
            except AppError:
                f.unlink(missing_ok=True)
                REAPED.labels(kind="orphan").inc()
        cams = self.s.data_dir / "cameras"
        for cid_dir in cams.glob("*") if cams.exists() else []:
            try:
                self.store.get_camera(cid_dir.name)
            except AppError:
                shutil.rmtree(cid_dir)
                REAPED.labels(kind="orphan").inc()


def main() -> int:
    settings = get_settings()
    setup_logging(settings.log_level, settings.error_log_window_s)
    log.info("worker starting", extra={"config": settings.redacted()})
    pf = run_preflight(settings)
    if not pf.ok or pf.detector is None:
        log.critical("preflight failed; refusing to start",
                     extra={"failed": [c.name for c in pf.checks if not c.ok]})
        return 1
    det = pf.detector
    store = Store(settings.db_path)
    MODEL.info({"name": det.info.name, "version": det.info.version})
    SELFTEST_OK.set(1)
    SELFTEST_TS.set(time.time())
    store.set_meta("selftest", {"ok": True, "detail": "boot"})
    store.set_meta("model", {"name": det.info.name, "version": det.info.version, "metrics": det.info.meta.get(
        "metrics")})
    start_http_server(settings.worker_metrics_port)
    w = Worker(settings, store, det)
    w.sweep_orphans()

    def on_signal(signum: int, _frame: FrameType | None) -> None:
        log.info("shutdown requested", extra={"signal": signum})
        w.stop.set()
        t = threading.Timer(SHUTDOWN_GRACE_S, _force_exit)
        t.daemon = True
        t.start()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    return w.run()


def _force_exit() -> None:
    log.critical("graceful shutdown deadline passed; forcing exit")
    os._exit(1)
