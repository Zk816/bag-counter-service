from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from bagcounter.config import Settings
from bagcounter.count.counter import CountEvent
from bagcounter.detect.openvino import Detector, Frame
from bagcounter.errors import AppError, InvalidLineError
from bagcounter.geometry import Line, line_from_dict
from bagcounter.logging import camera_id as camera_ctx
from bagcounter.metrics import (
    BAGS_COUNTED,
    CAMERA_INFO,
    DETECTION_SCORE,
    FRAME_LATENCY,
    FRAMES_DECODED,
    FRAMES_DROPPED,
    FRAMES_PROCESSED,
    LAST_FRAME_TS,
    STREAM_RECONNECTS,
    STREAM_UP,
    TRACKS_CONFIRMED,
)
from bagcounter.pipeline import CountingPipeline
from bagcounter.sinks import draw, save_snapshot
from bagcounter.store import Camera, Store
from bagcounter.video.sources import RtspSource, StreamState

log = logging.getLogger(__name__)

STOP_TIMEOUT_S = 10.0
LATENCY_WINDOW = 300


def write_atomic_jpeg(path: Path, img: Frame) -> None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(path)


def camera_dir(settings: Settings, cid: str) -> Path:
    return settings.data_dir / "cameras" / cid


class CameraRunner(threading.Thread):
    def __init__(self, cam: Camera, detector: Detector, store: Store, settings: Settings) -> None:
        super().__init__(name=f"camera-{cam.id}", daemon=True)
        self.cam, self.det, self.store, self.s = cam, detector, store, settings
        self.revision = cam.revision
        self._stop_evt = threading.Event()
        self.dir = camera_dir(settings, cam.id)
        self.line: Line | None = None
        if cam.line:
            try:
                self.line = line_from_dict(cam.line)
            except InvalidLineError as e:
                log.warning("camera line invalid, counting disabled", extra={"camera": cam.id, "error": str(e)})

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        camera_ctx.set(self.cam.id)
        self.dir.mkdir(parents=True, exist_ok=True)
        CAMERA_INFO.labels(stream=self.cam.id, name=self.cam.name[:80]).set(1)
        src = RtspSource(self.cam.url, open_timeout_s=self.s.rtsp_open_timeout_s,
                         read_timeout_s=self.s.rtsp_read_timeout_s, backoff_min_s=self.s.reconnect_backoff_min_s,
                         backoff_max_s=self.s.reconnect_backoff_max_s, stall_after_s=self.s.stall_after_s,
                         frozen_after_s=self.s.frozen_after_s).start()
        try:
            self._loop(src)
        except AppError as e:
            log.exception("camera pipeline failed", extra={"code": e.code})
            self.store.set_camera_status(self.cam.id, "error", {"error": e.code, "message": e.message})
        finally:
            src.stop(STOP_TIMEOUT_S)
            for st in StreamState:
                STREAM_UP.labels(stream=self.cam.id, state=st.value).set(0)
            CAMERA_INFO.remove(self.cam.id, self.cam.name[:80])

    def _loop(self, src: RtspSource) -> None:
        cid = self.cam.id
        pipe: CountingPipeline | None = None
        shape: tuple[int, int] | None = None
        interval = 1.0 / self.s.target_fps
        preview_every = 1.0 / self.s.preview_fps
        last_proc = last_preview = last_flush = 0.0
        latencies: list[float] = []
        processed = 0
        started = time.time()
        wall0 = time.time() - (time.monotonic() - src.started_at)
        reported_tracks = 0
        while not self._stop_evt.is_set():
            now = time.monotonic()
            if now - last_flush >= self.s.camera_state_flush_s:
                last_flush = now
                self._flush_status(src, pipe, processed, latencies, started)
            wait = interval - (time.monotonic() - last_proc)
            if wait > 0 and self._stop_evt.wait(wait):
                return
            pkt = src.take(timeout_s=interval)
            if pkt is None:
                continue
            last_proc = time.monotonic()
            img = pkt.image
            if shape != img.shape[:2]:
                shape = (img.shape[0], img.shape[1])
                pipe = None if self.line is None else CountingPipeline(
                    self.det, self.line, shape[1], shape[0], self.s.target_fps, self.s, stream=cid)
            tracks, count, flash = [], 0, set()
            if pipe is not None:
                res = pipe.process(img, pkt.t_s)
                processed += 1
                FRAMES_PROCESSED.labels(stream=cid).inc()
                for t in res.tracks:
                    DETECTION_SCORE.labels(stream=cid).observe(t.score)
                if pipe.tracker.confirmed_total > reported_tracks:
                    TRACKS_CONFIRMED.labels(stream=cid).inc(pipe.tracker.confirmed_total - reported_tracks)
                    reported_tracks = pipe.tracker.confirmed_total
                if res.events:
                    self._record(res.events, img, wall0)
                    flash = {e.track_id for e in res.events}
                tracks, count = res.tracks, pipe.counter.count
                lat = time.monotonic() - (src.started_at + pkt.t_s)
                FRAME_LATENCY.labels(stream=cid).observe(lat)
                latencies.append(lat)
                if len(latencies) > LATENCY_WINDOW:
                    del latencies[: LATENCY_WINDOW // 2]
            LAST_FRAME_TS.labels(stream=cid).set(time.time())
            if time.monotonic() - last_preview >= preview_every:
                last_preview = time.monotonic()
                frame = draw(img, pipe.pixel_line, tracks, count, flash) if pipe else img
                write_atomic_jpeg(self.dir / "preview.jpg", frame)

    def _record(self, events: list[CountEvent], img: Frame, wall0: float) -> None:
        day = time.strftime("%Y%m%d")
        snap_dir = self.dir / "snapshots" / day
        snap_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for ev in events:
            ts = wall0 + ev.t_s
            name = save_snapshot(img, ev, snap_dir, self.s.snapshot_max_side)
            final = snap_dir / f"{int(ts * 1000)}_{ev.track_id}.jpg"
            (snap_dir / name).rename(final)
            rows.append({"ts": ts, "track_id": ev.track_id, "direction": ev.direction,
                         "confidence": round(ev.confidence, 3), "late": ev.late,
                         "snapshot": str(final.relative_to(self.s.data_dir)), "model_version": self.det.version})
            BAGS_COUNTED.labels(stream=self.cam.id, late=str(ev.late).lower()).inc()
        self.store.add_events(self.cam.id, rows)

    def _flush_status(self, src: RtspSource, pipe: CountingPipeline | None, processed: int, latencies: list[float],
                      started: float) -> None:
        st = src.state
        state = st.value if (pipe is not None or st is not StreamState.LIVE) else "needs_line"
        for s in StreamState:
            STREAM_UP.labels(stream=self.cam.id, state=s.value).set(1 if s is st else 0)
        FRAMES_DECODED.labels(stream=self.cam.id).set(src.stats.decoded)
        FRAMES_DROPPED.labels(stream=self.cam.id).set(src.stats.dropped)
        STREAM_RECONNECTS.labels(stream=self.cam.id).set(src.stats.reconnects)
        elapsed = max(1e-6, time.time() - started)
        self.store.set_camera_status(self.cam.id, state, {
            "width": src.stats.width, "height": src.stats.height,
            "decoded_fps": round(src.stats.decoded / elapsed, 1),
            "processed_fps": round(processed / elapsed, 1),
            "dropped_share": round(src.stats.dropped / max(1, src.stats.decoded), 3),
            "latency_ms_p50": round(1000 * float(np.percentile(latencies, 50)), 1) if latencies else None,
            "latency_ms_p95": round(1000 * float(np.percentile(latencies, 95)), 1) if latencies else None,
            "reconnects": src.stats.reconnects,
            "session_count": pipe.counter.count if pipe else 0,
            "tracks_confirmed": pipe.tracker.confirmed_total if pipe else 0,
            "model_version": self.det.version,
        })


class CameraManager:

    def __init__(self, detector: Detector, store: Store, settings: Settings) -> None:
        self.det, self.store, self.s = detector, store, settings
        self.runners: dict[str, CameraRunner] = {}

    def sync(self) -> None:
        cams = {c.id: c for c in self.store.list_cameras() if c.enabled}
        for cid, r in list(self.runners.items()):
            c = cams.get(cid)
            if c is None or c.revision != r.revision or not r.is_alive():
                r.stop()
                r.join(timeout=STOP_TIMEOUT_S)
                del self.runners[cid]
        for cid, c in cams.items():
            if cid not in self.runners:
                r = CameraRunner(c, self.det, self.store, self.s)
                r.start()
                self.runners[cid] = r
        for c in self.store.list_cameras():
            if not c.enabled and c.state != "disabled":
                self.store.set_camera_status(c.id, "disabled", c.status or {})

    def stop_all(self) -> None:
        for r in self.runners.values():
            r.stop()
        for r in self.runners.values():
            r.join(timeout=STOP_TIMEOUT_S)
        self.runners.clear()
