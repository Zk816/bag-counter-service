from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from bagcounter.config import Settings
from bagcounter.count.counter import CountEvent
from bagcounter.detect.openvino import Detector, Frame
from bagcounter.errors import JobCancelledError, JobTimeoutError
from bagcounter.geometry import Line
from bagcounter.metrics import BAGS_COUNTED, DETECTION_SCORE, FRAMES_PROCESSED, TRACKS_CONFIRMED
from bagcounter.pipeline import CountingPipeline
from bagcounter.sinks import H264Writer, draw, event_row, save_snapshot, write_reports
from bagcounter.video.sources import FileSource

PROGRESS_EVERY_S = 1.0
FLASH_FRAMES = 8


def _ms(xs: list[float], q: float) -> float:
    return round(1000 * float(np.percentile(xs, q)), 2) if xs else 0.0


def run_video(src: FileSource, line: Line, settings: Settings, detector: Detector, out_dir: Path, *,
              annotate: bool = True, progress: Callable[[float, int], None] | None = None,
              cancelled: Callable[[], bool] | None = None) -> dict[str, object]:
    info = src.info
    out_dir.mkdir(parents=True, exist_ok=True)
    snaps = out_dir / "snapshots"
    snaps.mkdir(exist_ok=True)
    pipe = CountingPipeline(detector, line, info.width, info.height, info.fps / src.stride, settings, stream="job")
    rows: list[dict[str, object]] = []
    flash: dict[int, int] = {}
    started = time.monotonic()
    last_progress = 0.0
    last_t = 0.0
    last_frame: Frame | None = None
    writer = H264Writer(out_dir / "annotated.mp4", info.width, info.height, info.fps / src.stride) if annotate else None

    def record(events: list[CountEvent], frame: Frame) -> None:
        for ev in events:
            rows.append(event_row(ev, f"snapshots/{save_snapshot(frame, ev, snaps, settings.snapshot_max_side)}"))
            flash[ev.track_id] = FLASH_FRAMES
            BAGS_COUNTED.labels(stream="job", late=str(ev.late).lower()).inc()

    try:
        for pkt in src:
            now = time.monotonic()
            if now - started > settings.job_timeout_s:
                raise JobTimeoutError("video job exceeded its time budget",
                                      context={"timeout_s": settings.job_timeout_s})
            res = pipe.process(pkt.image, pkt.t_s)
            FRAMES_PROCESSED.labels(stream="job").inc()
            for t in res.tracks:
                DETECTION_SCORE.labels(stream="job").observe(t.score)
            record(res.events, pkt.image)
            if writer:
                writer.write(draw(pkt.image, pipe.pixel_line, res.tracks, pipe.counter.count, set(flash)))
            flash = {k: v - 1 for k, v in flash.items() if v > 1}
            last_t, last_frame = pkt.t_s, pkt.image
            if now - last_progress >= PROGRESS_EVERY_S:
                last_progress = now
                if cancelled and cancelled():
                    raise JobCancelledError("job was cancelled")
                if progress:
                    progress(min(1.0, pkt.index / max(1, info.frames)), pipe.counter.count)
        if last_frame is not None:
            record(pipe.finish(last_t), last_frame)
        if writer:
            writer.close()
    except BaseException:
        if writer:
            writer.abort()
        raise

    TRACKS_CONFIRMED.labels(stream="job").inc(pipe.tracker.confirmed_total)
    elapsed = time.monotonic() - started
    count = pipe.counter.count
    summary: dict[str, object] = {
        "count": count,
        "late_events": sum(1 for r in rows if r["late"]),
        "line": line.to_dict(),
        "model_version": detector.version,
        "video": {"width": info.width, "height": info.height, "fps": round(info.fps, 3), "frames": info.frames,
                  "duration_s": round(info.duration_s, 2)},
        "frames_processed": pipe.frames,
        "processing_s": round(elapsed, 2),
        "processing_fps": round(pipe.frames / elapsed, 1) if elapsed else 0.0,
        "tracks_confirmed": pipe.tracker.confirmed_total,
        "fragmentation": round(pipe.tracker.confirmed_total / count, 2) if count else None,
        "latency_ms": {"detect_p50": _ms(pipe.times.detect, 50), "detect_p95": _ms(pipe.times.detect, 95),
                       "track_count_p50": _ms(pipe.times.track_count, 50)},
        "outputs": {"annotated_video": "annotated.mp4" if annotate else None, "events_csv": "events.csv",
                    "report_json": "report.json"},
    }
    write_reports(out_dir, rows, summary)
    if progress:
        progress(1.0, count)
    return summary
