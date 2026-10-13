from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from types import TracebackType
from typing import IO

import cv2
import numpy as np

from bagcounter.count.counter import CountEvent
from bagcounter.detect.openvino import Frame
from bagcounter.errors import StorageError
from bagcounter.geometry import PixelLine
from bagcounter.track.bytetrack import TrackBox

GREEN, RED, YELLOW, BLACK = (60, 200, 60), (40, 40, 230), (0, 210, 255), (0, 0, 0)
FFMPEG_CLOSE_TIMEOUT_S = 120.0


def draw(frame: Frame, line: PixelLine, tracks: list[TrackBox], count: int, flash: set[int]) -> Frame:
    out = frame.copy()
    scale = max(0.5, min(out.shape[:2]) / 720)
    for t in tracks:
        color = YELLOW if t.track_id in flash else GREEN
        cv2.rectangle(out, (int(t.x1), int(t.y1)), (int(t.x2), int(t.y2)), color, max(1, round(2 * scale)))
        cv2.putText(out, f"#{t.track_id} {t.score:.2f}", (int(t.x1), max(12, int(t.y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, color, 1, cv2.LINE_AA)
    a = (int(line.a[0]), int(line.a[1]))
    b = (int(line.b[0]), int(line.b[1]))
    cv2.line(out, a, b, RED, max(2, round(3 * scale)))
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = np.hypot(dx, dy) or 1.0
    arrow = 40 * scale
    if line.direction.value != "negative":
        cv2.arrowedLine(out, (int(mx), int(my)), (int(mx - dy / n * arrow), int(my + dx / n * arrow)), RED, 2,
                        tipLength=0.3)
    if line.direction.value != "positive":
        cv2.arrowedLine(out, (int(mx), int(my)), (int(mx + dy / n * arrow), int(my - dx / n * arrow)), RED, 2,
                        tipLength=0.3)
    label = f"Bags: {count}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(out, (8, 8), (20 + tw, 20 + th + 6), BLACK, -1)
    cv2.putText(out, label, (14, 14 + th), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2, cv2.LINE_AA)
    return out


class H264Writer:

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path = path
        w, h = width - width % 2, height - height % 2
        self._size = (w, h)
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", f"{fps:.3f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        self._log_path = path.with_suffix(".ffmpeg.log")
        try:
            self._log = self._log_path.open("wb")
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log)
        except OSError as e:
            raise StorageError("cannot start ffmpeg for the annotated video") from e
        if self._proc.stdin is None:
            raise StorageError("ffmpeg has no stdin pipe")
        self._stdin: IO[bytes] = self._proc.stdin

    def write(self, frame: Frame) -> None:
        w, h = self._size
        try:
            self._stdin.write(np.ascontiguousarray(frame[:h, :w]).tobytes())
        except BrokenPipeError as e:
            raise StorageError("ffmpeg stopped while writing the annotated video", context=self._tail()) from e

    def _tail(self) -> dict[str, object]:
        self._log.flush()
        return {"ffmpeg": self._log_path.read_bytes()[-500:].decode(errors="replace")}

    def close(self) -> None:
        try:
            if not self._stdin.closed:
                self._stdin.close()
            try:
                rc = self._proc.wait(timeout=FFMPEG_CLOSE_TIMEOUT_S)
            except subprocess.TimeoutExpired as e:
                self._proc.kill()
                self._proc.wait(timeout=10)
                raise StorageError("ffmpeg did not finish the annotated video in time") from e
            if rc != 0:
                raise StorageError("ffmpeg failed", context=self._tail())
        finally:
            self._log.close()
        self._log_path.unlink(missing_ok=True)

    def abort(self) -> None:
        try:
            if not self._stdin.closed:
                self._stdin.close()
        except BrokenPipeError:
            self._proc.kill()
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait(timeout=10)
        self._log.close()
        self.path.unlink(missing_ok=True)
        self._log_path.unlink(missing_ok=True)

    def __enter__(self) -> H264Writer:
        return self

    def __exit__(self, et: type[BaseException] | None, e: BaseException | None, tb: TracebackType | None) -> None:
        if et is None:
            self.close()
        else:
            self.abort()


def save_snapshot(frame: Frame, ev: CountEvent, out_dir: Path, max_side: int) -> str:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = ev.box
    px, py = (x2 - x1) * 0.15, (y2 - y1) * 0.15
    crop = frame[max(0, int(y1 - py)):min(h, int(y2 + py)), max(0, int(x1 - px)):min(w, int(x2 + px))]
    if crop.size == 0:
        crop = frame
    s = max_side / max(crop.shape[:2])
    if s < 1:
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)
    name = f"event_{ev.seq:05d}.jpg"
    if not cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise StorageError("cannot write snapshot", context={"file": name})
    return name


EVENT_FIELDS = ["seq", "time_s", "frame", "track_id", "direction", "confidence", "late", "x1", "y1", "x2", "y2",
                "snapshot"]


def event_row(ev: CountEvent, snapshot: str | None) -> dict[str, object]:
    return {"seq": ev.seq, "time_s": round(ev.t_s, 3), "frame": ev.frame_index, "track_id": ev.track_id,
            "direction": ev.direction, "confidence": round(ev.confidence, 3), "late": ev.late,
            "x1": round(ev.box[0], 1), "y1": round(ev.box[1], 1), "x2": round(ev.box[2], 1),
            "y2": round(ev.box[3], 1), "snapshot": snapshot}


def write_reports(out_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    with (out_dir / "events.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    (out_dir / "report.json").write_text(json.dumps({**summary, "events": rows}, indent=2))
