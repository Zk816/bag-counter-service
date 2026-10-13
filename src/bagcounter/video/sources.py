from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2

from bagcounter.detect.openvino import Frame
from bagcounter.errors import InvalidVideoError

MIN_FPS, MAX_FPS = 1.0, 240.0


@dataclass(frozen=True)
class Packet:
    image: Frame
    index: int
    t_s: float


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


class FileSource:
    def __init__(self, path: Path, *, max_seconds: float, max_pixels: int, stride: int = 1) -> None:
        self.path, self.stride = path, max(1, stride)
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        try:
            if not cap.isOpened():
                raise InvalidVideoError("file is not a readable video")
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            ok, first = cap.read()
        finally:
            cap.release()
        if not ok or first is None:
            raise InvalidVideoError("video has no decodable frame")
        if not MIN_FPS <= fps <= MAX_FPS:
            raise InvalidVideoError("video frame rate out of range", context={"fps": fps})
        if w * h > max_pixels or w < 32 or h < 32:
            raise InvalidVideoError("video resolution out of range", context={"width": w, "height": h})
        self.info = VideoInfo(w, h, fps, n)
        if self.info.duration_s > max_seconds:
            raise InvalidVideoError("video is too long", context={"seconds": round(self.info.duration_s, 1)})
        self.first_frame: Frame = first

    def __iter__(self) -> Iterator[Packet]:
        cap = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
        try:
            i = -1
            while True:
                ok = cap.grab()
                if not ok:
                    return
                i += 1
                if i % self.stride:
                    continue
                ok, img = cap.retrieve()
                if not ok:
                    return
                yield Packet(img, i, i / self.info.fps)
        finally:
            cap.release()


class StreamState(str, Enum):
    CONNECTING = "connecting"
    LIVE = "live"
    STALLED = "stalled"
    FROZEN = "frozen"
    STOPPED = "stopped"


@dataclass
class StreamStats:
    decoded: int = 0
    dropped: int = 0
    reconnects: int = 0
    last_frame_at: float = 0.0
    width: int = 0
    height: int = 0


def _fingerprint(img: Frame) -> bytes:
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA)
    return bytes(small)


class RtspSource:

    def __init__(self, url: str, *, open_timeout_s: float, read_timeout_s: float, backoff_min_s: float,
                 backoff_max_s: float, stall_after_s: float, frozen_after_s: float) -> None:
        self.url = url
        self._open_ms = int(open_timeout_s * 1000)
        self._read_ms = int(read_timeout_s * 1000)
        self._backoff = (backoff_min_s, backoff_max_s)
        self._stall_after, self._frozen_after = stall_after_s, frozen_after_s
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest: tuple[Frame, float] | None = None
        self._stop = threading.Event()
        self._fp: bytes = b""
        self._fp_since = 0.0
        self._connected = False
        self.stats = StreamStats()
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name=f"rtsp-{url[-24:]}", daemon=True)

    def start(self) -> RtspSource:
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=timeout_s)

    @property
    def state(self) -> StreamState:
        if self._stop.is_set():
            return StreamState.STOPPED
        now = time.monotonic()
        with self._lock:
            last, fp_since, connected = self.stats.last_frame_at, self._fp_since, self._connected
        if not last:
            return StreamState.CONNECTING if now - self.started_at < self._stall_after else StreamState.STALLED
        if now - last > self._stall_after:
            return StreamState.STALLED
        if not connected:
            return StreamState.CONNECTING
        if fp_since and now - fp_since > self._frozen_after:
            return StreamState.FROZEN
        return StreamState.LIVE

    def take(self, timeout_s: float) -> Packet | None:
        with self._cond:
            if self._latest is None:
                self._cond.wait(timeout=timeout_s)
            item, self._latest = self._latest, None
        if item is None:
            return None
        img, t = item
        return Packet(img, self.stats.decoded, t - self.started_at)

    def _run(self) -> None:
        delay = self._backoff[0]
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._open_ms,
                                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._read_ms])
            try:
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    delay = self._backoff[0]
                    self._read_loop(cap)
            finally:
                cap.release()
                with self._lock:
                    self._connected = False
            if self._stop.wait(delay):
                return
            delay = min(delay * 2, self._backoff[1])
            self.stats.reconnects += 1

    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        while not self._stop.is_set():
            ok, img = cap.read()
            if not ok or img is None:
                return
            now = time.monotonic()
            fp = _fingerprint(img)
            with self._cond:
                self._connected = True
                if fp != self._fp:
                    self._fp, self._fp_since = fp, 0.0
                elif not self._fp_since:
                    self._fp_since = now
                if self._latest is not None:
                    self.stats.dropped += 1
                self._latest = (img, now)
                self.stats.decoded += 1
                self.stats.last_frame_at = now
                self.stats.height, self.stats.width = img.shape[:2]
                self._cond.notify()
