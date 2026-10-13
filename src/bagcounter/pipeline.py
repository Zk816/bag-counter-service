from __future__ import annotations

import time
from dataclasses import dataclass, field

from bagcounter.config import Settings
from bagcounter.count.counter import CountEvent, LineCounter
from bagcounter.detect.openvino import Detector, Frame
from bagcounter.geometry import Line
from bagcounter.metrics import STAGE_SECONDS
from bagcounter.track.bytetrack import ByteTracker, TrackBox


@dataclass
class FrameResult:
    tracks: list[TrackBox]
    events: list[CountEvent]
    detections: int


@dataclass
class StageTimes:
    detect: list[float] = field(default_factory=list)
    track_count: list[float] = field(default_factory=list)


class CountingPipeline:
    def __init__(self, detector: Detector, line: Line, width: int, height: int, fps: float,
                 settings: Settings, *, stream: str) -> None:
        self.session = detector.session()
        self.line = line
        self.pixel_line = line.in_pixels(width, height)
        buffer_frames = max(1, round(settings.track_buffer_s * fps))
        self.tracker = ByteTracker(settings.track_high_thresh, settings.track_low_thresh, settings.new_track_thresh,
                                   buffer_frames, settings.match_thresh)
        self.counter = LineCounter(self.pixel_line, settings.min_hits, settings.line_margin,
                                   stale_after_frames=2 * buffer_frames + 2)
        self.frames = 0
        self.times = StageTimes()
        self._stream = stream

    def process(self, image: Frame, t_s: float) -> FrameResult:
        a = time.perf_counter()
        dets = self.session.detect(image)
        b = time.perf_counter()
        tracks, removed = self.tracker.update(dets)
        events = self.counter.update(tracks, removed, self.frames, t_s)
        c = time.perf_counter()
        self.frames += 1
        self.times.detect.append(b - a)
        self.times.track_count.append(c - b)
        if len(self.times.detect) > 10_000:
            del self.times.detect[:5_000], self.times.track_count[:5_000]
        STAGE_SECONDS.labels(stage="detect", stream=self._stream).observe(b - a)
        STAGE_SECONDS.labels(stage="track_count", stream=self._stream).observe(c - b)
        return FrameResult(tracks, events, len(dets))

    def finish(self, t_s: float) -> list[CountEvent]:
        self.tracker.flush()
        return self.counter.finish(self.frames, t_s)
