"""CPU detection + tracking + counting, with OpenVINO directly (no Ultralytics in the hot path).

  frame (BGR, any size) -> letterbox 640 -> OpenVINO (FP32 or INT8) -> decode [1, 5, 8400]
  -> confidence filter + NMS -> ByteTrack -> line-crossing counter

Doing pre/post-processing ourselves is what an edge deployment needs: the runtime is a 50 MB
OpenVINO library, not PyTorch + Ultralytics, and every millisecond is accounted for.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import openvino as ov
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from common import CONF, IMGSZ, IOU, LINE, MIN_HITS, THREADS

TRACKER_ARGS = SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                               track_buffer=30, match_thresh=0.8, fuse_score=True)
_core = ov.Core()


def compile_model(model_dir: Path, *, threads: int = THREADS, hint: str = "LATENCY") -> ov.CompiledModel:
    """LATENCY = one request as fast as possible; THROUGHPUT = several parallel streams (multi-camera)."""
    xml = next(model_dir.glob("*.xml"))
    return _core.compile_model(_core.read_model(xml), "CPU", {"INFERENCE_NUM_THREADS": threads,
                                                              "PERFORMANCE_HINT": hint})


def letterbox(frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio, pad to IMGSZ x IMGSZ with grey; return NCHW float tensor, scale, padding."""
    h, w = frame.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), r, (left, top)


def decode(out: np.ndarray, r: float, pad: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    """[1, 5, 8400] (cx, cy, w, h, score) -> [N, 6] (x1, y1, x2, y2, score, cls) in frame pixels, after NMS."""
    p = out[0].T
    p = p[p[:, 4] >= CONF]
    if not len(p):
        return np.zeros((0, 6), np.float32)
    cx, cy, w, h = (p[:, 0] - pad[0]) / r, (p[:, 1] - pad[1]) / r, p[:, 2] / r, p[:, 3] / r
    boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), p[:, 4].tolist(), CONF, IOU)
    keep = np.array(keep).reshape(-1)
    b = boxes[keep]
    x1 = np.clip(b[:, 0], 0, shape[1]); y1 = np.clip(b[:, 1], 0, shape[0])
    x2 = np.clip(b[:, 0] + b[:, 2], 0, shape[1]); y2 = np.clip(b[:, 1] + b[:, 3], 0, shape[0])
    return np.stack([x1, y1, x2, y2, p[keep, 4], np.zeros(len(keep))], axis=1).astype(np.float32)


@dataclass
class Counter:
    """ByteTrack + the counting rule of infer_video.py: count a track when its centre crosses the line."""

    frame_h: int
    fps: float = 30.0
    tracker: BYTETracker = field(init=False)
    hits: dict = field(default_factory=lambda: defaultdict(int))
    min_cy: dict = field(default_factory=lambda: defaultdict(lambda: float("inf")))
    max_cy: dict = field(default_factory=lambda: defaultdict(lambda: float("-inf")))
    last_cy: dict = field(default_factory=dict)
    counted: set = field(default_factory=set)

    def __post_init__(self) -> None:
        self.tracker = BYTETracker(TRACKER_ARGS, frame_rate=round(self.fps))
        self.line_y = int(self.frame_h * LINE)

    def update(self, dets: np.ndarray, shape: tuple[int, int]) -> int:
        tracks = self.tracker.update(Boxes(dets, shape)) if len(dets) else self.tracker.update(Boxes(np.zeros((0, 6)), shape))
        for t in tracks:
            tid, cy = int(t[4]), (t[1] + t[3]) / 2
            self.hits[tid] += 1
            self.min_cy[tid] = min(self.min_cy[tid], cy)
            self.max_cy[tid] = max(self.max_cy[tid], cy)
            if tid in self.last_cy and self.hits[tid] >= MIN_HITS and tid not in self.counted:
                prev = self.last_cy[tid]
                if (prev < self.line_y <= cy) or (prev > self.line_y >= cy):
                    self.counted.add(tid)
            self.last_cy[tid] = cy
        return len(self.counted)

    def finish(self) -> int:
        """End-of-video pass of infer_video.py: tracks that spanned the line but were missed at the crossing."""
        for tid, n in self.hits.items():
            if tid not in self.counted and n >= MIN_HITS and self.min_cy[tid] < self.line_y < self.max_cy[tid]:
                self.counted.add(tid)
        return len(self.counted)


def count_video(model_dir: Path, video: Path) -> dict:
    """Run the full pipeline over a file, frame by frame; timing per stage."""
    compiled = compile_model(model_dir)
    req = compiled.create_infer_request()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    counter = None
    t_pre, t_inf, t_post, frames = [], [], [], 0
    started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if counter is None:
            counter = Counter(frame.shape[0], fps)
        a = time.perf_counter()
        x, r, pad = letterbox(frame)
        b = time.perf_counter()
        out = req.infer({0: x})[compiled.output(0)]
        c = time.perf_counter()
        counter.update(decode(out, r, pad, frame.shape[:2]), frame.shape[:2])
        d = time.perf_counter()
        t_pre.append(b - a); t_inf.append(c - b); t_post.append(d - c)
        frames += 1
    cap.release()
    total = time.perf_counter() - started
    return {"video": video.name, "frames": frames, "count": counter.finish() if counter else 0,
            "fps": frames / total, "ms": {"pre": 1000 * float(np.mean(t_pre)), "infer": 1000 * float(np.mean(t_inf)),
                                          "post_track": 1000 * float(np.mean(t_post))}}
