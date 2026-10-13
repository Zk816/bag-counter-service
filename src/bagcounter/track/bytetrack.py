from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from bagcounter.track.kalman import F64, KalmanXYAH


class State(Enum):
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass(frozen=True)
class TrackBox:

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class _Track:
    mean: F64
    cov: F64
    score: float
    start_frame: int
    last_frame: int
    state: State = State.TRACKED
    confirmed: bool = False
    track_id: int = 0
    hits: int = 1

    @property
    def tlbr(self) -> F64:
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _xyah(box: F64) -> F64:
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w / max(h, 1e-6), h])


def _iou(a: F64, b: F64) -> F64:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    out: F64 = inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)
    return out


def _match(cost: F64, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= thresh]
    mr = {r for r, _ in pairs}
    mc = {c for _, c in pairs}
    return pairs, [r for r in range(cost.shape[0]) if r not in mr], [c for c in range(cost.shape[1]) if c not in mc]


@dataclass
class ByteTracker:
    high_thresh: float = 0.25
    low_thresh: float = 0.1
    new_thresh: float = 0.25
    buffer_frames: int = 30
    match_thresh: float = 0.8
    frame: int = 0
    _tracked: list[_Track] = field(default_factory=list)
    _lost: list[_Track] = field(default_factory=list)
    _next_id: int = 1
    _kf: KalmanXYAH = field(default_factory=KalmanXYAH)
    confirmed_total: int = 0

    def _confirm(self, t: _Track) -> None:
        if not t.confirmed:
            t.confirmed, t.track_id = True, self._next_id
            self._next_id += 1
            self.confirmed_total += 1

    def _update(self, t: _Track, box: F64, score: float) -> None:
        t.mean, t.cov = self._kf.update(t.mean, t.cov, _xyah(box))
        t.score, t.last_frame, t.state = score, self.frame, State.TRACKED
        t.hits += 1

    def update(self, dets: npt.NDArray[np.float32]) -> tuple[list[TrackBox], list[int]]:
        self.frame += 1
        boxes = dets[:, :4].astype(np.float64) if len(dets) else np.zeros((0, 4))
        scores = dets[:, 4].astype(np.float64) if len(dets) else np.zeros(0)
        hi = np.flatnonzero(scores >= self.high_thresh)
        lo = np.flatnonzero((scores >= self.low_thresh) & (scores < self.high_thresh))

        unconfirmed = [t for t in self._tracked if not t.confirmed]
        pool = [t for t in self._tracked if t.confirmed] + self._lost
        for t in pool:
            if t.state is not State.TRACKED:
                t.mean[7] = 0.0
            t.mean, t.cov = self._kf.predict(t.mean, t.cov)
        for t in unconfirmed:
            t.mean, t.cov = self._kf.predict(t.mean, t.cov)

        pool_boxes = np.array([t.tlbr for t in pool]).reshape(-1, 4)
        cost = 1 - _iou(pool_boxes, boxes[hi]) * scores[hi][None, :]
        pairs, rest_tracks, rest_hi = _match(cost, self.match_thresh)
        for ti, di in pairs:
            self._update(pool[ti], boxes[hi[di]], float(scores[hi[di]]))

        second = [pool[i] for i in rest_tracks if pool[i].state is State.TRACKED]
        cost = 1 - _iou(np.array([t.tlbr for t in second]).reshape(-1, 4), boxes[lo])
        pairs, rest_second, _ = _match(cost, 0.5)
        for ti, di in pairs:
            self._update(second[ti], boxes[lo[di]], float(scores[lo[di]]))
        for i in rest_second:
            second[i].state = State.LOST

        hi_left = hi[rest_hi]
        unconf_boxes = np.array([t.tlbr for t in unconfirmed]).reshape(-1, 4)
        cost = 1 - _iou(unconf_boxes, boxes[hi_left]) * scores[hi_left][None, :]
        pairs, rest_unconf, rest_new = _match(cost, 0.7)
        for ti, di in pairs:
            self._update(unconfirmed[ti], boxes[hi_left[di]], float(scores[hi_left[di]]))
            self._confirm(unconfirmed[ti])
        for i in rest_unconf:
            unconfirmed[i].state = State.REMOVED

        for di in hi_left[rest_new]:
            if scores[di] < self.new_thresh:
                continue
            mean, cov = self._kf.initiate(_xyah(boxes[di]))
            t = _Track(mean, cov, float(scores[di]), self.frame, self.frame)
            if self.frame == 1:
                self._confirm(t)
            self._tracked.append(t)

        everyone = self._tracked + self._lost
        removed_ids: list[int] = []
        for t in everyone:
            if t.state is State.LOST and self.frame - t.last_frame > self.buffer_frames:
                t.state = State.REMOVED
            if t.state is State.REMOVED and t.confirmed:
                removed_ids.append(t.track_id)
        self._tracked = [t for t in everyone if t.state is State.TRACKED]
        self._lost = [t for t in everyone if t.state is State.LOST]
        self._drop_duplicates()

        out = []
        for t in self._tracked:
            if t.confirmed and t.last_frame == self.frame:
                x1, y1, x2, y2 = (float(v) for v in t.tlbr)
                out.append(TrackBox(t.track_id, x1, y1, x2, y2, t.score))
        return out, removed_ids

    def _drop_duplicates(self) -> None:
        if not self._tracked or not self._lost:
            return
        iou = _iou(np.array([t.tlbr for t in self._tracked]), np.array([t.tlbr for t in self._lost]))
        drop_t, drop_l = set(), set()
        for i, j in zip(*np.nonzero(iou > 0.85), strict=True):
            a, b = self._tracked[i], self._lost[j]
            if a.last_frame - a.start_frame > b.last_frame - b.start_frame:
                drop_l.add(j)
            else:
                drop_t.add(i)
        self._tracked = [t for i, t in enumerate(self._tracked) if i not in drop_t]
        self._lost = [t for j, t in enumerate(self._lost) if j not in drop_l]

    def flush(self) -> list[int]:
        ids = [t.track_id for t in self._tracked + self._lost if t.confirmed]
        self._tracked, self._lost = [], []
        return ids
