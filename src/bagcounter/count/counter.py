from __future__ import annotations

from dataclasses import dataclass, field

from bagcounter.geometry import PixelLine
from bagcounter.track.bytetrack import TrackBox


@dataclass(frozen=True)
class CountEvent:
    seq: int
    track_id: int
    frame_index: int
    t_s: float
    direction: int
    box: tuple[float, float, float, float]
    confidence: float
    late: bool


@dataclass
class _TrackState:
    hits: int = 0
    first: tuple[float, float] = (0.0, 0.0)
    last: tuple[float, float] = (0.0, 0.0)
    last_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_frame: int = 0
    score_sum: float = 0.0
    counted: bool = False


@dataclass
class LineCounter:
    line: PixelLine
    min_hits: int = 3
    margin: float = 0.05
    stale_after_frames: int = 60
    count: int = 0
    tracks_seen: int = 0
    _tracks: dict[int, _TrackState] = field(default_factory=dict)

    def update(self, tracks: list[TrackBox], removed: list[int], frame_index: int, t_s: float) -> list[CountEvent]:
        events: list[CountEvent] = []
        for tb in tracks:
            st = self._tracks.get(tb.track_id)
            c = tb.center
            if st is None:
                st = self._tracks[tb.track_id] = _TrackState(first=c, last=c)
                self.tracks_seen += 1
            prev = st.last
            st.hits += 1
            st.score_sum += tb.score
            st.last, st.last_box, st.last_frame = c, (tb.x1, tb.y1, tb.x2, tb.y2), frame_index
            if st.counted or st.hits < self.min_hits:
                continue
            sign = self.line.crossing(prev, c, self.margin)
            if self.line.counts(sign):
                events.append(self._emit(tb.track_id, st, sign, frame_index, t_s, late=False))

        stale = [tid for tid, st in self._tracks.items() if frame_index - st.last_frame > self.stale_after_frames]
        for tid in {*removed, *stale}:
            ev = self._finalise(tid, frame_index, t_s)
            if ev:
                events.append(ev)
        return events

    def finish(self, frame_index: int, t_s: float) -> list[CountEvent]:
        return [ev for tid in list(self._tracks) if (ev := self._finalise(tid, frame_index, t_s))]

    def _finalise(self, tid: int, frame_index: int, t_s: float) -> CountEvent | None:
        st = self._tracks.pop(tid, None)
        if st is None or st.counted or st.hits < self.min_hits:
            return None
        sign = self.line.crossing(st.first, st.last, self.margin)
        if not self.line.counts(sign):
            return None
        return self._emit(tid, st, sign, frame_index, t_s, late=True)

    def _emit(self, tid: int, st: _TrackState, sign: int, frame_index: int, t_s: float, *, late: bool) -> CountEvent:
        st.counted = True
        self.count += 1
        return CountEvent(self.count, tid, frame_index, t_s, sign, st.last_box, st.score_sum / st.hits, late)

    @property
    def open_tracks(self) -> int:
        return len(self._tracks)
