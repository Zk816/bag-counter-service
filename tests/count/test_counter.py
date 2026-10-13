from bagcounter.count.counter import LineCounter
from bagcounter.geometry import Line
from bagcounter.track.bytetrack import TrackBox


def counter(direction="positive", min_hits=3):
    line = Line.parse("0.0,0.5,1.0,0.5", direction).in_pixels(100, 100)
    return LineCounter(line, min_hits=min_hits, margin=0.0, stale_after_frames=10)


def box(tid, cy, cx=50.0):
    return TrackBox(tid, cx - 5, cy - 5, cx + 5, cy + 5, 0.9)


def run(c, frames):
    events = []
    for i, dets in enumerate(frames):
        events += c.update([box(t, y) for t, y in dets], [], i, i / 30)
    return events


def test_counts_one_bag_crossing_once():
    c = counter()
    ev = run(c, [[(1, y)] for y in (30, 40, 45, 55, 60, 70)])
    assert c.count == 1 and len(ev) == 1 and not ev[0].late
    assert ev[0].direction == 1 and ev[0].track_id == 1


def test_jitter_at_the_line_counts_once():
    c = counter(direction="both")
    run(c, [[(1, y)] for y in (40, 45, 48, 52, 49, 53, 47, 55, 60)])
    assert c.count == 1


def test_wrong_direction_not_counted():
    c = counter("positive")
    run(c, [[(1, y)] for y in (70, 60, 55, 45, 40, 30)])
    assert c.count == 0


def test_min_hits_filters_short_tracks():
    c = counter(min_hits=3)
    run(c, [[(1, 45)], [(1, 55)]])
    c.finish(2, 0.1)
    assert c.count == 0


def test_dropout_at_the_line_counted_when_track_reappears():
    c = counter()
    events = run(c, [[(1, 30)], [(1, 35)], [(1, 40)], [], [], [(1, 62)], [(1, 70)]])
    assert c.count == 1 and not events[0].late


def test_track_that_spanned_line_is_finalised_late():
    c = counter()
    run(c, [[(3, 30)], [(3, 35)], [(3, 40)]])
    c._tracks[3].last = (50.0, 60.0)
    ev = c.update([], [3], 5, 5 / 30)
    assert c.count == 1 and ev[0].late


def test_stale_tracks_are_finalised_without_tracker_removal():
    c = counter()
    run(c, [[(4, 30)], [(4, 35)], [(4, 40)]])
    c._tracks[4].last = (50.0, 60.0)
    events = []
    for i in range(3, 20):
        events += c.update([], [], i, i / 30)
    assert c.count == 1 and events[0].late
    assert c.open_tracks == 0


def test_many_bags_counted_independently():
    c = counter()
    frames = [[(t, 20 + 5 * (i - t)) for t in range(1, 6) if 0 <= i - t < 15] for i in range(25)]
    run(c, frames)
    assert c.count == 5
