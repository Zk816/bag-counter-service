import numpy as np

from bagcounter.track.bytetrack import ByteTracker


def dets(*boxes):
    return np.array(boxes, dtype=np.float32).reshape(-1, 5)


def moving(i, x0=10.0, y0=10.0, vy=5.0, score=0.9):
    return (x0, y0 + vy * i, x0 + 40, y0 + 40 + vy * i, score)


def test_id_is_stable_for_a_moving_object():
    t = ByteTracker()
    ids = set()
    for i in range(20):
        out, _ = t.update(dets(moving(i)))
        ids |= {o.track_id for o in out}
    assert ids == {1}


def test_two_identical_objects_keep_separate_ids():
    t = ByteTracker()
    for i in range(15):
        out, _ = t.update(dets(moving(i, x0=10), moving(i, x0=200)))
    assert sorted(o.track_id for o in out) == [1, 2]


def test_low_score_boxes_keep_a_track_alive():
    t = ByteTracker(high_thresh=0.5, low_thresh=0.1)
    for i in range(5):
        t.update(dets(moving(i)))
    for i in range(5, 10):
        out, _ = t.update(dets(moving(i, score=0.2)))
    assert [o.track_id for o in out] == [1]


def test_short_gap_reuses_id_long_gap_removes():
    t = ByteTracker(buffer_frames=5)
    for i in range(5):
        t.update(dets(moving(i)))
    for _ in range(3):
        t.update(dets())
    out, _ = t.update(dets(moving(8)))
    assert [o.track_id for o in out] == [1]

    removed_all = []
    for _ in range(10):
        _, removed = t.update(dets())
        removed_all += removed
    assert removed_all == [1]


def test_single_frame_false_positive_gets_no_id():
    t = ByteTracker()
    t.update(dets(moving(0)))
    t.update(dets(moving(1), (400, 400, 440, 440, 0.9)))
    for i in range(2, 6):
        out, _ = t.update(dets(moving(i)))
    assert t.confirmed_total == 1
    assert [o.track_id for o in out] == [1]


def test_empty_frames_are_fine():
    t = ByteTracker()
    for _ in range(5):
        out, removed = t.update(dets())
    assert out == [] and removed == []
    assert t.flush() == []
