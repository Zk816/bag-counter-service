import pytest

from bagcounter.errors import InvalidLineError
from bagcounter.geometry import Direction, Line


def horizontal(direction: str = "positive"):
    return Line.parse("0.1,0.5,0.9,0.5", direction).in_pixels(100, 100)


def test_downward_crossing_is_positive():
    assert horizontal().crossing((50, 40), (50, 60), 0.0) == 1


def test_upward_crossing_is_negative():
    assert horizontal().crossing((50, 60), (50, 40), 0.0) == -1


def test_no_crossing_on_same_side():
    assert horizontal().crossing((50, 10), (50, 40), 0.0) == 0


def test_crossing_outside_segment_is_ignored():
    assert horizontal().crossing((95, 40), (95, 60), 0.0) == 0
    assert horizontal().crossing((95, 40), (95, 60), 0.1) == 1


def test_landing_exactly_on_line_counts_once():
    ln = horizontal()
    first = ln.crossing((50, 40), (50, 50), 0.0)
    second = ln.crossing((50, 50), (50, 60), 0.0)
    assert (first, second) == (1, 0)


def test_direction_filter():
    assert horizontal("positive").counts(1) and not horizontal("positive").counts(-1)
    assert horizontal("negative").counts(-1) and not horizontal("negative").counts(1)
    assert horizontal("both").counts(1) and horizontal("both").counts(-1)
    assert not horizontal("both").counts(0)


def test_vertical_line_left_to_right():
    ln = Line(0.5, 0.1, 0.5, 0.9, Direction.NEGATIVE).in_pixels(200, 100)
    assert ln.crossing((80, 50), (120, 50), 0.0) == -1


@pytest.mark.parametrize("spec", ["0.1,0.2,0.3", "a,b,c,d", "0.1,0.2,1.5,0.2", "0.5,0.5,0.505,0.5"])
def test_invalid_lines_rejected(spec):
    with pytest.raises(InvalidLineError):
        Line.parse(spec)


def test_unknown_direction_rejected():
    with pytest.raises(InvalidLineError):
        Line.parse("0.1,0.5,0.9,0.5", "sideways")
