from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from bagcounter.errors import InvalidLineError

Point = tuple[float, float]


class Direction(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BOTH = "both"


@dataclass(frozen=True)
class Line:
    ax: float
    ay: float
    bx: float
    by: float
    direction: Direction = Direction.POSITIVE

    def __post_init__(self) -> None:
        for v in (self.ax, self.ay, self.bx, self.by):
            if not 0.0 <= v <= 1.0:
                raise InvalidLineError("line coordinates must be normalised to 0..1", context={"value": v})
        if (self.bx - self.ax) ** 2 + (self.by - self.ay) ** 2 < 0.02**2:
            raise InvalidLineError("line is too short (under 2% of the frame)")

    @classmethod
    def parse(cls, spec: str, direction: str = "positive") -> Line:
        try:
            ax, ay, bx, by = (float(v) for v in spec.split(","))
            d = Direction(direction)
        except ValueError as e:
            raise InvalidLineError(f"cannot parse line {spec!r} / direction {direction!r}") from e
        return cls(ax, ay, bx, by, d)

    def to_dict(self) -> dict[str, object]:
        return {"ax": self.ax, "ay": self.ay, "bx": self.bx, "by": self.by, "direction": self.direction.value}

    def in_pixels(self, width: int, height: int) -> PixelLine:
        return PixelLine((self.ax * width, self.ay * height), (self.bx * width, self.by * height), self.direction)


@dataclass(frozen=True)
class PixelLine:
    a: Point
    b: Point
    direction: Direction

    def side(self, p: Point) -> float:
        (ax, ay), (bx, by) = self.a, self.b
        return (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)

    def param(self, p: Point) -> float:
        (ax, ay), (bx, by) = self.a, self.b
        dx, dy = bx - ax, by - ay
        return ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)

    def crossing(self, prev: Point, cur: Point, margin: float) -> int:
        s0, s1 = self.side(prev), self.side(cur)
        if (s0 >= 0) == (s1 >= 0):
            return 0
        t = s0 / (s0 - s1)
        hit = (prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1]))
        u = self.param(hit)
        if not -margin <= u <= 1 + margin:
            return 0
        return 1 if s1 >= 0 else -1

    def counts(self, sign: int) -> bool:
        if sign == 0:
            return False
        if self.direction is Direction.BOTH:
            return True
        return (sign > 0) == (self.direction is Direction.POSITIVE)


def line_from_dict(d: dict[str, object]) -> Line:
    try:
        spec = ",".join(str(float(d[k])) for k in ("ax", "ay", "bx", "by"))  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as e:
        raise InvalidLineError("line needs numeric ax, ay, bx, by") from e
    return Line.parse(spec, str(d.get("direction", "positive")))
