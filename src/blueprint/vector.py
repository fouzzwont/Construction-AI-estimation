"""Native PDF vector-geometry extraction.

Real construction drawings are usually vector CAD exports.  Reading the
geometry straight out of the PDF (instead of inferring it from a raster)
gives exact member dimensions, which is what a quantity take-off needs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)

PT_PER_MM = 72.0 / 25.4  # 2.8346 PDF points per millimetre


@dataclass
class Segment:
    """A straight line in PDF point space."""

    x1: float
    y1: float
    x2: float
    y2: float
    width: float = 0.0
    layer: str = ""

    @property
    def length_pt(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1)) % 180.0

    @property
    def is_horizontal(self) -> bool:
        a = self.angle_deg
        return a < 8.0 or a > 172.0

    @property
    def is_vertical(self) -> bool:
        return 82.0 < self.angle_deg < 98.0

    @property
    def is_axis_aligned(self) -> bool:
        return self.is_horizontal or self.is_vertical


@dataclass
class Rect:
    """An axis-aligned rectangle in PDF point space."""

    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return abs(self.w) * abs(self.h)

    @property
    def perimeter(self) -> float:
        return 2.0 * (abs(self.w) + abs(self.h))

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def overlaps(self, other: "Rect", tol: float = 1.0) -> bool:
        return not (
            self.x2 + tol < other.x
            or other.x2 + tol < self.x
            or self.y2 + tol < other.y
            or other.y2 + tol < self.y
        )

    def contained_in(self, other: "Rect", tol: float = 1.5) -> bool:
        return (
            self.x >= other.x - tol
            and self.y >= other.y - tol
            and self.x2 <= other.x2 + tol
            and self.y2 <= other.y2 + tol
        )


@dataclass
class Circle:
    cx: float
    cy: float
    r: float


@dataclass
class PageGeometry:
    """All vector content of one page, in PDF points."""

    segments: list[Segment] = field(default_factory=list)
    rects: list[Rect] = field(default_factory=list)
    circles: list[Circle] = field(default_factory=list)
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def width_pt(self) -> float:
        return self.bounds[2] - self.bounds[0]

    @property
    def height_pt(self) -> float:
        return self.bounds[3] - self.bounds[1]

    def horizontal_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.is_horizontal]

    def vertical_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.is_vertical]

    def is_empty(self) -> bool:
        return not (self.segments or self.rects or self.circles)


def _flatten(items: Iterable[Any]) -> tuple[list[Segment], list[Rect], list[Circle]]:
    segments: list[Segment] = []
    rects: list[Rect] = []
    circles: list[Circle] = []

    for item in items:
        op = item[0]
        try:
            if op == "l":
                p1, p2 = item[1], item[2]
                segments.append(
                    Segment(float(p1.x), float(p1.y), float(p2.x), float(p2.y))
                )
            elif op == "re":
                r = item[1]
                rects.append(Rect(float(r.x0), float(r.y0), float(r.width), float(r.height)))
            elif op == "c":
                # cubic bezier - treat as circle when the control points form one
                p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
                xs = [float(p.x) for p in (p1, p2, p3, p4)]
                ys = [float(p.y) for p in (p1, p2, p3, p4)]
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                if w > 0 and h > 0 and 0.85 <= w / h <= 1.18:
                    circles.append(Circle((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, w / 2.0))
                elif w > 1 and h > 1:
                    segments.append(Segment(xs[0], ys[0], xs[3], ys[3]))
            elif op == "qu":
                q = item[1]
                xs = [float(q.ul.x), float(q.ur.x), float(q.lr.x), float(q.ll.x)]
                ys = [float(q.ul.y), float(q.ur.y), float(q.lr.y), float(q.ll.y)]
                x0, y0 = min(xs), min(ys)
                rects.append(Rect(x0, y0, max(xs) - x0, max(ys) - y0))
        except Exception:
            continue

    return segments, rects, circles


def extract_geometry(drawings: list[dict[str, Any]], min_length_pt: float = 2.0) -> PageGeometry:
    """Convert PyMuPDF ``get_drawings()`` output into a :class:`PageGeometry`."""
    segments: list[Segment] = []
    rects: list[Rect] = []
    circles: list[Circle] = []

    for path in drawings or []:
        items = path.get("items") or []
        s, r, c = _flatten(items)
        width = float(path.get("width") or 0.0)
        layer = str(path.get("layer") or "")
        for seg in s:
            seg.width = width
            seg.layer = layer
            if seg.length_pt >= min_length_pt:
                segments.append(seg)
        rects.extend(r)
        circles.extend(c)

    xs: list[float] = []
    ys: list[float] = []
    for seg in segments:
        xs.extend((seg.x1, seg.x2))
        ys.extend((seg.y1, seg.y2))
    for rect in rects:
        xs.extend((rect.x, rect.x2))
        ys.extend((rect.y, rect.y2))
    for circ in circles:
        xs.extend((circ.cx - circ.r, circ.cx + circ.r))
        ys.extend((circ.cy - circ.r, circ.cy + circ.r))

    bounds = (min(xs), min(ys), max(xs), max(ys)) if xs else (0.0, 0.0, 0.0, 0.0)
    return PageGeometry(segments=segments, rects=rects, circles=circles, bounds=bounds)


def pt_to_paper_mm(value: float) -> float:
    """Convert PDF points to millimetres *as drawn* (i.e. on paper)."""
    return value / PT_PER_MM


def scale_denominator_of(denominator: float) -> float:
    """Guard against nonsensical scale values."""
    d = float(denominator)
    if d <= 0:
        return 100.0
    if d < 5:
        return 100.0
    return min(d, 5000.0)
