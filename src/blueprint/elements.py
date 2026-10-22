"""Classify PDF vector geometry into structural elements.

The rules below encode how a plan drawing is normally drawn:

* **Zones / slabs** - large closed rectangles (rooms, bays).
* **Walls** - thin closed rectangles, or pairs of parallel lines (double-line
  wall convention), usually bounding a zone.
* **Columns** - small squares/rectangles, typically sitting on the grid.
* **Beams** - long single straight members that do not bound any zone.

Every threshold is expressed in *real world millimetres*, so the same rules
work at 1:50 and at 1:500.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable

from .models import Element
from .vector import PT_PER_MM, PageGeometry, Rect, Segment

logger = logging.getLogger(__name__)


def pt_to_real(value_pt: float, denominator: float) -> float:
    """PDF points -> millimetres on paper -> millimetres in the real world."""
    return (value_pt / PT_PER_MM) * denominator


def _words_in_rect(words: Iterable[dict[str, Any]], rect: Rect) -> int:
    n = 0
    for w in words:
        cx = (float(w[0]) + float(w[2])) / 2.0
        cy = (float(w[1]) + float(w[3])) / 2.0
        if rect.x <= cx <= rect.x2 and rect.y <= cy <= rect.y2:
            n += 1
    return n


def find_annotation_panels(
    words: Iterable[dict[str, Any]], min_words: int = 8, cell: float = 70.0
) -> list[Rect]:
    """Locate annotation regions (title block, schedules, legends, notes).

    These carry dense text and their frame lines are *not* structure, so
    everything inside them is excluded from the take-off.
    """
    pts: list[tuple[float, float]] = []
    for w in words:
        try:
            pts.append(((float(w[0]) + float(w[2])) / 2.0, (float(w[1]) + float(w[3])) / 2.0))
        except Exception:
            continue
    if len(pts) < min_words:
        return []

    # bucket the words into a coarse grid, then flood fill neighbouring cells
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, (x, y) in enumerate(pts):
        key = (int(x // cell), int(y // cell))
        buckets.setdefault(key, []).append(idx)

    seen: set[tuple[int, int]] = set()
    panels: list[Rect] = []
    for key in list(buckets):
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        cluster: list[int] = []
        while stack:
            cx, cy = stack.pop()
            cluster.extend(buckets.get((cx, cy), []))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nxt = (cx + dx, cy + dy)
                    if nxt in buckets and nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
        if len(cluster) < min_words:
            continue
        xs = [pts[i][0] for i in cluster]
        ys = [pts[i][1] for i in cluster]
        pad = cell * 0.5
        panels.append(Rect(min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad))
    return panels


def _point_in_rect(x: float, y: float, rect: Rect, tol: float = 0.0) -> bool:
    return (rect.x - tol <= x <= rect.x2 + tol) and (rect.y - tol <= y <= rect.y2 + tol)


def _overlap_len(a: Segment, b: Segment, horizontal: bool) -> float:
    """Length of the projection two parallel segments share."""
    if horizontal:
        lo = max(min(a.x1, a.x2), min(b.x1, b.x2))
        hi = min(max(a.x1, a.x2), max(b.x1, b.x2))
    else:
        lo = max(min(a.y1, a.y2), min(b.y1, b.y2))
        hi = min(max(a.y1, a.y2), max(b.y1, b.y2))
    return max(0.0, hi - lo)


def classify_geometry(
    geometry: PageGeometry,
    denominator: float,
    words: Iterable[dict[str, Any]] = (),
    *,
    min_zone_mm: float = 1500.0,
    max_wall_thickness_mm: float = 450.0,
    min_wall_length_mm: float = 800.0,
    min_column_mm: float = 120.0,
    max_column_mm: float = 1400.0,
    min_beam_length_mm: float = 1200.0,
    wall_thickness_mm: float = 230.0,
) -> list[Element]:
    """Turn raw page geometry into classified structural elements."""
    words = list(words or [])
    elements: list[Element] = []

    if geometry.is_empty():
        return elements

    # ------------------------------------------------------------------ #
    # 0. annotation regions (title block, schedules, legends)
    # ------------------------------------------------------------------ #
    panels = find_annotation_panels(words)
    page_area = max(1.0, geometry.width_pt * geometry.height_pt)

    def in_panel(x: float, y: float) -> bool:
        return any(_point_in_rect(x, y, p, tol=1.0) for p in panels)

    # ------------------------------------------------------------------ #
    # 1. rectangles: zones, columns, walls
    # ------------------------------------------------------------------ #
    zones: list[Rect] = []
    columns: list[Rect] = []
    wall_rects: list[Rect] = []

    for rect in geometry.rects:
        w_mm = pt_to_real(abs(rect.w), denominator)
        h_mm = pt_to_real(abs(rect.h), denominator)
        if w_mm <= 0 or h_mm <= 0:
            continue

        # legend / schedule / title-block panels carry a lot of text
        text_density = _words_in_rect(words, rect)
        if text_density > 12 and (rect.area / page_area) > 0.01:
            continue
        if any(_rect_overlaps(rect, p, 0.35) for p in panels):
            continue

        shorter, longer = min(w_mm, h_mm), max(w_mm, h_mm)

        if shorter >= min_zone_mm and longer >= min_zone_mm:
            zones.append(rect)
        elif min_column_mm <= shorter and longer <= max_column_mm and shorter / max(longer, 1e-6) >= 0.35:
            columns.append(rect)
        elif shorter <= max_wall_thickness_mm and longer >= min_wall_length_mm:
            wall_rects.append(rect)

    # the building footprint - used to reject grid extension lines
    if zones:
        zone_bbox = Rect(
            min(z.x for z in zones),
            min(z.y for z in zones),
            max(z.x2 for z in zones) - min(z.x for z in zones),
            max(z.y2 for z in zones) - min(z.y for z in zones),
        )
    else:
        zone_bbox = Rect(
            geometry.bounds[0], geometry.bounds[1],
            geometry.width_pt, geometry.height_pt,
        )

    # ------------------------------------------------------------------ #
    # 2. segments -> grid lines, walls (parallel pairs), beams
    # ------------------------------------------------------------------ #
    h_segs = [s for s in geometry.segments if s.is_horizontal and s.length_pt > 3.0]
    v_segs = [s for s in geometry.segments if s.is_vertical and s.length_pt > 3.0]

    used_as_wall: set[int] = set()
    wall_elements: list[Element] = []

    def _pair_walls(segs: list[Segment], horizontal: bool) -> None:
        """Detect double-line wall conventions."""
        n = len(segs)
        for i in range(n):
            a = segs[i]
            if i in used_as_wall:
                continue
            a_len = pt_to_real(a.length_pt, denominator)
            if a_len < min_wall_length_mm:
                continue
            best: tuple[float, int] | None = None
            for j in range(i + 1, n):
                if j in used_as_wall:
                    continue
                b = segs[j]
                gap_pt = (
                    abs(((b.y1 + b.y2) / 2.0) - ((a.y1 + a.y2) / 2.0))
                    if horizontal
                    else abs(((b.x1 + b.x2) / 2.0) - ((a.x1 + a.x2) / 2.0))
                )
                gap_mm = pt_to_real(gap_pt, denominator)
                if not (20.0 <= gap_mm <= max_wall_thickness_mm):
                    continue
                shared = _overlap_len(a, b, horizontal)
                if shared < 0.45 * min(a.length_pt, b.length_pt):
                    continue
                if best is None or gap_mm < best[0]:
                    best = (gap_mm, j)
            if best is None:
                continue
            gap_mm, j = best
            b = segs[j]
            used_as_wall.add(i)
            used_as_wall.add(j)
            shared_pt = _overlap_len(a, b, horizontal)
            x1 = min(a.x1, a.x2, b.x1, b.x2)
            x2 = max(a.x1, a.x2, b.x1, b.x2)
            y1 = min(a.y1, a.y2, b.y1, b.y2)
            y2 = max(a.y1, a.y2, b.y1, b.y2)
            wall_elements.append(
                Element(
                    kind="wall",
                    source="vector",
                    confidence=0.72,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    length_mm=pt_to_real(shared_pt, denominator),
                    width_mm=gap_mm,
                    metadata={"convention": "double-line", "direction": "h" if horizontal else "v"},
                )
            )

    _pair_walls(h_segs, True)
    _pair_walls(v_segs, False)

    # ------------------------------------------------------------------ #
    # 3. long segments that are not walls -> beams
    # ------------------------------------------------------------------ #
    # Grid / axis lines are drawn much lighter than beams - use the stroke
    # width relative to the heaviest line on the sheet as the signal.
    candidate_beams = [
        s for s in geometry.segments
        if s.is_axis_aligned and pt_to_real(s.length_pt, denominator) >= min_beam_length_mm
    ]
    widths = [s.width for s in candidate_beams if s.width > 0]
    max_width = max(widths) if widths else 0.0

    beam_elements: list[Element] = []
    for idx, seg in enumerate(geometry.segments):
        if idx in used_as_wall:
            continue
        if not seg.is_axis_aligned:
            continue
        length_mm = pt_to_real(seg.length_pt, denominator)
        if length_mm < min_beam_length_mm:
            continue

        mid_x = (seg.x1 + seg.x2) / 2.0
        mid_y = (seg.y1 + seg.y2) / 2.0
        if in_panel(mid_x, mid_y):
            continue
        # grid extension lines live outside the building footprint
        if not _segment_reaches(seg, zone_bbox):
            continue

        has_width = max_width > 0 and seg.width > 0
        is_light = has_width and seg.width < 0.35 * max_width
        is_heavy = has_width and seg.width >= 0.60 * max_width

        # light-weight lines are grid / axis references, not structure
        if is_light:
            continue
        # a light line bounding a zone is the room outline (a wall edge);
        # a heavy line there is a beam drawn over the grid
        if any(_touches_zone(seg, z) for z in zones) and not is_heavy:
            continue

        beam_elements.append(
            Element(
                kind="beam",
                source="vector",
                confidence=0.55,
                x1=seg.x1,
                y1=seg.y1,
                x2=seg.x2,
                y2=seg.y2,
                length_mm=length_mm,
                width_mm=wall_thickness_mm,
                metadata={"direction": "h" if seg.is_horizontal else "v", "stroke": round(seg.width, 2)},
            )
        )

    # ------------------------------------------------------------------ #
    # 4. emit
    # ------------------------------------------------------------------ #
    for rect in zones:
        w_mm = pt_to_real(abs(rect.w), denominator)
        h_mm = pt_to_real(abs(rect.h), denominator)
        elements.append(
            Element(
                kind="slab",
                source="vector",
                confidence=0.6,
                x1=rect.x,
                y1=rect.y,
                x2=rect.x2,
                y2=rect.y2,
                length_mm=w_mm,
                width_mm=h_mm,
                area_mm2=w_mm * h_mm,
                perimeter_mm=2.0 * (w_mm + h_mm),
                metadata={"role": "zone"},
            )
        )

    for rect in columns:
        w_mm = pt_to_real(abs(rect.w), denominator)
        h_mm = pt_to_real(abs(rect.h), denominator)
        elements.append(
            Element(
                kind="column",
                source="vector",
                confidence=0.7,
                x1=rect.x,
                y1=rect.y,
                x2=rect.x2,
                y2=rect.y2,
                length_mm=w_mm,
                width_mm=w_mm,
                depth_mm=h_mm,
                area_mm2=w_mm * h_mm,
                perimeter_mm=2.0 * (w_mm + h_mm),
                metadata={"shape": f"{w_mm:.0f}x{h_mm:.0f} mm"},
            )
        )

    for rect in wall_rects:
        w_mm = pt_to_real(abs(rect.w), denominator)
        h_mm = pt_to_real(abs(rect.h), denominator)
        length_mm, thick_mm = max(w_mm, h_mm), min(w_mm, h_mm)
        elements.append(
            Element(
                kind="wall",
                source="vector",
                confidence=0.68,
                x1=rect.x,
                y1=rect.y,
                x2=rect.x2,
                y2=rect.y2,
                length_mm=length_mm,
                width_mm=thick_mm,
                metadata={"convention": "closed-rect"},
            )
        )

    elements.extend(wall_elements)
    elements.extend(beam_elements)
    return elements


def _touches_zone(seg: Segment, zone: Rect, tol: float = 2.5) -> bool:
    """True when the segment lies along an edge of the zone rectangle."""
    if seg.is_horizontal:
        if abs(seg.y1 - zone.y) > tol and abs(seg.y1 - zone.y2) > tol:
            return False
        lo, hi = min(seg.x1, seg.x2), max(seg.x1, seg.x2)
        return not (hi < zone.x - tol or lo > zone.x2 + tol)
    if seg.is_vertical:
        if abs(seg.x1 - zone.x) > tol and abs(seg.x1 - zone.x2) > tol:
            return False
        lo, hi = min(seg.y1, seg.y2), max(seg.y1, seg.y2)
        return not (hi < zone.y - tol or lo > zone.y2 + tol)
    return False


def _segment_reaches(seg: Segment, box: Rect, tol: float = 2.0) -> bool:
    """True when the segment touches or crosses the given rectangle.

    Grid extension lines and dimension lines sit entirely outside the
    building footprint and must not be counted as beams.
    """
    if seg.is_horizontal:
        if seg.y1 < box.y - tol or seg.y1 > box.y2 + tol:
            return False
        lo, hi = min(seg.x1, seg.x2), max(seg.x1, seg.x2)
        return not (hi < box.x - tol or lo > box.x2 + tol)
    if seg.is_vertical:
        if seg.x1 < box.x - tol or seg.x1 > box.x2 + tol:
            return False
        lo, hi = min(seg.y1, seg.y2), max(seg.y1, seg.y2)
        return not (hi < box.y - tol or lo > box.y2 + tol)
    return True


def _rect_overlaps(a: Rect, b: Rect, min_ratio: float = 0.2) -> bool:
    """True when rectangle ``a`` overlaps ``b`` by at least ``min_ratio``."""
    ox = min(a.x2, b.x2) - max(a.x, b.x)
    oy = min(a.y2, b.y2) - max(a.y, b.y)
    if ox <= 0 or oy <= 0:
        return False
    inter = ox * oy
    area = max(1e-6, a.area)
    return inter / area >= min_ratio


def merge_slabs(elements: list[Element], overlap_ratio: float = 0.35) -> list[Element]:
    """Merge nested/overlapping slab zones into a single floor area."""
    slabs = [e for e in elements if e.kind == "slab"]
    others = [e for e in elements if e.kind != "slab"]
    if len(slabs) < 2:
        return elements

    kept: list[Element] = []
    for slab in sorted(slabs, key=lambda e: -(e.area_mm2 or 0.0)):
        merged = False
        for k in kept:
            inter = _rect_overlap_mm(k, slab)
            if inter and slab.area_mm2 and inter / slab.area_mm2 > overlap_ratio:
                k.metadata.setdefault("merged_zones", 0)
                k.metadata["merged_zones"] += 1
                merged = True
                break
        if not merged:
            kept.append(slab)
    return others + kept


def _rect_overlap_mm(a: Element, b: Element) -> float:
    ba, bb = a.bbox, b.bbox
    if not ba or not bb:
        return 0.0
    ox = min(ba[2], bb[2]) - max(ba[0], bb[0])
    oy = min(ba[3], bb[3]) - max(ba[1], bb[1])
    if ox <= 0 or oy <= 0:
        return 0.0
    denom = min(1.0, 1.0)
    return (ox * oy) / denom
