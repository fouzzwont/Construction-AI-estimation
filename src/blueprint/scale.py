"""Drawing-scale detection.

Three independent strategies are combined and the most confident one wins:

1. **Text** - ``SCALE 1:100`` in the title block or a general note.
2. **Dimension** - a numeric dimension label matched against the length of
   the dimension line next to it (``3200`` next to a 28.3 pt line at 1:100).
3. **Scale bar** - a graphical bar with ticks plus its numeric label.

Because the scale multiplies every single quantity, the engine refuses to
guess silently: when confidence is low the caller falls back to a user
supplied value (the UI exposes a scale override field).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Iterable

from .models import ScaleInfo
from .vector import PT_PER_MM, PageGeometry, Rect, Segment

logger = logging.getLogger(__name__)

SCALE_TEXT_RE = re.compile(r"(?:scale|scl)\s*[:\-]?\s*1\s*[:/]\s*(\d{1,4})", re.IGNORECASE)
RATIO_RE = re.compile(r"\b1\s*[:/]\s*(\d{1,4})\b")
CM_TO_M_RE = re.compile(r"1\s*cm\s*=\s*(\d{1,4})\s*(m|mt|meter|metre)", re.IGNORECASE)
DIMENSION_RE = re.compile(r"^\d{3,5}$")
METRE_LABEL_RE = re.compile(r"^\d{1,4}(\.\d+)?$")

MIN_DENOM = 5.0
MAX_DENOM = 2000.0


def _words_with_bbox(words: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for w in words or []:
        try:
            out.append(
                {
                    "text": str(w[4]).strip(),
                    "x0": float(w[0]),
                    "y0": float(w[1]),
                    "x1": float(w[2]),
                    "y1": float(w[3]),
                    "cx": (float(w[0]) + float(w[2])) / 2.0,
                    "cy": (float(w[1]) + float(w[3])) / 2.0,
                }
            )
        except Exception:
            continue
    return [w for w in out if w["text"]]


def detect_scale_from_text(text: str) -> ScaleInfo | None:
    """Look for an explicit scale statement in the page text."""
    if not text:
        return None

    m = SCALE_TEXT_RE.search(text)
    if m:
        d = float(m.group(1))
        if MIN_DENOM <= d <= MAX_DENOM:
            return ScaleInfo(d, "text", 0.95, m.group(0).strip())

    m = CM_TO_M_RE.search(text)
    if m:
        d = float(m.group(1))
        if MIN_DENOM <= d <= MAX_DENOM:
            return ScaleInfo(d, "text", 0.9, m.group(0).strip())

    m = RATIO_RE.search(text)
    if m:
        d = float(m.group(1))
        if MIN_DENOM <= d <= MAX_DENOM:
            return ScaleInfo(d, "text", 0.7, m.group(0).strip())

    return None


def detect_scale_from_dimensions(
    words: list[dict[str, Any]], geometry: PageGeometry
) -> ScaleInfo | None:
    """Match numeric dimension labels against the line they annotate."""
    candidates: list[float] = []
    samples: list[str] = []

    segments = [s for s in geometry.segments if s.is_axis_aligned and 4.0 <= s.length_pt <= 400.0]
    if not segments:
        return None

    for w in words:
        raw = w["text"].replace(",", "")
        if not DIMENSION_RE.match(raw):
            continue
        try:
            dim_mm = float(raw)
        except ValueError:
            continue
        if not (100.0 <= dim_mm <= 200000.0):
            continue

        best: Segment | None = None
        best_dist = 1e9
        for seg in segments:
            mx, my = (seg.x1 + seg.x2) / 2.0, (seg.y1 + seg.y2) / 2.0
            dist = abs(mx - w["cx"]) + abs(my - w["cy"])
            if dist < best_dist:
                best_dist, best = dist, seg

        if best is None or best_dist > 220.0:
            continue

        paper_mm = best.length_pt / PT_PER_MM
        if paper_mm <= 0.5:
            continue
        denom = dim_mm / paper_mm
        if MIN_DENOM <= denom <= MAX_DENOM:
            candidates.append(denom)
            samples.append(f"{int(dim_mm)}mm over {paper_mm:.1f}mm -> 1:{denom:.0f}")

    if len(candidates) < 2:
        return None

    # take the modal (rounded) value so one stray label cannot dominate
    rounded = [round(c / 5.0) * 5.0 for c in candidates]
    best_value, count = Counter(rounded).most_common(1)[0]
    if count < 2:
        return None
    confidence = min(0.9, 0.45 + 0.12 * count)
    return ScaleInfo(best_value, "dimension", confidence, "; ".join(samples[:3]))


def detect_scale_from_bar(
    words: list[dict[str, Any]], geometry: PageGeometry
) -> ScaleInfo | None:
    """Detect a classic graphical scale bar (a line with perpendicular ticks)."""
    h_segs = [s for s in geometry.segments if s.is_horizontal and s.length_pt > 30.0]
    v_segs = [s for s in geometry.segments if s.is_vertical and s.length_pt > 1.0 and s.length_pt < 20.0]
    if not h_segs or not v_segs:
        return None

    for bar in h_segs:
        ticks = [
            v
            for v in v_segs
            if min(bar.x1, bar.x2) - 2 <= v.x1 <= max(bar.x1, bar.x2) + 2
            and min(bar.y1, bar.y2) - 4 <= v.y1 <= max(bar.y1, bar.y2) + 4
        ]
        if len(ticks) < 3:
            continue

        right_x = max(bar.x1, bar.x2)
        near = [
            w
            for w in words
            if w["cx"] > right_x - 140 and abs(w["cy"] - bar.y1) < 30 and METRE_LABEL_RE.match(w["text"])
        ]
        if not near:
            continue
        label = max(near, key=lambda w: w["cx"])
        try:
            metres = float(label["text"])
        except ValueError:
            continue
        if metres <= 0:
            continue
        paper_mm = bar.length_pt / PT_PER_MM
        if paper_mm <= 1:
            continue
        denom = (metres * 1000.0) / paper_mm
        if MIN_DENOM <= denom <= MAX_DENOM:
            return ScaleInfo(denom, "bar", 0.8, f"scale bar labelled {metres} m")

    return None


def detect_scale(
    text: str,
    words: Iterable[dict[str, Any]],
    geometry: PageGeometry,
    default: float = 100.0,
) -> ScaleInfo:
    """Combine every strategy and return the most trustworthy result."""
    word_list = _words_with_bbox(words)

    strategies = [
        detect_scale_from_text(text),
        detect_scale_from_dimensions(word_list, geometry),
        detect_scale_from_bar(word_list, geometry),
    ]
    strategies = [s for s in strategies if s is not None]

    if strategies:
        # Prefer text > bar > dimension when confidences are close.
        order = {"text": 3, "bar": 2, "dimension": 1}
        strategies.sort(key=lambda s: (s.confidence + 0.05 * order.get(s.method, 0)), reverse=True)
        best = strategies[0]
        if best.confidence >= 0.5:
            return best

    return ScaleInfo(default, "default", 0.2, "no scale statement found; using default")


def consolidate_scales(scales: list[ScaleInfo]) -> ScaleInfo:
    """Pick one scale for the whole drawing set (title block wins)."""
    if not scales:
        return ScaleInfo()
    explicit = [s for s in scales if s.method in ("text", "manual") and s.confidence >= 0.6]
    if explicit:
        explicit.sort(key=lambda s: s.confidence, reverse=True)
        return explicit[0]
    valued = [s for s in scales if s.confidence >= 0.45]
    if valued:
        valued.sort(key=lambda s: s.confidence, reverse=True)
        return valued[0]
    return scales[0]
