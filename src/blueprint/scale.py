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

# Imperial (US) drawing scales. ``SCALE 1/4" = 1'-0"`` means a quarter inch on
# paper represents one foot, i.e. 1:48 - *not* 1:4. Reading the bare fraction as
# the denominator inflates every area by a factor of 144, so these forms are
# parsed before the metric ones.
_IMPERIAL_FRACTION_RE = re.compile(
    r'(?:(\d+)\s*[-\s]\s*)?(\d+)\s*/\s*(\d+)\s*["\u201d]\s*=\s*(\d+(?:\.\d+)?)\s*(?:\'|\u2019|ft\b|feet\b)',
    re.IGNORECASE,
)
# A stacked fraction extracts without its slash: ``SCALE 1 4" = 1'-0"``.
_IMPERIAL_STACKED_RE = re.compile(
    r'SCALE\s*[:\-]?\s*1\s+(\d{1,2})\s*["\u201d]\s*=\s*(\d+(?:\.\d+)?)\s*(?:\'|\u2019|ft\b|feet\b)',
    re.IGNORECASE,
)
# ``SCALE 1" = 50'`` -> 1:600 (site plans).
_IMPERIAL_WHOLE_RE = re.compile(
    r'(?<![\d/.])(\d+(?:\.\d+)?)\s*["\u201d]\s*=\s*(\d+(?:\.\d+)?)\s*(?:\'|\u2019|ft\b|feet\b)',
    re.IGNORECASE,
)
NTS_RE = re.compile(r"\bN\.?\s?T\.?\s?S\.?\b|\bnot\s+to\s+scale\b", re.IGNORECASE)

# Sheets that describe the building in elevation/section cannot produce plan
# quantities: a wall seen edge-on is a line, not a footprint. Pages that are
# explicitly plans are never skipped, even if they reference a section.
NON_PLAN_RE = re.compile(r"\b(elevation|section|detail)\b", re.IGNORECASE)
PLAN_RE = re.compile(r"\bplan\b", re.IGNORECASE)

MIN_DENOM = 5.0
MAX_DENOM = 2000.0

# Conventional denominators that detected scales are snapped to. Values far
# from every convention are treated as measurement noise and rejected.
STANDARD_DENOMINATORS = (
    10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400,
    500, 600, 750, 1000, 1250, 1500, 2000, 2500,
)


def snap_to_standard(denominator: float, tolerance: float = 0.12) -> float | None:
    """Return the conventional scale nearest ``denominator``, or None if far."""
    if denominator <= 0:
        return None
    best = min(STANDARD_DENOMINATORS, key=lambda s: abs(denominator - s) / s)
    if abs(denominator - best) / best <= tolerance:
        return float(best)
    return None


def _imperial_scale(text: str) -> tuple[float, str] | None:
    """Interpret ``1/4" = 1'-0"`` style scales as a ratio denominator."""
    m = _IMPERIAL_FRACTION_RE.search(text)
    if m:
        whole = float(m.group(1) or 0.0)
        paper_in = whole + float(m.group(2)) / float(m.group(3) or 1)
        feet = float(m.group(4))
        if paper_in > 0 and feet > 0:
            return (feet * 12.0) / paper_in, m.group(0).strip()

    m = _IMPERIAL_STACKED_RE.search(text)
    if m:
        paper_in = 1.0 / float(m.group(1))
        feet = float(m.group(2))
        if feet > 0:
            return (
                (feet * 12.0) / paper_in,
                f'{m.group(0).strip()} (stacked 1/{m.group(1)}")',
            )

    m = _IMPERIAL_WHOLE_RE.search(text)
    if m:
        paper_in = float(m.group(1))
        feet = float(m.group(2))
        if paper_in > 0 and feet > 0:
            return (feet * 12.0) / paper_in, m.group(0).strip()

    return None


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


def detect_scale_from_text(text: str, default: float = 100.0) -> ScaleInfo | None:
    """Look for an explicit scale statement in the page text."""
    if not text:
        return None

    imperial = _imperial_scale(text)
    if imperial:
        denom, evidence = imperial
        if MIN_DENOM <= denom <= MAX_DENOM:
            return ScaleInfo(denom, "text", 0.95, evidence)

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

    # "SCALE: NTS" means the sheet carries no usable scale. Say so instead of
    # letting the scale bar / detail references invent one.
    if NTS_RE.search(text):
        return ScaleInfo(default, "nts", 0.95, "sheet marked NTS (not to scale)")

    m = RATIO_RE.search(text)
    if m:
        # Only a bare ratio that is already a conventional scale is trusted;
        # otherwise detail references such as "detail 1/14" are read as 1:14.
        snapped = snap_to_standard(float(m.group(1)), tolerance=0.03)
        if snapped is not None and MIN_DENOM <= snapped <= MAX_DENOM:
            return ScaleInfo(snapped, "text", 0.7, m.group(0).strip())

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
    snapped = snap_to_standard(best_value, tolerance=0.15)
    value = snapped if snapped is not None else best_value
    confidence = min(0.9, 0.45 + 0.12 * count)
    return ScaleInfo(value, "dimension", confidence, "; ".join(samples[:3]))


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
            snapped = snap_to_standard(denom, tolerance=0.15)
            value = snapped if snapped is not None else denom
            evidence = f"scale bar labelled {metres} m"
            if snapped is not None and abs(snapped - denom) > 0.5:
                evidence += f" (measured 1:{denom:.0f})"
            return ScaleInfo(value, "bar", 0.8, evidence)

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
        detect_scale_from_text(text, default),
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


def is_non_plan_sheet(text: str) -> bool:
    """True when the sheet is an elevation/section/detail rather than a plan."""
    if not text:
        return False
    return bool(NON_PLAN_RE.search(text)) and not bool(PLAN_RE.search(text))


def consolidate_scales(scales: list[ScaleInfo]) -> ScaleInfo:
    """Pick one scale for the whole drawing set (the modal title-block value)."""
    if not scales:
        return ScaleInfo()
    # NTS sheets carry no scale, so they must never define the document scale.
    explicit = [
        s
        for s in scales
        if s.method in ("text", "manual") and s.confidence >= 0.6
    ]
    if explicit:
        # The modal scale wins, not the first page: a site plan at 1:600 must
        # not define the architectural sheets of the same set.
        groups: dict[float, list[ScaleInfo]] = {}
        for s in explicit:
            groups.setdefault(round(s.denominator, 1), []).append(s)
        best_key = max(
            groups,
            key=lambda k: (len(groups[k]), max(x.confidence for x in groups[k])),
        )
        return max(groups[best_key], key=lambda s: s.confidence)
    valued = [s for s in scales if s.confidence >= 0.45 and s.method != "nts"]
    if valued:
        valued.sort(key=lambda s: s.confidence, reverse=True)
        return valued[0]
    return scales[0]
