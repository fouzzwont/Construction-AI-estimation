"""Reinforcement-schedule (BBS) parsing.

Most construction drawings contain a *Bar Bending Schedule* table.  When the
PDF keeps its text as text (the usual case for CAD exports) the schedule is
by far the most accurate rebar source available - far better than guessing
from the drawing geometry.  The parser below turns table rows into
:class:`~src.blueprint.models.RebarItem` objects.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .models import RebarItem

logger = logging.getLogger(__name__)

SCHEDULE_KEYWORDS = (
    "rebar schedule",
    "bar bending schedule",
    "schedule of reinforcement",
    "reinforcement schedule",
    "bar schedule",
    "steel schedule",
    "bbs",
    "rebar details",
    "reinforcement details",
)

DIAMETER_RE = re.compile(r"[ØφΦ⌀∅\ufffd]?\s*(?:DIA|DIA\.|D)\s*[:\-]?\s*(\d{1,2})(?!\d)", re.IGNORECASE)
# \ufffd covers PDFs whose font encoding mangles the diameter sign
DIAMETER_SYMBOL_RE = re.compile(r"[ØφΦ⌀∅\ufffd]\s*(\d{1,2})(?!\d)")
SPACING_RE = re.compile(r"(\d{2,3})\s*(?:c\s*/\s*c|cc|mm\s*c\s*/\s*c)", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d{1,6}")
MARK_RE = re.compile(r"^[A-Z]{1,3}\s*[-–]?\s*\d{1,3}$")


@dataclass
class TextLine:
    """A single line of text with its bounding box."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0


def extract_text_lines(text_dict: dict[str, Any] | list[Any]) -> list[TextLine]:
    """Flatten PyMuPDF ``get_text('dict')`` output into positioned lines.

    PDF text objects are individual cells, so a table row arrives as several
    separate lines.  Cells that share a baseline are therefore merged back
    into a single logical line, which is what the schedule parser expects.
    """
    blocks = text_dict if isinstance(text_dict, list) else (text_dict or {}).get("blocks", [])
    raw: list[TextLine] = []
    for block in blocks:
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            bbox = line.get("bbox", (0, 0, 0, 0))
            raw.append(TextLine(text, float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))

    return _merge_by_baseline(raw)


def _merge_by_baseline(lines: list[TextLine], y_tolerance: float = 2.5) -> list[TextLine]:
    """Merge text cells that sit on the same baseline into one line."""
    merged: list[TextLine] = []
    for line in sorted(lines, key=lambda l: l.cy):
        if merged and abs(line.cy - merged[-1].cy) <= y_tolerance:
            prev = merged[-1]
            merged[-1] = TextLine(
                f"{prev.text} {line.text}",
                min(prev.x0, line.x0),
                min(prev.y0, line.y0),
                max(prev.x1, line.x1),
                max(prev.y1, line.y1),
            )
        else:
            merged.append(line)
    return merged


def _find_schedule_region(lines: list[TextLine], page_height: float) -> tuple[float, float] | None:
    """Return the vertical band (y0, y1) that contains the schedule table.

    The heading may sit *above* the table (typical) or *below* it (a caption),
    so the band is generous and the caller only uses it to boost confidence.
    """
    for line in lines:
        low = line.text.lower()
        for kw in SCHEDULE_KEYWORDS:
            if kw in low:
                return (max(0.0, line.y0 - 40.0), min(page_height or 1000.0, line.y1 + 320.0))
    return None


def _parse_row(text: str, page: int) -> RebarItem | None:
    """Parse a single schedule row into a rebar item."""
    raw = text.strip()
    if not raw:
        return None

    diameter: float | None = None
    m = DIAMETER_SYMBOL_RE.search(raw)
    if m:
        diameter = float(m.group(1))
    else:
        m = DIAMETER_RE.search(raw)
        if m:
            diameter = float(m.group(1))

    if diameter is None or not (4.0 <= diameter <= 50.0):
        return None

    spacing: float | None = None
    working = raw
    ms = SPACING_RE.search(working)
    if ms:
        spacing = float(ms.group(1))
        working = working.replace(ms.group(0), " ")

    # strip the diameter occurrences so they do not pollute the number list
    working = DIAMETER_SYMBOL_RE.sub(" ", working)
    working = DIAMETER_RE.sub(" ", working)

    nums = [int(n) for n in NUMBER_RE.findall(working)]
    nums = [n for n in nums if n > 0]
    if len(nums) < 2:
        # a schedule row always carries at least a length and a quantity
        return None

    length_mm: float | None = None
    count = 1

    length_candidates = [n for n in nums if 400 <= n <= 12000]
    count_candidates = [n for n in nums if 1 <= n <= 3000]

    if length_candidates:
        length_mm = float(max(length_candidates))
        count_candidates = [n for n in count_candidates if n != length_mm]
    if count_candidates:
        count = max(count_candidates)

    if length_mm is None:
        return None

    member = ""
    head = re.split(r"[\dØφΦ]", raw, maxsplit=1)[0].strip(" -–:|")
    if head and len(head) <= 24:
        member = head

    return RebarItem(
        diameter_mm=diameter,
        length_m=length_mm / 1000.0,
        count=count,
        spacing_mm=spacing,
        member=member,
        page=page,
        confidence=0.75,
        source="schedule",
    )


def parse_rebar_schedule(
    lines: list[TextLine], page: int = 0, page_height: float = 0.0
) -> tuple[list[RebarItem], list[str]]:
    """Extract rebar items from a reinforcement schedule table.

    Returns the parsed items plus human readable notes about what was found.
    """
    notes: list[str] = []
    if not lines:
        return [], notes

    region = _find_schedule_region(lines, page_height or 1000.0)
    if region:
        notes.append(f"schedule heading detected near y={region[0]:.0f}")

    items: list[RebarItem] = []
    for line in lines:
        text = line.text
        if len(text) < 4 or len(text) > 90:
            continue
        item = _parse_row(text, page)
        if item is None:
            continue
        # boost confidence for rows close to the schedule heading
        if region and region[0] <= line.cy <= region[1]:
            item.confidence = min(0.95, item.confidence + 0.15)
        items.append(item)

    # de-duplicate identical rows (some PDFs repeat the table per floor)
    seen: set[tuple[float, float, int]] = set()
    unique: list[RebarItem] = []
    for it in items:
        key = (round(it.diameter_mm, 1), round(it.length_m, 3), it.count)
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)

    if unique:
        total_kg = sum(i.weight_kg for i in unique)
        notes.append(f"{len(unique)} rebar rows parsed ({total_kg/1000:.2f} t)")
    else:
        notes.append("no rebar schedule rows could be parsed")

    return unique, notes


def find_bar_marks(lines: list[TextLine]) -> dict[str, list[TextLine]]:
    """Collect bar marks (e.g. ``B1``, ``SL-3``) and where they appear."""
    marks: dict[str, list[TextLine]] = {}
    for line in lines:
        for token in re.split(r"[\s,]+", line.text):
            token = token.strip(".,;:()[]")
            if MARK_RE.match(token) and any(ch.isdigit() for ch in token):
                marks.setdefault(token, []).append(line)
    return marks
