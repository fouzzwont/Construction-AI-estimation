"""Shared data structures for the estimation pipeline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def num(value: Any, default: float = 0.0) -> float:
    """Coerce numpy / pandas scalars into plain Python floats."""
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


@dataclass
class Element:
    """A single detected construction element on a drawing page.

    Geometry is stored twice: in **PDF points** (``bbox``) for drawing the
    annotation overlay, and in **real-world millimetres** (the ``*_mm``
    fields) for the quantity take-off.
    """

    kind: str                       # column | beam | slab | wall | footing | rebar
    page: int = 0
    source: str = "vector"          # vector | cv | yolo | schedule | manual
    confidence: float = 0.5

    # page-space geometry (PDF points)
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None

    # real-world dimensions (mm)
    length_mm: float | None = None
    width_mm: float | None = None
    depth_mm: float | None = None
    height_mm: float | None = None
    area_mm2: float | None = None
    perimeter_mm: float | None = None

    # rebar specifics
    diameter_mm: float | None = None
    count: int | None = None
    spacing_mm: float | None = None

    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def bbox(self) -> tuple[float, float, float, float] | None:
        if None in (self.x1, self.y1, self.x2, self.y2):
            return None
        x1, x2 = sorted((float(self.x1), float(self.x2)))
        y1, y2 = sorted((float(self.y1), float(self.y2)))
        return (x1, y1, x2, y2)

    @property
    def centre(self) -> tuple[float, float] | None:
        b = self.bbox
        if b is None:
            return None
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox) if self.bbox else None
        data["centre"] = list(self.centre) if self.centre else None
        for key in (
            "x1", "y1", "x2", "y2", "length_mm", "width_mm", "depth_mm", "height_mm",
            "area_mm2", "perimeter_mm", "diameter_mm", "spacing_mm", "confidence",
        ):
            if data.get(key) is not None:
                data[key] = num(data[key])
        if data.get("count") is not None:
            data["count"] = int(data["count"])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Element":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class RebarItem:
    """A line item from a reinforcement schedule."""

    diameter_mm: float
    length_m: float
    count: int
    spacing_mm: float | None = None
    member: str = ""
    page: int = 0
    confidence: float = 0.6
    source: str = "schedule"

    @property
    def total_length_m(self) -> float:
        return self.length_m * self.count

    @property
    def weight_kg(self) -> float:
        """Bar weight from the standard formula: W = d^2 / 162 (kg/m)."""
        return self.total_length_m * (self.diameter_mm ** 2) / 162.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_length_m"] = round(num(self.total_length_m), 2)
        data["weight_kg"] = round(num(self.weight_kg), 2)
        data["weight_t"] = round(num(self.weight_kg) / 1000.0, 4)
        data["diameter_mm"] = num(self.diameter_mm)
        data["length_m"] = num(self.length_m)
        data["count"] = int(self.count)
        if self.spacing_mm is not None:
            data["spacing_mm"] = num(self.spacing_mm)
        return data


@dataclass
class ScaleInfo:
    """Drawing scale, e.g. denominator 100 means 1:100."""

    denominator: float = 100.0
    method: str = "default"          # text | bar | dimension | manual | default
    confidence: float = 0.2
    evidence: str = ""

    @property
    def ratio(self) -> str:
        d = self.denominator
        return f"1:{int(d)}" if abs(d - round(d)) < 0.01 else f"1:{d:.1f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "denominator": self.denominator,
            "ratio": self.ratio,
            "method": self.method,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
        }


@dataclass
class PageAnalysis:
    """Everything the engine learned about a single page."""

    index: int
    image_path: str | None = None
    width_pt: float = 0.0
    height_pt: float = 0.0
    zoom: float = 1.0
    elements: list[Element] = field(default_factory=list)
    rebar_items: list[RebarItem] = field(default_factory=list)
    scale: ScaleInfo = field(default_factory=ScaleInfo)
    notes: list[str] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "image_path": self.image_path,
            "width_pt": self.width_pt,
            "height_pt": self.height_pt,
            "zoom": self.zoom,
            "scale": self.scale.to_dict(),
            "elements": [e.to_dict() for e in self.elements],
            "rebar_items": [r.to_dict() for r in self.rebar_items],
            "notes": self.notes,
            "text_preview": self.text[:600],
        }


@dataclass
class QuantityLine:
    """One row of the measured BOQ."""

    category: str                     # Concrete | Reinforcement | Formwork | Masonry
    item: str                         # M25 concrete, Fe500 D16 steel, ...
    quantity: float
    unit: str                         # m3 | kg | t | m2
    rate: float
    amount: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "item": self.item,
            "quantity": round(self.quantity, 3),
            "unit": self.unit,
            "rate": round(self.rate, 2),
            "amount": round(self.amount, 2),
            "detail": self.detail,
        }


def rect_area_mm2(length_mm: float | None, width_mm: float | None) -> float | None:
    if length_mm is None or width_mm is None:
        return None
    return length_mm * width_mm


def circle_area_mm2(radius_mm: float | None) -> float | None:
    if radius_mm is None:
        return None
    return math.pi * radius_mm * radius_mm
