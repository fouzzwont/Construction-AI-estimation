"""Quantity take-off and cost estimation.

Turns detected elements into a measured Bill of Quantities:

============  ==========================================================
element       derived quantities
============  ==========================================================
column        concrete volume, formwork area, reinforcement weight
beam          concrete volume, formwork area, reinforcement weight
slab / zone   concrete volume, formwork area, reinforcement weight
wall          concrete *or* masonry volume, formwork area
footing       concrete volume, formwork area, reinforcement weight
rebar (BBS)   length and weight per diameter, taken verbatim from the
              reinforcement schedule when the drawing provides one
============  ==========================================================

Reinforcement that cannot be read from a schedule is estimated from the
standard steel ratios (kg per m3 of concrete) and split across diameters
using the distribution configured in ``config.yaml``.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .blueprint.models import Element, PageAnalysis, RebarItem, num
from .config import Config, get_config
from .rates import RateCard, load_rate_card

logger = logging.getLogger(__name__)

MM2_PER_M2 = 1_000_000.0
MM3_PER_M3 = 1_000_000_000.0

STOREY_PATTERNS = (
    re.compile(r"\bG\s*\+\s*(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\bGF\s*\+\s*(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\bGROUND\s*FLOOR\s*\+\s*(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2})\s*(?:NO\.?\s*)?STOREYS?\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2})\s*FLOORS?\b", re.IGNORECASE),
)


@dataclass
class EstimateResult:
    """Everything needed to render a client facing estimate."""

    project_name: str
    source_file: str
    scale: dict[str, Any] = field(default_factory=dict)
    storeys: int = 1
    counts: dict[str, int] = field(default_factory=dict)
    quantities: dict[str, float] = field(default_factory=dict)
    boq: list[dict[str, Any]] = field(default_factory=list)
    rebar: list[dict[str, Any]] = field(default_factory=list)
    schedule_rows: int = 0
    pages: list[dict[str, Any]] = field(default_factory=list)
    totals: dict[str, float] = field(default_factory=dict)
    grand_total: float = 0.0
    currency_symbol: str = "₹"
    market: str = "india_urban"
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "source_file": self.source_file,
            "scale": self.scale,
            "storeys": self.storeys,
            "counts": self.counts,
            "quantities": {k: round(num(v), 3) for k, v in self.quantities.items()},
            "boq": self.boq,
            "rebar": self.rebar,
            "schedule_rows": self.schedule_rows,
            "pages": self.pages,
            "totals": {k: round(num(v), 2) for k, v in self.totals.items()},
            "grand_total": round(num(self.grand_total), 2),
            "currency_symbol": self.currency_symbol,
            "market": self.market,
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "confidence": round(self.confidence, 3),
        }


class Estimator:
    """Compute quantities and costs from analysed pages."""

    def __init__(self, config: Config | None = None, rate_card: RateCard | None = None):
        self.cfg = config or get_config()
        self.est = self.cfg.section("estimation")
        self.rates = rate_card or load_rate_card(self.cfg)

    # ------------------------------------------------------------------ #
    def estimate(
        self,
        analyses: list[PageAnalysis],
        source_file: str = "",
        project_name: str = "",
    ) -> EstimateResult:
        storeys = self._detect_storeys(analyses)
        storey_height_m = float(self.est.get("storey_height_m", 3.0))
        wall_type = str(self.est.get("wall_type", "masonry")).lower()

        concrete_grade = str(self.est.get("default_concrete_grade", "M25"))
        steel_grade = str(self.est.get("default_steel_grade", "Fe500"))

        counts: dict[str, int] = {}
        concrete_m3: dict[str, float] = {}
        formwork_m2: dict[str, float] = {}
        steel_by_dia: dict[float, float] = {}
        masonry_m3 = 0.0
        schedule_items: list[RebarItem] = []
        estimates: list[dict[str, Any]] = []

        for page in analyses:
            counts_page: dict[str, int] = {}
            for el in page.elements:
                counts_page[el.kind] = counts_page.get(el.kind, 0) + 1
                counts[el.kind] = counts.get(el.kind, 0) + 1

                if el.kind == "column":
                    self._column(el, storey_height_m, storeys, concrete_m3, formwork_m2, steel_by_dia)
                elif el.kind == "beam":
                    self._beam(el, concrete_m3, formwork_m2, steel_by_dia)
                elif el.kind == "slab":
                    self._slab(el, concrete_m3, formwork_m2, steel_by_dia)
                elif el.kind == "wall":
                    if wall_type == "rcc":
                        self._wall(el, storey_height_m, storeys, concrete_m3, formwork_m2, steel_by_dia)
                    else:
                        masonry_m3 += self._masonry(el, storey_height_m, storeys)
                elif el.kind == "footing":
                    self._footing(el, concrete_m3, formwork_m2, steel_by_dia)
                elif el.kind == "rebar":
                    # a rebar mat drawn on the plan: length x lines
                    length_m = (el.length_mm or 0.0) / 1000.0
                    dia = float(el.metadata.get("diameter_mm", 10))
                    count = int(el.count or 1)
                    steel_by_dia[dia] = steel_by_dia.get(dia, 0.0) + length_m * count * dia**2 / 162.0

            page_counts = counts_page
            analysis_payload = page.to_dict()
            analysis_payload["counts"] = page_counts
            estimates.append(analysis_payload)

        # ---- reinforcement from the schedule wins ---------------------- #
        for page in analyses:
            schedule_items.extend(page.rebar_items)

        schedule_kg = sum(i.weight_kg for i in schedule_items)
        ratio_kg = sum(steel_by_dia.values())

        if schedule_items and schedule_kg > 0:
            steel_by_dia = self._steel_from_schedule(schedule_items)
            steel_source = "reinforcement schedule"
        elif ratio_kg > 0:
            steel_source = "standard steel ratios (kg/m3)"
        else:
            steel_source = "none detected"

        # ---- BOQ ------------------------------------------------------- #
        boq: list[dict[str, Any]] = []
        totals: dict[str, float] = {}

        def add(category: str, item: str, qty: float, unit: str, rate: float, detail: str = "") -> None:
            qty = num(qty)
            if qty <= 0:
                return
            amount = num(qty * num(rate))
            boq.append(
                {
                    "category": category,
                    "item": item,
                    "quantity": round(qty, 3),
                    "unit": unit,
                    "rate": round(num(rate), 2),
                    "amount": round(amount, 2),
                    "detail": detail,
                }
            )
            totals[category] = num(totals.get(category, 0.0) + amount)

        total_concrete = sum(concrete_m3.values())
        if total_concrete > 0:
            add(
                "Concrete",
                f"{concrete_grade} ready-mix concrete",
                total_concrete,
                "m3",
                self.rates.concrete_rate(concrete_grade),
                "columns, beams, slabs and footings",
            )

        total_steel_kg = sum(steel_by_dia.values())
        if total_steel_kg > 0:
            steel_rate = self.rates.steel_rate(steel_grade)  # per tonne
            add(
                "Reinforcement",
                f"{steel_grade} TMT reinforcement steel",
                total_steel_kg / 1000.0,
                "t",
                steel_rate,
                f"source: {steel_source}",
            )

        for kind, qty in formwork_m2.items():
            if qty > 0:
                add(
                    "Formwork",
                    f"{kind.capitalize()} shuttering",
                    qty,
                    "m2",
                    self.rates.formwork_rate(kind),
                )

        if masonry_m3 > 0:
            add(
                "Masonry",
                "Brick masonry in cement mortar 1:5",
                masonry_m3,
                "m3",
                self.rates.masonry_rate(),
                "assumed from wall thickness and storey height",
            )

        grand_total = sum(totals.values())

        # ---- rebar schedule table -------------------------------------- #
        rebar_rows = []
        if schedule_items:
            agg: dict[float, dict[str, Any]] = {}
            for item in schedule_items:
                row = agg.setdefault(
                    item.diameter_mm,
                    {"diameter_mm": item.diameter_mm, "total_length_m": 0.0, "weight_kg": 0.0, "count": 0},
                )
                row["total_length_m"] += item.total_length_m
                row["weight_kg"] += item.weight_kg
                row["count"] += item.count
            for dia in sorted(agg):
                row = agg[dia]
                row["weight_t"] = round(row["weight_kg"] / 1000.0, 3)
                row["total_length_m"] = round(row["total_length_m"], 2)
                row["weight_kg"] = round(row["weight_kg"], 1)
                rebar_rows.append(row)

        # ---- assumptions & warnings ------------------------------------ #
        assumptions: list[str] = []
        warnings: list[str] = []

        scale = analyses[0].scale.to_dict() if analyses else {}
        assumptions.append(f"Drawing scale {scale.get('ratio', 'n/a')} (detected via {scale.get('method', 'n/a')}).")
        assumptions.append(f"Storey height assumed {storey_height_m:.2f} m over {storeys} storey(s).")
        assumptions.append(f"Slab thickness assumed {self.est.get('slab_thickness_mm', 125)} mm.")
        assumptions.append(f"Beam assumed {self.est.get('beam_width_mm', 230)} x {self.est.get('beam_depth_mm', 450)} mm.")
        assumptions.append(f"Wall thickness assumed {self.est.get('wall_thickness_mm', 230)} mm.")
        assumptions.append(f"Rates: {self.rates.market} (x{self.rates.multiplier:.2f}), currency INR.")
        assumptions.append(f"Reinforcement source: {steel_source}.")

        if scale.get("method") == "default":
            warnings.append(
                "Drawing scale could not be detected - quantities use the default scale. "
                "Please confirm the scale for a reliable estimate."
            )
        if not schedule_items:
            warnings.append(
                "No reinforcement schedule was readable - steel quantities are derived from "
                "standard kg/m3 ratios and should be verified."
            )
        if not analyses or not any(a.elements for a in analyses):
            warnings.append(
                "No structural elements were detected. The drawing may be scanned, or use a "
                "linework style the detector does not recognise."
            )

        confidence = self._confidence(analyses, schedule_items, counts)

        return EstimateResult(
            project_name=project_name or self.cfg.get("project.name", "Construction Estimate"),
            source_file=source_file,
            scale=scale,
            storeys=storeys,
            counts=counts,
            quantities={
                "concrete_m3": num(total_concrete),
                "steel_kg": num(total_steel_kg),
                "steel_t": num(total_steel_kg / 1000.0),
                "formwork_m2": num(sum(formwork_m2.values())),
                "masonry_m3": num(masonry_m3),
                "elements": int(sum(counts.values())),
            },
            boq=boq,
            rebar=rebar_rows,
            schedule_rows=len(schedule_items),
            pages=estimates,
            totals={k: num(v) for k, v in totals.items()},
            grand_total=num(grand_total),
            currency_symbol=self.rates.currency_symbol,
            market=self.rates.market,
            assumptions=assumptions,
            warnings=warnings,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    # per element quantity maths
    # ------------------------------------------------------------------ #
    def _ratio(self, kind: str) -> float:
        table = self.est.get("steel_ratio_kg_per_m3", {}) or {}
        return float(table.get(kind, 80))

    def _add_steel(self, steel: dict[float, float], volume_m3: float, kind: str) -> None:
        kg = volume_m3 * self._ratio(kind)
        if kg <= 0:
            return
        distribution = self.est.get("rebar_distribution", {}) or {}
        diameters = self.est.get("rebar_diameters_mm", [8, 10, 12, 16, 20, 25]) or []
        if not distribution:
            distribution = {str(d): 1.0 / max(1, len(diameters)) for d in diameters}
        total_share = sum(float(v) for v in distribution.values()) or 1.0
        for dia_str, share in distribution.items():
            try:
                dia = float(dia_str)
            except ValueError:
                continue
            steel[dia] = steel.get(dia, 0.0) + kg * (float(share) / total_share)

    def _column(
        self,
        el: Element,
        storey_h: float,
        storeys: int,
        concrete: dict[str, float],
        formwork: dict[str, float],
        steel: dict[float, float],
    ) -> None:
        w_mm = el.width_mm or el.length_mm or float(self.est.get("column_size_mm", 300))
        d_mm = el.depth_mm or w_mm
        height_m = storey_h * max(1, storeys)
        volume = (w_mm * d_mm) / MM2_PER_M2 * height_m
        concrete["column"] = concrete.get("column", 0.0) + volume
        perimeter_m = 2.0 * (w_mm + d_mm) / 1000.0
        formwork["column"] = formwork.get("column", 0.0) + perimeter_m * height_m
        self._add_steel(steel, volume, "column")

    def _beam(
        self,
        el: Element,
        concrete: dict[str, float],
        formwork: dict[str, float],
        steel: dict[float, float],
    ) -> None:
        length_m = (el.length_mm or 0.0) / 1000.0
        w_mm = el.width_mm or float(self.est.get("beam_width_mm", 230))
        d_mm = el.depth_mm or float(self.est.get("beam_depth_mm", 450))
        volume = length_m * (w_mm / 1000.0) * (d_mm / 1000.0)
        concrete["beam"] = concrete.get("beam", 0.0) + volume
        shutter = length_m * ((w_mm + 2 * d_mm) / 1000.0)
        formwork["beam"] = formwork.get("beam", 0.0) + shutter
        self._add_steel(steel, volume, "beam")

    def _slab(
        self,
        el: Element,
        concrete: dict[str, float],
        formwork: dict[str, float],
        steel: dict[float, float],
    ) -> None:
        area_m2 = (el.area_mm2 or 0.0) / MM2_PER_M2
        thickness_mm = float(self.est.get("slab_thickness_mm", 125))
        volume = area_m2 * thickness_mm / 1000.0
        concrete["slab"] = concrete.get("slab", 0.0) + volume
        formwork["slab"] = formwork.get("slab", 0.0) + area_m2
        self._add_steel(steel, volume, "slab")

    def _wall(
        self,
        el: Element,
        storey_h: float,
        storeys: int,
        concrete: dict[str, float],
        formwork: dict[str, float],
        steel: dict[float, float],
    ) -> None:
        length_m = (el.length_mm or 0.0) / 1000.0
        thick_mm = el.width_mm or float(self.est.get("wall_thickness_mm", 230))
        height_m = storey_h * max(1, storeys)
        volume = length_m * (thick_mm / 1000.0) * height_m
        concrete["wall"] = concrete.get("wall", 0.0) + volume
        formwork["wall"] = formwork.get("wall", 0.0) + 2.0 * length_m * height_m
        self._add_steel(steel, volume, "wall")

    def _masonry(self, el: Element, storey_h: float, storeys: int) -> float:
        length_m = (el.length_mm or 0.0) / 1000.0
        thick_mm = el.width_mm or float(self.est.get("wall_thickness_mm", 230))
        height_m = storey_h * max(1, storeys)
        return length_m * (thick_mm / 1000.0) * height_m

    def _footing(
        self,
        el: Element,
        concrete: dict[str, float],
        formwork: dict[str, float],
        steel: dict[float, float],
    ) -> None:
        area_m2 = (el.area_mm2 or 0.0) / MM2_PER_M2
        depth_mm = float(self.est.get("footing_depth_mm", 600))
        volume = area_m2 * depth_mm / 1000.0
        concrete["footing"] = concrete.get("footing", 0.0) + volume
        formwork["footing"] = formwork.get("footing", 0.0) + 2.0 * area_m2 * (depth_mm / 1000.0)
        self._add_steel(steel, volume, "footing")

    @staticmethod
    def _steel_from_schedule(items: list[RebarItem]) -> dict[float, float]:
        out: dict[float, float] = {}
        for item in items:
            out[item.diameter_mm] = out.get(item.diameter_mm, 0.0) + item.weight_kg
        return out

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _detect_storeys(self, analyses: list[PageAnalysis]) -> int:
        for page in analyses:
            text = page.text or ""
            for pattern in STOREY_PATTERNS:
                m = pattern.search(text)
                if m:
                    try:
                        value = int(m.group(1))
                    except (TypeError, ValueError):
                        continue
                    if 1 <= value <= 40:
                        if "STOREY" in pattern.pattern.upper() or "FLOOR" in pattern.pattern.upper():
                            return value
                        return value + 1  # G+n means n upper floors over the ground floor
        return 1

    @staticmethod
    def _confidence(
        analyses: list[PageAnalysis], schedule_items: list[RebarItem], counts: dict[str, int]
    ) -> float:
        if not analyses:
            return 0.0
        score = 0.2
        scales = [a.scale for a in analyses]
        if any(s.method in ("text", "manual") for s in scales):
            score += 0.25
        elif any(s.method == "bar" for s in scales):
            score += 0.2
        elif any(s.method == "dimension" for s in scales):
            score += 0.15
        if schedule_items:
            score += 0.2
        if counts.get("column", 0) or counts.get("beam", 0):
            score += 0.15
        if counts.get("slab", 0) or counts.get("wall", 0):
            score += 0.1
        return min(0.95, score)
