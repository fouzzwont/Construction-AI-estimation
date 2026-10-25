"""Hybrid detection engine.

For every page the engine combines four independent information sources:

============  ==========================================================
source        what it contributes
============  ==========================================================
PDF vector    exact member geometry (walls, columns, zones, beams)
PDF text      drawing scale, dimensions, reinforcement schedule, notes
Raster CV     fallback / cross-check for scanned or partially raster pages
YOLOv8        custom trained detector (used when weights are available)
============  ==========================================================

Results are merged, de-duplicated and returned per page so the estimator can
work on a single, coherent element list.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from ..config import Config, get_config
from ..pdf_processor import PdfPage
from . import cv_elements, scale as scale_mod, schedule as schedule_mod
from .elements import classify_geometry, merge_slabs
from .models import Element, PageAnalysis, RebarItem, ScaleInfo
from .vector import extract_geometry, scale_denominator_of

logger = logging.getLogger(__name__)

ZOOOM_FROM_DPI = lambda dpi: dpi / 72.0  # noqa: E731


def dpi_zoom(cfg: Config) -> float:
    return ZOOOM_FROM_DPI(int(cfg.get("pdf_processing.dpi", 300)))


class BlueprintDetector:
    """Analyse a set of PDF pages into structured elements."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or get_config()
        self.est_cfg = self.cfg.section("estimation")
        self.inf_cfg = self.cfg.section("inference")
        self.scale_cfg = self.cfg.section("scale")
        self.yolo_cfg = self.cfg.section("yolo")
        self.labels = self.cfg.labels
        self._yolo = None
        self._yolo_loaded = False

    # ------------------------------------------------------------------ #
    # optional YOLO model
    # ------------------------------------------------------------------ #
    @property
    def yolo_available(self) -> bool:
        if not self.yolo_cfg.get("enabled", True):
            return False
        return self._load_yolo() is not None

    def _load_yolo(self):
        if self._yolo_loaded:
            return self._yolo
        self._yolo_loaded = True
        weights = self.cfg.path("yolo.weights_path")
        if not weights.exists():
            logger.info("No YOLO weights at %s - using geometry + CV only", weights)
            self._yolo = None
            return None
        try:
            from ultralytics import YOLO

            self._yolo = YOLO(str(weights))
            logger.info("Loaded YOLO weights: %s", weights)
        except Exception as exc:
            logger.warning("Could not load YOLO weights (%s) - continuing without", exc)
            self._yolo = None
        return self._yolo

    # ------------------------------------------------------------------ #
    # document level
    # ------------------------------------------------------------------ #
    def analyse_document(self, pages: list[PdfPage]) -> list[PageAnalysis]:
        """Analyse every page, then settle on one document-wide scale."""
        analyses: list[PageAnalysis] = []
        default_scale = float(self.scale_cfg.get("default", 100))

        for page in pages:
            analysis = self._analyse_page(page, page.zoom or (dpi_zoom(self.cfg)), default_scale)
            analyses.append(analysis)
            # Free the heavy per-page structures as we go: a 20-page set keeps
            # ~500 MB alive otherwise, which is an OOM kill on a 512 MB host.
            page.release_content()

        if analyses:
            doc_scale = scale_mod.consolidate_scales([a.scale for a in analyses])
            for a in analyses:
                if a.scale.method in ("default",) and doc_scale.method != "default":
                    a.scale = doc_scale
                elif a.scale.method == "dimension" and doc_scale.method == "text":
                    a.scale = doc_scale
        return analyses

    # ------------------------------------------------------------------ #
    # page level
    # ------------------------------------------------------------------ #
    def _analyse_page(self, page: PdfPage, zoom: float, default_scale: float) -> PageAnalysis:
        geometry = extract_geometry(page.drawings)
        lines = schedule_mod.extract_text_lines(page.text_blocks)

        # ---- scale ---------------------------------------------------- #
        page_scale = scale_mod.detect_scale(
            page.text, page.words, geometry, default=default_scale
        )
        denominator = scale_denominator_of(page_scale.denominator)

        analysis = PageAnalysis(
            index=page.index,
            image_path=page.image_path,
            width_pt=page.width_pt,
            height_pt=page.height_pt,
            zoom=zoom,
            scale=ScaleInfo(denominator, page_scale.method, page_scale.confidence, page_scale.evidence),
            text=page.text,
        )

        # ---- reinforcement schedule ----------------------------------- #
        items, sched_notes = schedule_mod.parse_rebar_schedule(
            lines, page=page.index, page_height=page.height_pt
        )
        analysis.rebar_items.extend(items)
        analysis.notes.extend(sched_notes)

        # ---- not to scale --------------------------------------------- #
        # A detail sheet marked NTS has no usable scale, so any geometry taken
        # from it would be priced against an invented size. The schedule above
        # is still valid because it is read from text, not from lengths.
        if page_scale.method == "nts":
            analysis.notes.append(
                "sheet is marked NTS (not to scale) - no quantities taken from it"
            )
            return analysis

        # ---- non-plan sheets ------------------------------------------ #
        if bool(self.cfg.get("pdf_processing.plan_sheets_only", True)) and scale_mod.is_non_plan_sheet(
            page.text
        ):
            analysis.notes.append(
                "sheet is an elevation/section/detail - no plan quantities taken from it"
            )
            return analysis

        # ---- vector geometry ------------------------------------------ #
        elements: list[Element] = []
        vector_rich = len(geometry.segments) + len(geometry.rects) > 8
        if vector_rich:
            elements.extend(
                classify_geometry(
                    geometry,
                    denominator,
                    page.words,
                    wall_thickness_mm=float(self.est_cfg.get("wall_thickness_mm", 230)),
                )
            )
        else:
            analysis.notes.append("page has little vector content - treated as raster")

        # ---- raster CV ------------------------------------------------ #
        cv_mode = str(self.cfg.get("pdf_processing.cv_mode", "auto")).lower()
        scanned = not vector_rich
        cv_max_dim = int(self.cfg.get("pdf_processing.cv_max_dimension", 2200))

        if page.image_path and cv_mode != "off":
            run_cv = scanned or cv_mode == "always"
            if run_cv:
                cv_elements_found = cv_elements.analyse_image(
                    page.image_path,
                    denominator,
                    zoom,
                    max_dimension=cv_max_dim,
                    max_elements=int(self.cfg.get("pdf_processing.cv_max_elements", 600)),
                )
                if scanned:
                    elements.extend(cv_elements_found)
                    analysis.notes.append(f"CV pass contributed {len(cv_elements_found)} elements")
                else:
                    elements = self._merge_cv(elements, cv_elements_found, denominator)
            elif cv_mode == "auto":
                analysis.notes.append("vision pass skipped (vector geometry is authoritative)")

        # ---- YOLO ------------------------------------------------------ #
        if page.image_path:
            yolo_elements = self._run_yolo(page.image_path)
            elements.extend(yolo_elements)

        for el in elements:
            el.page = page.index

        analysis.elements = merge_slabs(self._dedupe(elements))
        analysis.notes.append(f"{len(analysis.elements)} elements detected on page {page.index + 1}")
        return analysis

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _merge_cv(
        self, vector_elements: list[Element], cv_found: list[Element], denominator: float
    ) -> list[Element]:
        """Add CV findings that do not duplicate a vector detection."""
        merged = list(vector_elements)
        for cand in cv_found:
            dup = False
            for existing in vector_elements:
                if existing.kind != cand.kind:
                    continue
                if self._bbox_close(existing, cand):
                    dup = True
                    break
            if not dup:
                merged.append(cand)
        return merged

    @staticmethod
    def _bbox_close(a: Element, b: Element, tol: float = 12.0) -> bool:
        ba, bb = a.bbox, b.bbox
        if not ba or not bb:
            return False
        ix = min(ba[2], bb[2]) - max(ba[0], bb[0])
        iy = min(ba[3], bb[3]) - max(ba[1], bb[1])
        if ix < 0 or iy < 0:
            return False
        inter = ix * iy
        area_a = (ba[2] - ba[0]) * (ba[3] - ba[1])
        area_b = (bb[2] - bb[0]) * (bb[3] - bb[1])
        union = area_a + area_b - inter
        return union > 0 and inter / union > 0.55

    @staticmethod
    def _dedupe(elements: list[Element]) -> list[Element]:
        """Remove exact duplicates (same kind, same geometry)."""
        seen: set[tuple[Any, ...]] = set()
        out: list[Element] = []
        for el in elements:
            b = el.bbox
            key = (
                el.kind,
                round(b[0], 1) if b else None,
                round(b[1], 1) if b else None,
                round(b[2], 1) if b else None,
                round(b[3], 1) if b else None,
                round(el.length_mm or 0.0, 1),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(el)
        return out

    def _run_yolo(self, image_path: str) -> list[Element]:
        model = self._load_yolo()
        if model is None:
            return []
        try:
            results = model.predict(
                source=str(image_path),
                conf=float(self.inf_cfg.get("conf_threshold", 0.35)),
                iou=float(self.inf_cfg.get("iou_threshold", 0.45)),
                device=self.cfg.device,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("YOLO inference failed: %s", exc)
            return []

        elements: list[Element] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            names = getattr(res, "names", {}) or {}
            for box in boxes:
                cls_id = int(box.cls[0]) if len(box.cls) else 0
                conf = float(box.conf[0]) if len(box.conf) else 0.0
                xy = box.xyxy[0].tolist()
                name = names.get(cls_id, self.labels[cls_id] if cls_id < len(self.labels) else "unknown")
                kind = self._map_label(str(name))
                elements.append(
                    Element(
                        kind=kind,
                        source="yolo",
                        confidence=conf,
                        x1=xy[0],
                        y1=xy[1],
                        x2=xy[2],
                        y2=xy[3],
                        metadata={"yolo_class": str(name), "yolo_conf": round(conf, 3)},
                    )
                )
        return elements

    @staticmethod
    def _map_label(name: str) -> str:
        n = name.strip().lower()
        aliases = {
            "rebar": "rebar", "bar": "rebar", "reinforcement": "rebar", "steel": "rebar",
            "beam": "beam", "beam-column": "beam",
            "slab": "slab", "floor": "slab",
            "column": "column", "col": "column", "pillar": "column",
            "wall": "wall", "brickwork": "wall", "masonry": "wall",
            "footing": "footing", "foundation": "footing", "pile": "footing",
        }
        return aliases.get(n, n if n in ("rebar", "beam", "slab", "column", "wall", "footing") else "column")
