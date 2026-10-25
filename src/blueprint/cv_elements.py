"""Raster computer-vision pass.

This pass runs on the rendered page image and is the *only* source of
geometry for scanned drawings (where the PDF has no usable vector content).
It also acts as a cross-check for vector pages.

Detectors
---------
``detect_straight_members`` - morphological + Hough line extraction for
    beams and wall centrelines.
``detect_rectangles``       - contour analysis for columns, rooms and slabs.
``detect_circles``          - section / column / bar callout markers.
``detect_rebar_mats``       - dense clusters of closely spaced parallel lines,
    the typical way a slab reinforcement is drawn in plan.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import Element

logger = logging.getLogger(__name__)

INK = (255, 255, 255)  # BGR white for drawing ink on the binary mask


def load_gray(image_path: str | Path) -> np.ndarray | None:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return img


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask where ink (dark lines) is 255."""
    blur = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 12
    )


def _polar(angle: float) -> float:
    return angle + 180 if angle < 0 else angle


def detect_straight_members(
    gray: np.ndarray,
    denominator: float,
    zoom: float,
    min_length_mm: float = 1000.0,
) -> list[Element]:
    """Find long straight members (beams / wall centrelines) via Hough."""
    h, w = gray.shape[:2]
    mask = ink_mask(gray)
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)

    min_len_px = max(30, int((min_length_mm / denominator) * (72.0 / 25.4) * zoom))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=min_len_px, maxLineGap=12
    )
    elements: list[Element] = []
    if lines is None or len(lines) == 0:
        return elements

    # OpenCV returns (N, 1, 4) in most builds and (N, 4) in some
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        angle = _polar(math.degrees(math.atan2(dy, dx)))
        horizontal = angle < 12 or angle > 168
        vertical = 78 < angle < 102
        if not (horizontal or vertical):
            continue
        length_mm = (length / zoom) * (72.0 / 25.4) * denominator
        elements.append(
            Element(
                kind="beam",
                source="cv",
                confidence=0.4,
                x1=x1 / zoom,
                y1=y1 / zoom,
                x2=x2 / zoom,
                y2=y2 / zoom,
                length_mm=length_mm,
                metadata={"cv_angle": round(angle, 1), "direction": "h" if horizontal else "v"},
            )
        )
    return elements


def detect_rectangles(
    gray: np.ndarray,
    denominator: float,
    zoom: float,
    min_column_mm: float = 120.0,
    max_column_mm: float = 1400.0,
    min_zone_mm: float = 1500.0,
) -> list[Element]:
    """Contour based detection of columns, zones and wall blocks."""
    h, w = gray.shape[:2]
    mask = ink_mask(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    elements: list[Element] = []
    page_area = float(w * h)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 12:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 4 or bh < 4:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        rectangularity = len(approx) in (4, 8, 9, 10)

        w_mm = (bw / zoom) * (72.0 / 25.4) * denominator
        h_mm = (bh / zoom) * (72.0 / 25.4) * denominator
        shorter, longer = min(w_mm, h_mm), max(w_mm, h_mm)
        fill = area / float(max(1, bw * bh))

        if shorter < min_column_mm or longer > max_column_mm:
            continue
        if longer / max(shorter, 1e-6) > 3.0:
            continue
        if fill < 0.35 and not rectangularity:
            continue
        if area / page_area > 0.08:
            continue

        elements.append(
            Element(
                kind="column",
                source="cv",
                confidence=0.45,
                x1=x / zoom,
                y1=y / zoom,
                x2=(x + bw) / zoom,
                y2=(y + bh) / zoom,
                length_mm=w_mm,
                width_mm=w_mm,
                depth_mm=h_mm,
                area_mm2=w_mm * h_mm,
                perimeter_mm=2.0 * (w_mm + h_mm),
                metadata={"cv_fill": round(fill, 2)},
            )
        )
    return elements


def detect_circles(
    gray: np.ndarray,
    denominator: float,
    zoom: float,
    min_dia_mm: float = 300.0,
    max_dia_mm: float = 3000.0,
) -> list[Element]:
    """Find circular callouts (column heads, bar sections, detail markers)."""
    h, w = gray.shape[:2]
    gray = cv2.medianBlur(gray, 3)
    min_r = max(4, int((min_dia_mm / denominator) * (72.0 / 25.4) * zoom / 2))
    max_r = max(min_r + 3, int((max_dia_mm / denominator) * (72.0 / 25.4) * zoom / 2))
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(12, min_r * 2),
        param1=120, param2=42, minRadius=min_r, maxRadius=max_r,
    )
    elements: list[Element] = []
    if circles is None:
        return elements
    for cx, cy, r in np.round(circles[0]).astype(int):
        dia_mm = (2 * r / zoom) * (72.0 / 25.4) * denominator
        elements.append(
            Element(
                kind="column",
                source="cv",
                confidence=0.35,
                x1=(cx - r) / zoom,
                y1=(cy - r) / zoom,
                x2=(cx + r) / zoom,
                y2=(cy + r) / zoom,
                length_mm=dia_mm,
                width_mm=dia_mm,
                depth_mm=dia_mm,
                area_mm2=math.pi * (dia_mm / 2) ** 2,
                perimeter_mm=math.pi * dia_mm,
                metadata={"cv_shape": "circle"},
            )
        )
    return elements


def detect_rebar_mats(
    gray: np.ndarray,
    denominator: float,
    zoom: float,
    min_spacing_mm: float = 60.0,
    max_spacing_mm: float = 250.0,
    min_lines: int = 4,
) -> list[Element]:
    """Find reinforcement mats drawn as closely spaced parallel lines."""
    h, w = gray.shape[:2]
    mask = ink_mask(gray)
    elements: list[Element] = []

    for horizontal in (True, False):
        if horizontal:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(w * 0.02)), 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(h * 0.02))))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        row_projection = opened.sum(axis=1 if horizontal else 0)
        rows = np.where(row_projection > (0.02 * (w if horizontal else h)))[0]
        if len(rows) < min_lines:
            continue

        # group contiguous detected rows into bands
        bands: list[list[int]] = []
        current = [rows[0]]
        for r in rows[1:]:
            if r - current[-1] <= 2:
                current.append(r)
            else:
                bands.append(current)
                current = [r]
        bands.append(current)

        for band in bands:
            if len(band) < min_lines:
                continue
            spacing_mm = ((band[-1] - band[0]) / max(1, len(band) - 1) / zoom) * (72.0 / 25.4) * denominator
            if not (min_spacing_mm <= spacing_mm <= max_spacing_mm):
                continue
            span_mm = ((band[-1] - band[0]) / zoom) * (72.0 / 25.4) * denominator
            if span_mm < 600:
                continue
            if horizontal:
                y = band[0]
                x1, y1, x2, y2 = 0.0, y / zoom, 0.0, (band[-1]) / zoom
            else:
                x = band[0]
                x1, y1, x2, y2 = x / zoom, 0.0, (band[-1]) / zoom, 0.0
            elements.append(
                Element(
                    kind="rebar",
                    source="cv",
                    confidence=0.35,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    length_mm=span_mm,
                    count=len(band),
                    spacing_mm=spacing_mm,
                    metadata={"cv_direction": "h" if horizontal else "v", "cv_lines": len(band)},
                )
            )
    return elements


def analyse_image(
    image_path: str | Path,
    denominator: float,
    zoom: float,
    *,
    max_dimension: int = 2200,
    max_elements: int = 600,
    run_members: bool = True,
    run_rects: bool = True,
    run_circles: bool = True,
    run_mats: bool = True,
) -> list[Element]:
    """Run the whole CV suite over a single rendered page.

    The image is downscaled to ``max_dimension`` pixels first - the vision
    detectors work on structure, not on fine detail, and the full size raster
    is both slow and memory hungry on large sheets.

    ``max_elements`` is a sanity bound. On a dense detail sheet the contour
    pass fires on every glyph and symbol, yielding thousands of "columns"; an
    A1 plan cannot contain that many, so the whole page's CV result is treated
    as noise and discarded rather than silently priced.
    """
    gray = load_gray(image_path)
    if gray is None:
        logger.warning("Could not read image %s", image_path)
        return []

    h, w = gray.shape[:2]
    longest = max(h, w)
    if max_dimension and longest > max_dimension:
        factor = max_dimension / float(longest)
        gray = cv2.resize(gray, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
        zoom = zoom * factor

    out: list[Element] = []
    if run_members:
        out.extend(detect_straight_members(gray, denominator, zoom))
    if run_rects:
        out.extend(detect_rectangles(gray, denominator, zoom))
    if run_circles:
        out.extend(detect_circles(gray, denominator, zoom))
    if run_mats:
        out.extend(detect_rebar_mats(gray, denominator, zoom))
    if max_elements and len(out) > max_elements:
        logger.warning(
            "CV pass on %s produced %d candidates (limit %d) - discarded as implausible",
            Path(image_path).name,
            len(out),
            max_elements,
        )
        return []
    logger.info("CV pass on %s -> %d candidate elements", Path(image_path).name, len(out))
    return out
