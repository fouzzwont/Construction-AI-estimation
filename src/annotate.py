"""Annotation overlay: draw detected elements onto the rendered page images."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .blueprint.models import Element

logger = logging.getLogger(__name__)

KIND_COLORS: dict[str, tuple[int, int, int]] = {
    "column": (214, 39, 40),      # red
    "beam": (44, 160, 44),        # green
    "slab": (0, 165, 255),        # amber
    "wall": (219, 39, 220),       # magenta
    "footing": (0, 140, 255),     # orange
    "rebar": (30, 30, 220),       # dark red
}
LEGEND_ORDER = ["rebar", "column", "beam", "slab", "wall", "footing"]


def _px(value: float, zoom: float, minimum: int = 1) -> int:
    """Scale a pixel measurement by the render zoom, never below *minimum*."""
    return max(minimum, int(round(value * zoom)))


def _scale_zoom(dpi: int) -> float:
    return dpi / 72.0


def annotate_page(
    image_path: str | Path,
    elements: list[Element],
    output_path: str | Path,
    zoom: float = 1.0,
    min_confidence: float = 0.0,
) -> str | None:
    """Draw the detected elements on a copy of the page image."""
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("annotate_page: cannot read %s", image_path)
        return None

    thickness = _px(2, zoom, 2)
    font_scale = max(0.4, 0.5 * zoom)
    counts: dict[str, int] = {}

    for el in elements:
        if el.confidence < min_confidence:
            continue
        bbox = el.bbox
        if bbox is None:
            continue
        x1, y1, x2, y2 = (
            int(bbox[0] * zoom),
            int(bbox[1] * zoom),
            int(bbox[2] * zoom),
            int(bbox[3] * zoom),
        )
        color = KIND_COLORS.get(el.kind, (0, 200, 200))
        counts[el.kind] = counts.get(el.kind, 0) + 1

        if el.kind in ("slab", "wall"):
            # translucent fill so the plan underneath stays readable
            alpha = 0.16 if el.kind == "slab" else 0.14
            layer = image.copy()
            cv2.rectangle(layer, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(layer, alpha, image, 1 - alpha, 0, image)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        label = el.kind
        if el.length_mm and el.length_mm > 0:
            label = f"{el.kind} {el.length_mm / 1000:.2f}m"
        elif el.area_mm2:
            label = f"{el.kind} {el.area_mm2 / 1e6:.1f}m2"
        cv2.putText(
            image,
            label,
            (x1 + _px(3, zoom), max(_px(14, zoom), y1 - _px(5, zoom))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, _px(1, zoom)),
            cv2.LINE_AA,
        )

    draw_legend(image, counts, zoom=zoom)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return str(output_path)


def draw_legend(image: np.ndarray, counts: dict[str, int], zoom: float = 1.0) -> np.ndarray:
    """Render a legend box in the top-left corner."""
    entries = [(k, counts.get(k, 0)) for k in LEGEND_ORDER if counts.get(k, 0)]
    if not entries:
        return image

    pad = _px(8, zoom)
    row = _px(22, zoom)
    h = _px(26, zoom) + row * len(entries)
    w = _px(250, zoom)
    panel = image.copy()
    cv2.rectangle(panel, (pad, pad), (pad + w, pad + h), (255, 255, 255), -1)
    cv2.addWeighted(panel, 0.82, image, 0.18, 0, image)
    cv2.putText(
        image,
        "DETECTED ELEMENTS",
        (pad + _px(8, zoom), pad + _px(18, zoom)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.4, 0.5 * zoom),
        (30, 30, 30),
        max(1, _px(1, zoom)),
        cv2.LINE_AA,
    )
    y = pad + _px(40, zoom)
    for kind, count in entries:
        cv2.rectangle(
            image,
            (pad + _px(8, zoom), y - _px(9, zoom)),
            (pad + _px(20, zoom), y + _px(3, zoom)),
            KIND_COLORS.get(kind, (0, 200, 200)),
            -1,
        )
        cv2.putText(
            image,
            f"{kind}: {count}",
            (pad + _px(28, zoom), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.4, 0.45 * zoom),
            (30, 30, 30),
            max(1, _px(1, zoom)),
            cv2.LINE_AA,
        )
        y += row
    return image
