"""PDF handling: page rasterisation and native vector/text extraction.

The project deliberately prefers **native PDF parsing** (PyMuPDF) over an
external Poppler install: vector CAD exports carry exact geometry, real text
and title-block information, all of which make quantity take-off dramatically
more accurate than working on a flattened raster.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import Config, get_config

logger = logging.getLogger(__name__)

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24 module name

    HAVE_FITZ = True
except Exception:  # pragma: no cover
    try:
        import fitz as fitz  # older PyMuPDF

        HAVE_FITZ = True
    except Exception:
        fitz = None  # type: ignore
        HAVE_FITZ = False


@dataclass
class PdfPage:
    """A single page of a PDF, both rasterised and parsed."""

    index: int
    width_pt: float
    height_pt: float
    image_path: str | None = None
    zoom: float = 1.0
    words: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    drawings: list[dict[str, Any]] = field(default_factory=list)
    text_blocks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def width_mm(self) -> float:
        return self.width_pt * 25.4 / 72.0

    @property
    def height_mm(self) -> float:
        return self.height_pt * 25.4 / 72.0


class PDFProcessor:
    """Convert PDF drawings to high-resolution images and parse their content."""

    def __init__(self, config: Config | None = None, config_path: str | Path | None = None):
        self.cfg = config or (get_config() if config_path is None else load_config_safe(config_path))
        self.pdf_cfg = self.cfg.section("pdf_processing")
        self.prep_cfg = self.cfg.section("image_preprocessing")
        self.output_path = self.cfg.path("paths.processed_images")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.page_count = 0
        logger.info("PDFProcessor ready (PyMuPDF available=%s)", HAVE_FITZ)

    # ------------------------------------------------------------------ #
    # rasterisation
    # ------------------------------------------------------------------ #
    def render_pages(self, pdf_path: str | Path, prefix: str | None = None) -> list[PdfPage]:
        """Render every page to JPEG and parse its native content."""
        if not HAVE_FITZ:  # pragma: no cover
            raise RuntimeError("PyMuPDF is required. Install it with: pip install pymupdf")

        pdf_path = Path(pdf_path)
        prefix = prefix or pdf_path.stem
        doc = fitz.open(str(pdf_path))
        dpi = int(self.pdf_cfg.get("dpi", 300))
        base_zoom = dpi / 72.0
        max_dimension = int(self.pdf_cfg.get("max_render_dimension", 5200))
        quality = int(self.pdf_cfg.get("quality", 92))
        max_pages = int(self.pdf_cfg.get("max_pages", 60))

        pages: list[PdfPage] = []
        for idx in range(min(doc.page_count, max_pages)):
            page = doc.load_page(idx)
            rect = page.rect
            pdf_page = PdfPage(index=idx, width_pt=rect.width, height_pt=rect.height)

            # Keep the raster within a sane pixel budget - an A1 sheet at
            # 300 dpi is 9933 px tall and would blow up memory and time.
            zoom = base_zoom
            page_max_pt = max(rect.width, rect.height)
            if max_dimension and page_max_pt * zoom > max_dimension:
                zoom = max_dimension / page_max_pt
            pdf_page.zoom = zoom
            matrix = fitz.Matrix(zoom, zoom)

            # ---- native content -------------------------------------- #
            try:
                pdf_page.words = page.get_text("words") or []
            except Exception:
                pdf_page.words = []
            try:
                pdf_page.text = page.get_text("text") or ""
            except Exception:
                pdf_page.text = ""
            try:
                pdf_page.text_blocks = page.get_text("dict") or {}
                pdf_page.text_blocks = pdf_page.text_blocks.get("blocks", [])
            except Exception:
                pdf_page.text_blocks = []
            try:
                pdf_page.drawings = page.get_drawings() or []
            except Exception:
                pdf_page.drawings = []

            # ---- raster ------------------------------------------------ #
            if self.pdf_cfg.get("render_for_cv", True):
                try:
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    out_file = self.output_path / f"{prefix}_{idx:03d}.jpg"
                    pix.save(str(out_file), jpg_quality=quality)
                    pdf_page.image_path = str(out_file)
                except Exception as exc:
                    logger.warning("Rasterising page %d failed: %s", idx, exc)

            pages.append(pdf_page)

        self.page_count = len(pages)
        doc.close()
        logger.info("Parsed %d page(s) from %s", len(pages), pdf_path.name)
        return pages

    # ------------------------------------------------------------------ #
    # legacy compatibility
    # ------------------------------------------------------------------ #
    def process_pdf(self, pdf_path: str | Path, prefix: str | None = None) -> list[str]:
        """Render a PDF and return the list of written image paths."""
        pages = self.render_pages(pdf_path, prefix=prefix)
        return [p.image_path for p in pages if p.image_path]

    def extract_text(self, pdf_path: str | Path) -> str:
        if not HAVE_FITZ:  # pragma: no cover
            return ""
        doc = fitz.open(str(pdf_path))
        text = "\n".join(doc.load_page(i).get_text("text") or "" for i in range(doc.page_count))
        doc.close()
        return text

    def page_count(self, pdf_path: str | Path) -> int:
        if not HAVE_FITZ:  # pragma: no cover
            return 0
        doc = fitz.open(str(pdf_path))
        n = doc.page_count
        doc.close()
        return n


def load_config_safe(config_path: str | Path) -> Config:
    from .config import load_config

    return load_config(config_path)
