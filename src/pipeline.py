"""End-to-end orchestration: PDF in, priced estimate + reports out."""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .annotate import annotate_page
from .blueprint.detector import BlueprintDetector
from .blueprint.models import ScaleInfo
from .config import Config, get_config
from .dataset import MANIFEST_NAME, build_manifest, write_manifest
from .estimator import EstimateResult, Estimator
from .pdf_processor import PDFProcessor
from .rates import load_rate_card
from .report import write_csv, write_pdf

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


@dataclass
class PipelineOutput:
    job_id: str
    result: EstimateResult
    pages: list[dict[str, Any]]
    csv_path: str | None
    pdf_path: str | None
    elapsed: float
    manifest_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "result": self.result.to_dict(),
            "pages": self.pages,
            "csv_path": self.csv_path,
            "pdf_path": self.pdf_path,
            "manifest_path": self.manifest_path,
            "elapsed_seconds": round(self.elapsed, 2),
        }


class EstimationPipeline:
    """Runs the full detection + estimation + reporting chain."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or get_config()
        self.cfg.ensure_dirs()
        self.pdf_processor = PDFProcessor(self.cfg)
        self.detector = BlueprintDetector(self.cfg)

    # ------------------------------------------------------------------ #
    def run(
        self,
        pdf_path: str | Path,
        *,
        job_id: str | None = None,
        upload_path: str | Path | None = None,
        consent: bool = False,
        scale_override: float | None = None,
        market: str | None = None,
        project_name: str = "",
        progress: ProgressFn | None = None,
        make_reports: bool = True,
    ) -> PipelineOutput:
        started = time.time()
        # Callers that already minted an id (the web API) pass it in, so the
        # upload, the output folder and the manifest all share one handle.
        job_id = job_id or uuid.uuid4().hex[:12]
        pdf_path = Path(pdf_path)

        def report(stage: str, pct: float) -> None:
            if progress:
                progress(stage, pct)

        report("Parsing drawing", 0.05)
        # Web uploads are stored as "<job_id>_<name>.pdf", so prefixing with the
        # id again would yield "<id>_<id>_<name>_000.jpg". Strip the copy first.
        stem = pdf_path.stem
        if stem.startswith(f"{job_id}_"):
            stem = stem[len(job_id) + 1 :]
        pages = self.pdf_processor.render_pages(pdf_path, prefix=f"{job_id}_{stem}")
        if not pages:
            raise RuntimeError("The PDF could not be read (no pages found).")

        report("Detecting scale and elements", 0.3)
        analyses = self.detector.analyse_document(pages)

        if scale_override:
            denom = float(scale_override)
            for a in analyses:
                a.scale = ScaleInfo(denom, "manual", 1.0, "supplied by user")
            logger.info("Scale overridden by user: 1:%s", denom)

        report("Estimating quantities", 0.65)
        rate_card = load_rate_card(self.cfg, market)
        estimator = Estimator(self.cfg, rate_card)
        result = estimator.estimate(
            analyses,
            source_file=pdf_path.name,
            project_name=project_name or pdf_path.stem,
        )

        report("Annotating drawings", 0.8)
        out_dir = self.cfg.path("paths.estimates") / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        page_payloads: list[dict[str, Any]] = []
        annotated_paths: list[str] = []
        page_lookup = {p.index: p for p in pages}
        for analysis in analyses:
            payload = analysis.to_dict()
            raster = page_lookup.get(analysis.index)
            if raster is not None:
                payload["px_width"] = raster.px_width
                payload["px_height"] = raster.px_height
                payload["image_name"] = Path(raster.image_path).name if raster.image_path else None
            annotated_path = None
            if analysis.image_path:
                target = out_dir / f"page_{analysis.index + 1:03d}_annotated.jpg"
                annotated_path = annotate_page(
                    analysis.image_path, analysis.elements, target, zoom=analysis.zoom or 1.0
                )
                if annotated_path:
                    annotated_paths.append(annotated_path)
                    payload["annotated_path"] = annotated_path
                    payload["counts"] = self._counts(analysis.elements)
            page_payloads.append(payload)

        csv_path = None
        pdf_path_out = None
        if make_reports:
            report("Building reports", 0.9)
            csv_path = write_csv(result, out_dir / f"{job_id}_estimate.csv")
            pdf_path_out = write_pdf(result, out_dir / f"{job_id}_estimate.pdf", annotated_paths)

        report("Done", 1.0)
        elapsed = time.time() - started

        payload = out_dir / "result.json"
        payload.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

        # The manifest is the durable half of the job: it records the source
        # PDF, its hash, and every detected box, so the corpus stays usable
        # after _jobs (memory only) is gone.
        manifest_path = None
        try:
            manifest = build_manifest(
                job_id=job_id,
                pdf_path=pdf_path,
                upload_path=upload_path,
                pages=pages,
                analyses=analyses,
                estimate=result,
                consent=consent,
                app_version=str(self.cfg.get("project.version", "2.0.0")),
            )
            manifest_path = write_manifest(out_dir / MANIFEST_NAME, manifest)
        except Exception as exc:  # never fail an estimate over bookkeeping
            logger.warning("Could not write manifest for %s: %s", job_id, exc)

        logger.info("Job %s finished in %.2fs", job_id, elapsed)
        return PipelineOutput(
            job_id=job_id,
            result=result,
            pages=page_payloads,
            csv_path=csv_path,
            pdf_path=pdf_path_out,
            elapsed=elapsed,
            manifest_path=manifest_path,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _counts(elements) -> dict[str, int]:
        counts: dict[str, int] = {}
        for el in elements:
            counts[el.kind] = counts.get(el.kind, 0) + 1
        return counts
