"""Training-data plumbing: job manifests, YOLO export and retention.

The estimation pipeline is a *producer* of labels: every detected element
carries a class name and a bounding box in PDF points. This module turns that
into two artefacts:

``output/estimates/<job_id>/manifest.json``
    A self-contained record of one job - which PDF it came from, its hash,
    every page raster and every detected element. Without this the link
    between an upload and its analysis existed only in memory.

``data/dataset/``
    A YOLO-ready corpus (``images/`` + ``labels/`` + ``dataset.yaml``),
    optionally cut into tiles so thin elements survive at training scale.

Nothing here imports ultralytics/torch, so export works on a machine that
cannot train.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .config import Config, get_config

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1
JPEG_QUALITY = 92


# --------------------------------------------------------------------------- #
# hashing / io helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_manifests(estimates_dir: str | Path) -> Iterator[Path]:
    """Yield every manifest under ``output/estimates/``, oldest job first."""
    root = Path(estimates_dir)
    if not root.is_dir():
        return
    found = sorted(root.glob(f"*/{MANIFEST_NAME}"))
    for path in found:
        yield path


# --------------------------------------------------------------------------- #
# manifest construction (called by the pipeline)
# --------------------------------------------------------------------------- #
def build_manifest(
    *,
    job_id: str,
    pdf_path: str | Path,
    upload_path: str | Path | None,
    pages: Sequence[Any],
    analyses: Sequence[Any],
    estimate: Any,
    consent: bool = False,
    app_version: str = "2.0.0",
) -> dict[str, Any]:
    """Assemble the manifest for one finished job.

    ``pages`` are :class:`~src.pdf_processor.PdfPage` objects and ``analyses``
    :class:`~src.blueprint.models.PageAnalysis`; only the fields the dataset
    needs are copied so the manifest stays readable.
    """
    pdf_path = Path(pdf_path)
    page_lookup = {p.index: p for p in pages}

    page_records: list[dict[str, Any]] = []
    for analysis in analyses:
        src = page_lookup.get(analysis.index)
        px_w = int(getattr(src, "px_width", 0) or 0)
        px_h = int(getattr(src, "px_height", 0) or 0)
        width_pt = float(analysis.width_pt or 0.0)
        height_pt = float(analysis.height_pt or 0.0)

        elements: list[dict[str, Any]] = []
        for el in analysis.elements:
            box = el.bbox
            if box is None:
                continue
            elements.append(
                {
                    "kind": el.kind,
                    "source": el.source,
                    "confidence": round(float(el.confidence), 3),
                    "bbox_pt": [round(float(v), 3) for v in box],
                    "length_mm": _f(el.length_mm),
                    "width_mm": _f(el.width_mm),
                    "depth_mm": _f(el.depth_mm),
                }
            )

        image_name = Path(src.image_path).name if (src and src.image_path) else None
        # analysis.zoom and the raster size can disagree in the last decimal
        # because the pixmap rounds; the exporter uses px_width/width_pt.
        zoom = float(getattr(analysis, "zoom", 0.0) or (src.zoom if src else 1.0))
        page_records.append(
            {
                "index": analysis.index,
                "image_name": image_name,
                "width_pt": round(width_pt, 3),
                "height_pt": round(height_pt, 3),
                "px_width": px_w,
                "px_height": px_h,
                "zoom": round(zoom, 5),
                "scale": analysis.scale.to_dict(),
                "element_count": len(elements),
                "elements": elements,
            }
        )

    source = {
        "filename": pdf_path.name,
        "sha256": sha256_file(pdf_path) if pdf_path.is_file() else "",
        "bytes": pdf_path.stat().st_size if pdf_path.is_file() else 0,
    }
    if upload_path:
        source["upload_path"] = str(upload_path)
        source["upload_name"] = Path(upload_path).name

    return {
        "schema": SCHEMA_VERSION,
        "job_id": job_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "app_version": app_version,
        "consent": bool(consent),
        "source": source,
        "project_name": getattr(estimate, "project_name", ""),
        "market": getattr(estimate, "market", ""),
        "estimate": {
            "counts": getattr(estimate, "counts", {}),
            "totals": getattr(estimate, "totals", {}),
            "grand_total": getattr(estimate, "grand_total", 0.0),
            "confidence": getattr(estimate, "confidence", 0.0),
        },
        "page_count": len(page_records),
        "pages": page_records,
    }


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(target)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def tile_windows(width: int, height: int, size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Cover a ``width`` x ``height`` raster with ``size`` px windows.

    ``size <= 0`` (or an image that already fits) yields a single window.
    Edge windows are pushed back inside the image so nothing is padded.
    """
    if size <= 0 or (width <= size and height <= size):
        return [(0, 0, width, height)]

    step = max(1, size - max(0, overlap))

    def offsets(total: int) -> list[int]:
        starts = list(range(0, max(total - size, 0) + 1, step))
        if not starts:
            starts = [0]
        if starts[-1] < total - size:
            starts.append(max(total - size, 0))
        return starts

    windows: list[tuple[int, int, int, int]] = []
    for y in offsets(height):
        for x in offsets(width):
            windows.append((x, y, min(x + size, width), min(y + size, height)))
    return windows


def clip_box(
    box: Sequence[float],
    window: Sequence[int],
    min_visible: float,
) -> tuple[float, float, float, float] | None:
    """Clip ``box`` to ``window``; drop it if too little survives."""
    bx1, by1, bx2, by2 = box
    wx1, wy1, wx2, wy2 = window
    ix1, iy1 = max(bx1, wx1), max(by1, wy1)
    ix2, iy2 = min(bx2, wx2), min(by2, wy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    original = (bx2 - bx1) * (by2 - by1)
    if original <= 0:
        return None
    if ((ix2 - ix1) * (iy2 - iy1)) / original < min_visible:
        return None
    return (ix1, iy1, ix2, iy2)


def yolo_line(class_id: int, box: Sequence[float], tile_w: int, tile_h: int) -> str:
    """``cls cx cy w h``, normalised to the tile and clamped to [0, 1]."""
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / tile_w
    cy = ((y1 + y2) / 2.0) / tile_h
    bw = (x2 - x1) / tile_w
    bh = (y2 - y1) / tile_h
    clamp = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    return f"{class_id} {clamp(cx):.6f} {clamp(cy):.6f} {clamp(bw):.6f} {clamp(bh):.6f}"


def thicken_box(
    box: Sequence[float],
    min_thickness: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Pad a degenerate box out to ``min_thickness`` px on its thin axis.

    Beams and grid lines are stored with zero extent perpendicular to their
    run (``y1 == y2``), because that is how they are drawn - as single lines.
    The annotation overlay hides this by stroking every rectangle with a
    minimum line width, but a zero-area box is not a trainable label, so the
    raster label has to invent the thickness the artist implied.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    half = min_thickness / 2.0

    if x2 - x1 < min_thickness:
        cx = (x1 + x2) / 2.0
        x1, x2 = cx - half, cx + half
    if y2 - y1 < min_thickness:
        cy = (y1 + y2) / 2.0
        y1, y2 = cy - half, cy + half

    # Slide back inside the raster instead of squashing the box.
    if x1 < 0:
        x2 -= x1
        x1 = 0.0
    if y1 < 0:
        y2 -= y1
        y1 = 0.0
    if x2 > width:
        x1 -= x2 - width
        x2 = float(width)
    if y2 > height:
        y1 -= y2 - height
        y2 = float(height)

    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    return (x1, y1, x2, y2)


def assign_split(job_id: str, ratios: dict[str, float]) -> str:
    """Deterministic per-job split.

    Splitting by *job* rather than by image keeps every tile of a drawing on
    one side of the fence - otherwise near-identical tiles leak between train
    and val and the metrics look far better than the model is.
    """
    digest = hashlib.sha1(job_id.encode("utf-8")).hexdigest()[:8]
    bucket = int(digest, 16) % 100
    train = float(ratios.get("train", 70))
    val = float(ratios.get("val", 15))
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
@dataclass
class ExportReport:
    manifests_found: int = 0
    jobs_used: int = 0
    jobs_skipped_no_consent: int = 0
    jobs_skipped_unreadable: int = 0
    images_written: dict[str, int] = field(default_factory=dict)
    boxes_written: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    split_jobs: dict[str, int] = field(default_factory=dict)
    rebalanced: list[str] = field(default_factory=list)
    background_tiles: int = 0
    boxes_dropped: int = 0
    boxes_thickened: int = 0
    kinds_unknown: dict[str, int] = field(default_factory=dict)
    dataset_yaml: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifests_found": self.manifests_found,
            "jobs_used": self.jobs_used,
            "jobs_skipped_no_consent": self.jobs_skipped_no_consent,
            "jobs_skipped_unreadable": self.jobs_skipped_unreadable,
            "images_written": self.images_written,
            "boxes_written": self.boxes_written,
            "class_counts": self.class_counts,
            "split_jobs": self.split_jobs,
            "rebalanced": self.rebalanced,
            "background_tiles": self.background_tiles,
            "boxes_dropped": self.boxes_dropped,
            "boxes_thickened": self.boxes_thickened,
            "kinds_unknown": self.kinds_unknown,
            "dataset_yaml": self.dataset_yaml,
        }

    def summary(self) -> str:
        img = ", ".join(f"{k}={v}" for k, v in sorted(self.images_written.items())) or "none"
        box = ", ".join(f"{k}={v}" for k, v in sorted(self.boxes_written.items())) or "none"
        lines = [
            f"manifests found      : {self.manifests_found}",
            f"jobs exported        : {self.jobs_used}",
            f"tiles written        : {img}",
            f"boxes written        : {box}",
            f"boxes clipped away   : {self.boxes_dropped}",
        ]
        if self.background_tiles:
            lines.append(f"background tiles     : {self.background_tiles} (no labels, used as negatives)")
        if self.boxes_thickened:
            lines.append(
                f"degenerate boxes     : {self.boxes_thickened} zero-width/height "
                "(single-line beams), padded to a minimum thickness"
            )
        if self.class_counts:
            per = ", ".join(f"{k}={v}" for k, v in sorted(self.class_counts.items()))
            lines.append(f"instances per class  : {per}")
            missing = [k for k, v in self.class_counts.items() if v == 0]
            if missing:
                lines.append(f"WARNING empty classes: {', '.join(missing)}")
        if self.split_jobs:
            per_split = ", ".join(f"{k}={v}" for k, v in self.split_jobs.items())
            lines.append(f"jobs per split       : {per_split}")
            if self.split_jobs.get("val", 0) == 0 and self.jobs_used:
                lines.append("WARNING no val jobs   : training needs a validation split")
        if self.rebalanced:
            lines.append(
                f"split rebalanced     : moved a job into {', '.join(self.rebalanced)} "
                "(too few jobs for the configured ratios)"
            )
        if self.jobs_skipped_no_consent:
            lines.append(f"skipped (no consent) : {self.jobs_skipped_no_consent}")
        if self.jobs_skipped_unreadable:
            lines.append(f"skipped (unreadable) : {self.jobs_skipped_unreadable}")
        if self.kinds_unknown:
            unknown = ", ".join(f"{k}={v}" for k, v in sorted(self.kinds_unknown.items()))
            lines.append(f"kinds not in labels  : {unknown}")
        if self.dataset_yaml:
            lines.append(f"dataset.yaml         : {self.dataset_yaml}")
        return "\n".join(lines)


def export_dataset(
    cfg: Config | None = None,
    *,
    tile_size: int | None = None,
    tile_overlap: int | None = None,
    min_visible: float | None = None,
    min_box_px: int | None = None,
    background_ratio: float | None = None,
    require_consent: bool | None = None,
    kinds: Iterable[str] | None = None,
    clean: bool = False,
    progress: Any = None,
) -> ExportReport:
    """Convert every manifest into YOLO images + labels."""
    cfg = cfg or get_config()
    ds_cfg = cfg.section("dataset")

    tile_size = _pick(tile_size, ds_cfg.get("tile_size", 1024))
    tile_overlap = int(_pick(tile_overlap, ds_cfg.get("tile_overlap", 128)))
    min_visible = float(_pick(min_visible, ds_cfg.get("min_visible", 0.35)))
    min_box_px = int(_pick(min_box_px, ds_cfg.get("min_box_px", 4)))
    min_thickness = float(ds_cfg.get("min_thickness_px", min_box_px))
    background_ratio = float(_pick(background_ratio, ds_cfg.get("background_ratio", 0.1)))
    require_consent = bool(_pick(require_consent, ds_cfg.get("require_consent", False)))

    labels = list(cfg.labels)
    class_of = {name: idx for idx, name in enumerate(labels)}
    wanted = set(kinds) if kinds else set(labels)

    dataset_root = cfg.path("paths.dataset_root")
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    if clean:
        for sub in (images_root, labels_root):
            if sub.exists():
                shutil.rmtree(sub)

    splits = ds_cfg.get("split", {"train": 70, "val": 15, "test": 15})
    for split in ("train", "val", "test"):
        (images_root / split).mkdir(parents=True, exist_ok=True)
        (labels_root / split).mkdir(parents=True, exist_ok=True)

    report = ExportReport()
    # Seed every class so empty ones surface as a warning rather than vanishing.
    for name in labels:
        report.class_counts[name] = 0

    jobs: list[tuple[Path, dict[str, Any], str]] = []

    for manifest_path in iter_manifests(cfg.path("paths.estimates")):
        report.manifests_found += 1
        try:
            manifest = read_manifest(manifest_path)
        except Exception as exc:  # pragma: no cover - corrupt file
            logger.warning("Unreadable manifest %s: %s", manifest_path, exc)
            report.jobs_skipped_unreadable += 1
            continue

        if require_consent and not manifest.get("consent"):
            report.jobs_skipped_no_consent += 1
            continue

        job_id = manifest.get("job_id") or manifest_path.parent.name
        jobs.append((manifest_path, manifest, job_id))

    assignments = {job_id: assign_split(job_id, splits) for _, _, job_id in jobs}
    report.rebalanced = _ensure_trainable(assignments)
    for split in ("train", "val", "test"):
        report.split_jobs[split] = sum(1 for s in assignments.values() if s == split)

    for manifest_path, manifest, job_id in jobs:
        used = _export_job(
            manifest,
            cfg,
            split=assignments[job_id],
            images_root=images_root,
            labels_root=labels_root,
            class_of=class_of,
            wanted=wanted,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            min_visible=min_visible,
            min_box_px=min_box_px,
            min_thickness=min_thickness,
            background_ratio=background_ratio,
            report=report,
        )
        if used:
            report.jobs_used += 1
        if progress:
            progress(manifest_path)

    report.dataset_yaml = _write_dataset_yaml(dataset_root, labels)
    logger.info("Dataset export complete:\n%s", report.summary())
    return report


def _pick(explicit: Any, fallback: Any) -> Any:
    return fallback if explicit is None else explicit


def _ensure_trainable(assignments: dict[str, str]) -> list[str]:
    """Guarantee the export has something to train (and validate) on.

    Splitting by job is what stops tiles from leaking across splits, but with
    only one or two drawings the hash can drop every job into a single split -
    which Ultralytics refuses to train. In that case the lexicographically
    first job is moved, which keeps the result deterministic.
    """
    moved: list[str] = []
    if not assignments:
        return moved

    needed = ["train", "val"] if len(assignments) > 1 else ["train"]
    for want in needed:
        if want in assignments.values():
            continue
        donors = sorted(job for job, split in assignments.items() if split != want)
        if not donors:
            continue
        assignments[donors[0]] = want
        moved.append(want)
    return moved


def _resolve_image(page: dict[str, Any], cfg: Config) -> Path | None:
    """Prefer the recorded path, fall back to the configured image folder."""
    recorded = page.get("image_path")
    if recorded and Path(recorded).is_file():
        return Path(recorded)
    name = page.get("image_name")
    if not name:
        return None
    candidate = cfg.path("paths.processed_images") / name
    return candidate if candidate.is_file() else None


def _export_job(
    manifest: dict[str, Any],
    cfg: Config,
    *,
    split: str,
    images_root: Path,
    labels_root: Path,
    class_of: dict[str, int],
    wanted: set[str],
    tile_size: int,
    tile_overlap: int,
    min_visible: float,
    min_box_px: int,
    min_thickness: float,
    background_ratio: float,
    report: ExportReport,
) -> bool:
    import cv2  # imported lazily: the estimator itself does not need it

    job_id = manifest.get("job_id", "unknown")
    wrote_any = False

    for page in manifest.get("pages", []):
        image_path = _resolve_image(page, cfg)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Could not read %s", image_path)
            continue
        height, width = image.shape[:2]

        # PDF point -> pixel. Derived from the recorded raster size rather
        # than zoom, so it is exact even when the pixmap rounded.
        width_pt = float(page.get("width_pt") or 0.0)
        height_pt = float(page.get("height_pt") or 0.0)
        sx = (width / width_pt) if width_pt > 0 else 1.0
        sy = (height / height_pt) if height_pt > 0 else 1.0

        boxes: list[tuple[str, tuple[float, float, float, float]]] = []
        for el in page.get("elements", []):
            kind = el.get("kind")
            box = el.get("bbox_pt")
            if not kind or not box:
                continue
            if kind not in class_of:
                report.kinds_unknown[kind] = report.kinds_unknown.get(kind, 0) + 1
                continue
            if kind not in wanted:
                continue
            px_box = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
            px_box = (
                min(max(px_box[0], 0.0), width),
                min(max(px_box[1], 0.0), height),
                min(max(px_box[2], 0.0), width),
                min(max(px_box[3], 0.0), height),
            )
            if (px_box[2] - px_box[0]) < min_thickness or (px_box[3] - px_box[1]) < min_thickness:
                px_box = thicken_box(px_box, min_thickness, width, height)
                report.boxes_thickened += 1
            boxes.append((kind, px_box))

        windows = tile_windows(width, height, tile_size, tile_overlap)
        tiled = len(windows) > 1

        for tile_index, (wx1, wy1, wx2, wy2) in enumerate(windows):
            tile_w, tile_h = wx2 - wx1, wy2 - wy1
            labels_here: list[tuple[str, str]] = []
            for kind, box in boxes:
                # A box that lies outside this tile is simply not here - it is
                # not a rejected label, so it must not inflate the drop count.
                if not (box[2] > wx1 and box[0] < wx2 and box[3] > wy1 and box[1] < wy2):
                    continue
                clipped = clip_box(
                    (box[0] - wx1, box[1] - wy1, box[2] - wx1, box[3] - wy1),
                    (0, 0, tile_w, tile_h),
                    min_visible,
                )
                if clipped is None:
                    report.boxes_dropped += 1
                    continue
                cw, ch = clipped[2] - clipped[0], clipped[3] - clipped[1]
                if cw < min_box_px or ch < min_box_px:
                    report.boxes_dropped += 1
                    continue
                labels_here.append((kind, yolo_line(class_of[kind], clipped, tile_w, tile_h)))

            stem = f"{job_id}_{page.get('index', 0):03d}"
            if tiled:
                stem += f"_t{tile_index:04d}"

            if not labels_here:
                # Pure background. Keeping a small, deterministic share of
                # these gives the model negatives to learn from; keeping all
                # of them would bury the signal.
                if background_ratio <= 0 or not _keep_background(stem, background_ratio):
                    continue
                report.background_tiles += 1
            else:
                report.boxes_written[split] = report.boxes_written.get(split, 0) + len(labels_here)
                for kind, _ in labels_here:
                    report.class_counts[kind] = report.class_counts.get(kind, 0) + 1

            image_out = images_root / split / f"{stem}.jpg"
            label_out = labels_root / split / f"{stem}.txt"

            if tiled:
                crop = image[wy1:wy2, wx1:wx2]
                cv2.imwrite(
                    str(image_out), crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )
            else:
                shutil.copy2(image_path, image_out)

            body = "\n".join(line for _, line in labels_here)
            label_out.write_text((body + "\n") if body else "", encoding="utf-8")

            report.images_written[split] = report.images_written.get(split, 0) + 1
            wrote_any = True

    return wrote_any


def _keep_background(stem: str, ratio: float) -> bool:
    """Deterministic thinning so repeated exports produce the same dataset."""
    bucket = int(hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8], 16) % 1000
    return bucket < int(ratio * 1000)


def _write_dataset_yaml(dataset_root: Path, labels: list[str]) -> str:
    """Same content Ultralytics expects, written without importing it."""
    dataset_root.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(labels))
    body = (
        f"path: {dataset_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        f"nc: {len(labels)}\n"
        "names:\n"
        f"{names}\n"
    )
    target = dataset_root / "dataset.yaml"
    target.write_text(body, encoding="utf-8")
    return str(target)


# --------------------------------------------------------------------------- #
# retention
# --------------------------------------------------------------------------- #
@dataclass
class PruneReport:
    uploads: list[str] = field(default_factory=list)
    estimates: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    bytes_freed: int = 0

    def summary(self, dry_run: bool) -> str:
        verb = "would delete" if dry_run else "deleted"
        return (
            f"{verb}: {len(self.uploads)} upload(s), {len(self.estimates)} estimate dir(s), "
            f"{len(self.images)} rendered page(s)\n"
            f"bytes {'recoverable' if dry_run else 'freed'}: {self.bytes_freed / 1e6:.1f} MB"
        )


def prune(
    cfg: Config | None = None,
    *,
    days: int | None = None,
    dry_run: bool = False,
) -> PruneReport:
    """Delete jobs older than ``days``.

    Structural drawings are client-confidential, so keeping every upload
    forever is a liability - this is the cheap way to bound it. On Render the
    disk is ephemeral anyway, so this matters mostly for local/self-hosted use.
    """
    cfg = cfg or get_config()
    ds_cfg = cfg.section("dataset")
    days = int(_pick(days, ds_cfg.get("retention_days", 7)))
    cutoff = time.time() - days * 86400

    report = PruneReport()
    uploads_dir = cfg.path("paths.uploads")
    estimates_dir = cfg.path("paths.estimates")
    images_dir = cfg.path("paths.processed_images")

    doomed: list[tuple[Path, str]] = []
    job_ids: set[str] = set()

    if uploads_dir.is_dir():
        for path in uploads_dir.iterdir():
            if path.is_file() and path.stat().st_mtime < cutoff:
                job_id = path.name.split("_", 1)[0]
                job_ids.add(job_id)
                doomed.append((path, "uploads"))

    if estimates_dir.is_dir():
        for path in estimates_dir.iterdir():
            if path.is_dir() and path.stat().st_mtime < cutoff:
                job_ids.add(path.name)
                doomed.append((path, "estimates"))

    if images_dir.is_dir():
        for path in images_dir.iterdir():
            if not path.is_file():
                continue
            job_id = path.name.split("_", 1)[0]
            if job_id in job_ids and path.stat().st_mtime < cutoff:
                doomed.append((path, "images"))
            elif path.stat().st_mtime < cutoff:
                # Orphaned raster with no surviving job.
                doomed.append((path, "images"))

    for path, bucket in doomed:
        size = _tree_size(path)
        report.bytes_freed += size
        getattr(report, bucket).append(str(path))
        if dry_run:
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:  # pragma: no cover
            logger.warning("Could not delete %s: %s", path, exc)

    logger.info("Prune complete:\n%s", report.summary(dry_run))
    return report


def _tree_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total
