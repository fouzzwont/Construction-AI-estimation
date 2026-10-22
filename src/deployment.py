"""Flask application: upload a drawing, get a priced estimate.

Endpoints
---------
``GET  /``                             web UI
``GET  /health``                       service health
``GET  /api/markets``                  available rate markets
``POST /api/upload``                   upload a PDF, start an analysis job
``GET  /api/job/<job_id>``             job progress
``GET  /api/result/<job_id>``          full estimate payload
``GET  /api/download/<job_id>/<fmt>``  download the CSV or PDF report
``GET  /api/page/<job_id>/<page>``     annotated page image
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from .config import get_config
from .pipeline import EstimationPipeline
from .rates import AVAILABLE_MARKETS, load_rate_card

logger = logging.getLogger(__name__)

CFG = get_config()
CFG.ensure_dirs()

WEB_ROOT = CFG.root / "web"

app = Flask(
    __name__,
    template_folder=str(WEB_ROOT / "templates"),
    static_folder=str(WEB_ROOT / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = int(CFG.get("deployment.upload_limit_mb", 80)) * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False

UPLOAD_DIR = CFG.path("paths.uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_pipeline = EstimationPipeline(CFG)


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def _set_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        job = _jobs.setdefault(job_id, {"job_id": job_id})
        job.update(fields)


def _get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def _run_job(job_id: str, pdf_path: Path, scale_override: float | None, market: str | None, project_name: str) -> None:
    try:
        def progress(stage: str, pct: float) -> None:
            _set_job(job_id, stage=stage, progress=round(pct, 3), status="running")

        output = _pipeline.run(
            pdf_path,
            scale_override=scale_override,
            market=market,
            project_name=project_name,
            progress=progress,
        )
        _set_job(
            job_id,
            status="done",
            stage="Done",
            progress=1.0,
            result=output.result.to_dict(),
            pages=output.pages,
            csv_path=output.csv_path,
            pdf_path=output.pdf_path,
            elapsed=output.elapsed,
        )
        logger.info("Job %s completed", job_id)
    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc)
        logger.debug(traceback.format_exc())
        _set_job(job_id, status="error", stage="Failed", error=str(exc))


# --------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------- #
@app.get("/")
def index():
    return render_template(
        "index.html",
        markets=AVAILABLE_MARKETS,
        default_market=CFG.get("rates.market", "india_urban"),
        default_scale=CFG.get("scale.default", 100),
        currency=CFG.get("rates.currency_symbol", "₹"),
        upload_limit=CFG.get("deployment.upload_limit_mb", 80),
        version=CFG.get("project.version", "2.0.0"),
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "version": CFG.get("project.version", "2.0.0"),
            "yolo_available": _pipeline.detector.yolo_available,
            "jobs": len(_jobs),
        }
    )


@app.get("/api/markets")
def markets():
    return jsonify({"markets": AVAILABLE_MARKETS, "default": CFG.get("rates.market")})


@app.post("/api/upload")
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file was uploaded."}), 400

    file = request.files["pdf"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF drawings are supported."}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid file name."}), 400

    job_id = uuid.uuid4().hex[:12]
    target = UPLOAD_DIR / f"{job_id}_{filename}"
    file.save(str(target))

    try:
        scale_override = float(request.form.get("scale") or 0) or None
    except ValueError:
        scale_override = None
    market = request.form.get("market") or None
    project_name = (request.form.get("project_name") or "").strip()

    _set_job(job_id, status="queued", stage="Queued", progress=0.0, filename=filename)

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, target, scale_override, market, project_name),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued", "filename": filename}), 202


@app.get("/api/job/<job_id>")
def job_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    payload = {
        "job_id": job_id,
        "status": job.get("status", "queued"),
        "stage": job.get("stage", ""),
        "progress": job.get("progress", 0.0),
    }
    if job.get("error"):
        payload["error"] = job["error"]
    return jsonify(payload)


@app.get("/api/result/<job_id>")
def job_result(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    if job.get("status") != "done":
        return jsonify({"status": job.get("status", "queued"), "progress": job.get("progress", 0.0)}), 202
    return jsonify(
        {
            "result": job.get("result"),
            "pages": job.get("pages", []),
            "elapsed_seconds": job.get("elapsed"),
        }
    )


@app.get("/api/download/<job_id>/<fmt>")
def download(job_id: str, fmt: str):
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404

    key = {"csv": "csv_path", "pdf": "pdf_path"}.get(fmt.lower())
    if key is None:
        return jsonify({"error": "Unknown format"}), 400

    path = job.get(key)
    if not path or not Path(path).exists():
        return jsonify({"error": f"{fmt.upper()} report not available"}), 404
    return send_file(path, as_attachment=True, download_name=Path(path).name)


@app.get("/api/page/<job_id>/<int:page_no>")
def page_image(job_id: str, page_no: int):
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    for page in job.get("pages", []):
        if page.get("index") == page_no:
            annotated = page.get("annotated_path")
            if annotated and Path(annotated).exists():
                return send_file(annotated, mimetype="image/jpeg")
            original = page.get("image_path")
            if original and Path(original).exists():
                return send_file(original, mimetype="image/jpeg")
    return jsonify({"error": "Page not found"}), 404


@app.errorhandler(413)
def too_large(_error):
    limit = int(CFG.get("deployment.upload_limit_mb", 80))
    return jsonify({"error": f"File is too large. Maximum upload size is {limit} MB."}), 413


def main() -> None:
    import os

    # Render / Railway / Fly inject PORT; fall back to config for local use.
    host = os.environ.get("HOST") or CFG.get("deployment.api_host", "127.0.0.1")
    port = int(os.environ.get("PORT") or CFG.get("deployment.api_port", 8000))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes") or (
        "FLASK_DEBUG" not in os.environ and bool(CFG.get("deployment.debug", False))
    )
    # The auto-reloader forks a second process and reloads open browser tabs
    # mid-request, so it stays off unless explicitly requested.
    reload = os.environ.get("FLASK_RELOAD", "").lower() in ("1", "true", "yes") or (
        "FLASK_RELOAD" not in os.environ and bool(CFG.get("deployment.reload", False))
    )
    print(f"\n  Construction AI Estimation  ->  http://{host}:{port}\n")
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=reload)


if __name__ == "__main__":
    main()
