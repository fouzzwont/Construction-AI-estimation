import logging
import os
from pathlib import Path

import yaml
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from src.pdf_processor import PDFProcessor
from src.inference import ConstructionInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
with open("config/config.yaml", "r") as f:
  CONFIG = yaml.safe_load(f)

app.config["UPLOAD_FOLDER"] = CONFIG["pdf_processing"]["input_path"]
app.config["MAX_CONTENT_LENGTH"] = CONFIG["deployment"]["upload_limit_mb"] * 1024 * 1024

pdf_processor = PDFProcessor()
weights = CONFIG["yolo"]["weights_path"]
inference_engine = None
if Path(weights).exists():
  inference_engine = ConstructionInference(weights)
else:
  logger.warning("Weights not found. Train model first.")

@app.get("/health")
def health():
  return jsonify({"status": "ok", "version": CONFIG["project"]["version"]})

@app.post("/api/batch-estimate")
def batch_estimate():
  if inference_engine is None:
    return jsonify({"error": "Model not ready"}), 503

  if "pdf" not in request.files:
    return jsonify({"error": "No PDF uploaded"}), 400

  f = request.files["pdf"]
  fname = secure_filename(f.filename)
  pdf_path = Path(app.config["UPLOAD_FOLDER"]) / fname
  pdf_path.parent.mkdir(parents=True, exist_ok=True)
  f.save(str(pdf_path))

  images = pdf_processor.process_pdf(str(pdf_path), prefix=pdf_path.stem)
  all_res = []
  for img in images:
    r = inference_engine.run_on_image(img)
    all_res.append(r)

  return jsonify({"success": True, "pages": len(images), "results": all_res})


if __name__ == "__main__":
  Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
  Path("output/estimates").mkdir(parents=True, exist_ok=True)
  app.run(
    host=CONFIG["deployment"]["api_host"],
    port=CONFIG["deployment"]["api_port"],
    debug=CONFIG["deployment"]["debug"],
  )
