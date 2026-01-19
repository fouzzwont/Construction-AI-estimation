import json
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConstructionInference:
  def __init__(self, weights_path: str, config_path="config/config.yaml"):
    with open(config_path, "r") as f:
      cfg = yaml.safe_load(f)
    self.cfg = cfg
    self.labels = cfg["annotation"]["labels"]
    self.inf_cfg = cfg["inference"]
    self.eng_cfg = cfg["engineering"]

    self.model = YOLO(weights_path)
    logger.info(f"Loaded model from {weights_path}")

  def _detections(self, result):
    dets = []
    for box in result.boxes:
      cls_id = int(box.cls)
      x1, y1, x2, y2 = box.xyxy[0].tolist()
      dets.append(
        {
          "class_id": cls_id,
          "class_name": self.labels[cls_id] if cls_id < len(self.labels) else "unknown",
          "confidence": float(box.conf),
          "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }
      )
    return dets

  def _quantities(self, detections, img_shape):
    # Simple placeholder logic – you can refine based on your engineering rules.
    scale = self.eng_cfg["pixel_to_mm_scale"]
    q = {
      "rebar": {"count": 0},
      "beam": {"count": 0},
      "slab": {"count": 0},
      "column": {"count": 0},
      "wall": {"count": 0},
    }
    for d in detections:
      name = d["class_name"]
      if name in q:
        q[name]["count"] += 1
    return q

  def run_on_image(self, image_path: str, out_dir="output/estimates"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = Path(image_path)

    res = self.model.predict(
      source=str(img_path),
      conf=self.inf_cfg["conf_threshold"],
      iou=self.inf_cfg["iou_threshold"],
      device=self.inf_cfg["device"],
      verbose=False,
    )[0]

    dets = self._detections(res)
    img = cv2.imread(str(img_path))
    q = self._quantities(dets, img.shape if img is not None else None)

    # Save annotated image
    annotated = res.plot()
    ann_path = out_dir / f"{img_path.stem}_annotated.jpg"
    cv2.imwrite(str(ann_path), annotated)

    out = {
      "image": str(img_path),
      "detections": dets,
      "quantities": q,
      "total_detections": len(dets),
      "avg_confidence": float(np.mean([d["confidence"] for d in dets])) if dets else 0.0,
    }
    json_path = out_dir / f"{img_path.stem}_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    logger.info(f"Saved results to {json_path}")
    return out
