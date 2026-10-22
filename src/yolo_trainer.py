"""YOLO training helper.

Generates the Ultralytics dataset descriptor from the project config and
runs training.  The dataset layout expected is::

    data/dataset/
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from ultralytics import YOLO

from .config import Config, get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOTrainer:
    def __init__(self, config: Config | None = None, config_path: str | Path | None = None):
        self.cfg = config or get_config()
        self.yolo_cfg = self.cfg.section("yolo")
        self.labels = self.cfg.labels
        Path(self.yolo_cfg["save_path"]).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def create_dataset_yaml(self, dataset_root: str | Path | None = None) -> str:
        root = Path(dataset_root or self.cfg.path("paths.dataset_root"))
        content = f"""path: {root}
train: images/train
val: images/val
test: images/test

nc: {len(self.labels)}
names: {self.labels}
"""
        out = root / "dataset.yaml"
        out.write_text(content, encoding="utf-8")
        logger.info("dataset.yaml written at %s", out)
        return str(out)

    # ------------------------------------------------------------------ #
    def train(self, dataset_yaml: str):
        model_name = self.yolo_cfg["model_name"]
        logger.info("Loading base model: %s.pt", model_name)
        model = YOLO(f"{model_name}.pt")

        results = model.train(
            data=dataset_yaml,
            epochs=int(self.yolo_cfg.get("epochs", 100)),
            imgsz=int(self.yolo_cfg.get("imgsz", 960)),
            batch=int(self.yolo_cfg.get("batch_size", 8)),
            device=self.yolo_cfg.get("device", "auto"),
            workers=int(self.yolo_cfg.get("workers", 2)),
            patience=int(self.yolo_cfg.get("patience", 15)),
            project=self.yolo_cfg["save_path"],
            name="construction_detector",
            exist_ok=True,
        )
        logger.info("Training finished")
        return results

    # ------------------------------------------------------------------ #
    def best_weights_path(self) -> str | None:
        p = Path(self.yolo_cfg["save_path"]) / "construction_detector" / "weights" / "best.pt"
        return str(p) if p.exists() else None
