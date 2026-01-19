import logging
from pathlib import Path

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOTrainer:
  def __init__(self, config_path="config/config.yaml"):
    with open(config_path, "r") as f:
      cfg = yaml.safe_load(f)
    self.cfg = cfg
    self.yolo_cfg = cfg["yolo"]
    self.labels = cfg["annotation"]["labels"]
    Path(self.yolo_cfg["save_path"]).mkdir(parents=True, exist_ok=True)

  def create_dataset_yaml(self, dataset_root="data/dataset") -> str:
    content = f"""path: {dataset_root}
train: images/train
val: images/val

nc: {len(self.labels)}
names: {self.labels}
"""
    out = Path(dataset_root) / "dataset.yaml"
    out.write_text(content)
    logger.info(f"dataset.yaml written at {out}")
    return str(out)

  def train(self, dataset_yaml: str):
    model_name = self.yolo_cfg["model_name"]
    logger.info(f"Loading base model: {model_name}.pt")
    model = YOLO(f"{model_name}.pt")

    results = model.train(
      data=dataset_yaml,
      epochs=self.yolo_cfg["epochs"],
      imgsz=self.yolo_cfg["imgsz"],
      batch=self.yolo_cfg["batch_size"],
      device=self.yolo_cfg["device"],
      workers=self.yolo_cfg["workers"],
      project=self.yolo_cfg["save_path"],
      name="construction_detector",
      exist_ok=True,
    )
    logger.info("Training finished")
    return results

  def best_weights_path(self) -> str | None:
    p = Path(self.yolo_cfg["save_path"]) / "construction_detector" / "weights" / "best.pt"
    return str(p) if p.exists() else None
