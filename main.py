import argparse
import logging
from pathlib import Path

from src.pdf_processor import PDFProcessor
from src.yolo_trainer import YOLOTrainer
from src.inference import ConstructionInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_dirs():
  for p in [
    "data/raw_pdfs",
    "data/processed_images",
    "data/dataset/images/train",
    "data/dataset/images/val",
    "data/dataset/images/test",
    "data/dataset/labels/train",
    "data/dataset/labels/val",
    "data/dataset/labels/test",
    "models/yolo_weights",
    "output/estimates",
    "logs",
  ]:
    Path(p).mkdir(parents=True, exist_ok=True)


def run_pdf(pdf_path: str):
  ensure_dirs()
  proc = PDFProcessor()
  proc.process_pdf(pdf_path, prefix=Path(pdf_path).stem)


def run_train():
  ensure_dirs()
  trainer = YOLOTrainer()
  ds_yaml = trainer.create_dataset_yaml()
  trainer.train(ds_yaml)
  logger.info(f"Best weights: {trainer.best_weights_path()}")


def run_infer(image_path: str):
  ensure_dirs()
  trainer = YOLOTrainer()  # just to read config
  weights = trainer.yolo_cfg["weights_path"]
  inf = ConstructionInference(weights)
  inf.run_on_image(image_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--pdf", type=str, help="PDF to convert")
  parser.add_argument("--train", action="store_true", help="Train YOLO")
  parser.add_argument("--infer", type=str, help="Run inference on image")
  args = parser.parse_args()

  if args.pdf:
    run_pdf(args.pdf)
  elif args.train:
    run_train()
  elif args.infer:
    run_infer(args.infer)
  else:
    parser.print_help()
