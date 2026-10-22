"""Command line entry point.

    python main.py --serve                 start the web UI
    python main.py --pdf drawing.pdf       run the full estimation pipeline
    python main.py --train                 train the YOLO detector
    python main.py --infer page.jpg        detect on a single image
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import get_config
from src.pipeline import EstimationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# Windows consoles default to cp1252 which cannot render the rupee sign
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def _print_estimate(result) -> None:
    sym = result.currency_symbol
    print("\n" + "=" * 66)
    print(f"  {result.project_name}")
    print(f"  source: {result.source_file}   scale: {result.scale.get('ratio', 'n/a')}"
          f"   storeys: {result.storeys}   confidence: {result.confidence*100:.0f}%")
    print("=" * 66)
    print(f"  {'CATEGORY':<16}{'ITEM':<42}{'QTY':>12} {'RATE':>12} {'AMOUNT':>14}")
    print("  " + "-" * 64)
    for line in result.boq:
        print(f"  {line['category']:<16}{line['item'][:40]:<42}"
              f"{line['quantity']:>12,.2f} {sym}{line['rate']:>11,.0f} {sym}{line['amount']:>14,.2f}")
    print("  " + "-" * 64)
    for cat, amount in result.totals.items():
        print(f"  {cat:<58}{sym}{amount:>16,.2f}")
    print("=" * 66)
    print(f"  {'TOTAL ESTIMATE':<58}{sym}{result.grand_total:>16,.2f}")
    print("=" * 66 + "\n")

    if result.rebar:
        print("  REINFORCEMENT SCHEDULE")
        print(f"  {'DIA':<8}{'LENGTH (m)':<16}{'NOS':<10}{'WEIGHT (t)':<14}")
        for row in result.rebar:
            print(f"  {row['diameter_mm']:<8}{row['total_length_m']:<16,.1f}"
                  f"{row['count']:<10}{row['weight_t']:<14,.3f}")
        print()

    if result.warnings:
        print("  WARNINGS")
        for w in result.warnings:
            print(f"   ! {w}")
        print()


def run_pdf(args: argparse.Namespace) -> int:
    cfg = get_config()
    cfg.ensure_dirs()
    pipeline = EstimationPipeline(cfg)

    def progress(stage: str, pct: float) -> None:
        print(f"  [{pct*100:5.1f}%] {stage}")

    output = pipeline.run(
        args.pdf,
        scale_override=args.scale,
        market=args.market,
        project_name=args.name or "",
        progress=progress,
    )
    _print_estimate(output.result)
    print(f"  CSV report : {output.csv_path}")
    print(f"  PDF report : {output.pdf_path}")
    print(f"  Elapsed    : {output.elapsed:.2f}s\n")
    return 0


def run_serve(args: argparse.Namespace) -> int:
    from src.deployment import app, main as serve_main

    cfg = get_config()
    if args.host:
        cfg.data.setdefault("deployment", {})["api_host"] = args.host
    if args.port:
        cfg.data.setdefault("deployment", {})["api_port"] = args.port
    serve_main()
    return 0


def run_train(args: argparse.Namespace) -> int:
    from src.yolo_trainer import YOLOTrainer

    cfg = get_config()
    cfg.ensure_dirs()
    trainer = YOLOTrainer(cfg)
    ds_yaml = trainer.create_dataset_yaml(str(cfg.path("paths.dataset_root")))
    trainer.train(ds_yaml)
    logger.info("Best weights: %s", trainer.best_weights_path())
    return 0


def run_infer(args: argparse.Namespace) -> int:
    from src.blueprint.detector import BlueprintDetector
    from src.blueprint.models import Element

    cfg = get_config()
    detector = BlueprintDetector(cfg)
    weights = cfg.path("yolo.weights_path")
    if not weights.exists():
        logger.error("No YOLO weights at %s - train first (python main.py --train)", weights)
        return 1

    from ultralytics import YOLO

    model = YOLO(str(weights))
    res = model.predict(source=args.infer, conf=cfg.get("inference.conf_threshold"), verbose=False)[0]
    for box in res.boxes:
        cls = int(box.cls[0])
        name = res.names.get(cls, cls)
        xy = box.xyxy[0].tolist()
        print(f"  {name:<14} conf={float(box.conf[0]):.2f}  bbox={[round(v) for v in xy]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="construction-ai",
        description="Construction AI Estimation - blueprint take-off and costing",
    )
    p.add_argument("--pdf", type=str, help="PDF drawing to analyse")
    p.add_argument("--serve", action="store_true", help="start the web UI")
    p.add_argument("--train", action="store_true", help="train the YOLO detector")
    p.add_argument("--infer", type=str, help="run detection on a single image")
    p.add_argument("--scale", type=float, help="override the drawing scale (e.g. 100 for 1:100)")
    p.add_argument("--market", type=str, help="rate market (india_urban, gcc, ...)")
    p.add_argument("--name", type=str, help="project name for the report")
    p.add_argument("--host", type=str, help="bind host for --serve")
    p.add_argument("--port", type=int, help="port for --serve")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.serve:
        return run_serve(args)
    if args.pdf:
        if not Path(args.pdf).is_file():
            logger.error("File not found: %s", args.pdf)
            return 1
        return run_pdf(args)
    if args.train:
        return run_train(args)
    if args.infer:
        return run_infer(args)

    build_parser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
