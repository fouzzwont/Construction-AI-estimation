"""Validation for the exported YOLO dataset.

Checks the things that silently ruin a training run: malformed label rows,
orphaned images, class ids out of range, degenerate boxes, and - most
importantly - tiles from the same drawing leaking across splits.

    python tools/dataset_test.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

from src.config import get_config  # noqa: E402

SPLITS = ("train", "val", "test")

failures: list[str] = []
checks = 0


def check(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(message)
        print(f"  FAIL  {message}")


def main() -> int:
    cfg = get_config()
    root = cfg.path("paths.dataset_root")
    images_root = root / "images"
    labels_root = root / "labels"
    class_names = list(cfg.labels)
    n_classes = len(class_names)

    print(f"dataset root : {root}")
    check(root.is_dir(), "dataset root does not exist")

    # ---- dataset.yaml -----------------------------------------------------
    yaml_path = root / "dataset.yaml"
    check(yaml_path.is_file(), "dataset.yaml missing")
    if yaml_path.is_file():
        spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        check(spec.get("nc") == n_classes, f"nc={spec.get('nc')} but {n_classes} labels configured")
        check(spec.get("train") == "images/train", "train path is not images/train")
        check(spec.get("val") == "images/val", "val path is not images/val")
        names = spec.get("names")
        if isinstance(names, dict):
            ok = all(names.get(i) == name for i, name in enumerate(class_names))
        else:
            ok = list(names or []) == class_names
        check(bool(ok), f"names {names!r} do not match config labels {class_names!r}")

    # ---- walk the splits --------------------------------------------------
    job_splits: dict[str, set[str]] = defaultdict(set)
    total_images = 0
    total_boxes = 0
    class_hist: Counter[int] = Counter()
    empty_labels = 0

    for split in SPLITS:
        img_dir = images_root / split
        lbl_dir = labels_root / split
        check(img_dir.is_dir(), f"images/{split} missing")
        check(lbl_dir.is_dir(), f"labels/{split} missing")
        if not img_dir.is_dir():
            continue

        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        stems = {p.stem for p in images}
        labels = {p.stem: p for p in lbl_dir.glob("*.txt")}

        orphans = set(labels) - stems
        check(not orphans, f"{split}: {len(orphans)} label(s) without an image: {sorted(orphans)[:3]}")

        missing = stems - set(labels)
        check(not missing, f"{split}: {len(missing)} image(s) without a label file: {sorted(missing)[:3]}")

        total_images += len(images)

        for image in images:
            # job id is the 12-hex prefix; page and tile follow
            job_id = image.stem.split("_", 1)[0]
            job_splits[job_id].add(split)

            label = labels.get(image.stem)
            if label is None:
                continue
            rows = [r for r in label.read_text(encoding="utf-8").splitlines() if r.strip()]
            if not rows:
                empty_labels += 1
                continue

            for lineno, row in enumerate(rows, 1):
                parts = row.split()
                if len(parts) != 5:
                    check(False, f"{split}/{label.name}:{lineno} has {len(parts)} fields, expected 5")
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, bw, bh = (float(v) for v in parts[1:])
                except ValueError:
                    check(False, f"{split}/{label.name}:{lineno} is not numeric")
                    continue

                if not 0 <= cls < n_classes:
                    check(False, f"{split}/{label.name}:{lineno} class {cls} out of range 0..{n_classes - 1}")
                    continue
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
                    check(
                        False,
                        f"{split}/{label.name}:{lineno} normalised value outside [0,1]: "
                        f"{cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}",
                    )
                    continue
                if bw <= 0 or bh <= 0:
                    check(False, f"{split}/{label.name}:{lineno} degenerate box w={bw} h={bh}")
                    continue
                # A box must lie inside the frame it is normalised to.
                if cx - bw / 2 < -1e-4 or cx + bw / 2 > 1.0001:
                    check(False, f"{split}/{label.name}:{lineno} box escapes horizontally")
                if cy - bh / 2 < -1e-4 or cy + bh / 2 > 1.0001:
                    check(False, f"{split}/{label.name}:{lineno} box escapes vertically")

                class_hist[cls] += 1
                total_boxes += 1

    # ---- leakage ----------------------------------------------------------
    leaked = {job: sorted(s) for job, s in job_splits.items() if len(s) > 1}
    check(not leaked, f"{len(leaked)} job(s) span multiple splits (leakage): {list(leaked.items())[:2]}")

    # ---- balance ----------------------------------------------------------
    check(total_images > 0, "no images exported at all")
    check(total_boxes > 0, "no boxes exported at all")
    check(len(job_splits) > 0, "no jobs represented")

    # ---- report -----------------------------------------------------------
    print()
    print(f"  images           : {total_images}  (empty/negative labels: {empty_labels})")
    print(f"  boxes            : {total_boxes}")
    print(f"  jobs             : {len(job_splits)}")
    print("  boxes per class  :")
    for idx, name in enumerate(class_names):
        print(f"      {idx} {name:<9} {class_hist.get(idx, 0)}")
    print()

    if failures:
        print(f"{len(failures)} FAILURE(S) out of {checks} checks")
        return 1
    print(f"{checks}/{checks} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
