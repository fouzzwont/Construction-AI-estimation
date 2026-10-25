"""End-to-end self test.

Generates a vector test drawing, runs the full pipeline over it and checks
that the expected quantities come out.  Run it after any change:

    python tools/selftest.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.config import get_config  # noqa: E402
from src.pipeline import EstimationPipeline  # noqa: E402
from src.blueprint.scale import detect_scale_from_text, is_non_plan_sheet  # noqa: E402


def build_sample(path: Path) -> Path:
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "make_sample_drawing.py"), "--out", str(path)],
        check=True,
        cwd=ROOT,
    )
    return path


def main() -> int:
    out_dir = ROOT / "output" / "selftest"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = build_sample(out_dir / "selftest_drawing.pdf")

    cfg = get_config()
    pipeline = EstimationPipeline(cfg)

    def progress(stage: str, pct: float) -> None:
        print(f"  [{pct*100:5.1f}%] {stage}")

    result = pipeline.run(sample, progress=progress)
    est = result.result

    print("\n" + "=" * 70)
    print(f"  scale      : {est.scale.get('ratio')}  (method={est.scale.get('method')})")
    print(f"  storeys    : {est.storeys}")
    print(f"  elements   : {est.counts}")
    print(f"  rebar rows : {len(est.rebar)}")
    print(f"  quantities : {est.quantities}")
    print(f"  confidence : {est.confidence*100:.0f}%")
    print("-" * 70)
    for line in est.boq:
        print(f"  {line['category']:<15}{line['item'][:38]:<40}"
              f"{line['quantity']:>10,.2f} {line['unit']:<4}{est.currency_symbol}{line['amount']:>13,.0f}")
    print("-" * 70)
    print(f"  GRAND TOTAL{'':<42}{est.currency_symbol}{est.grand_total:>17,.0f}")
    print("=" * 70)

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    check("scale detected from title block", est.scale.get("method") == "text",
          f"method={est.scale.get('method')} ratio={est.scale.get('ratio')}")
    check("scale is 1:100", est.scale.get("ratio") == "1:100", str(est.scale.get("ratio")))

    # Imperial title blocks: ``1/4" = 1'-0"`` is 1:48, not 1:4. Reading the bare
    # fraction as the denominator inflated real US sets by ~144x.
    imperial_cases = [
        ('SCALE 1/4" = 1\'-0"', 48.0),
        ('SCALE 1/8" = 1\'-0"', 96.0),
        ('SCALE 1/2" = 1\'-0"', 24.0),
        ('SCALE 1" = 50\'', 600.0),
        ('SCALE 1 4" = 1\'-0"', 48.0),  # stacked fraction, slash lost in extraction
    ]
    for text, expected in imperial_cases:
        got = detect_scale_from_text(text)
        check(
            f"imperial scale {text!r} -> 1:{int(expected)}",
            got is not None and abs(got.denominator - expected) < 0.5,
            f"got {got.denominator if got else None}",
        )

    nts = detect_scale_from_text("SCALE: NTS")
    check("NTS sheet is marked not-to-scale", nts is not None and nts.method == "nts",
          f"method={nts.method if nts else None}")
    check("detail reference 1/14 is not read as a scale",
          detect_scale_from_text("detail 1/14 AND 2/14") is None)
    check("elevation sheet is non-plan", is_non_plan_sheet("REAR ELEVATION\nSCALE 1/4\" = 1'-0\""))
    check("floor plan is a plan", not is_non_plan_sheet("FLOOR PLAN EXAMPLE\nSCALE 1:100"))
    check("storeys detected (G+1 -> 2)", est.storeys == 2, f"got {est.storeys}")
    check("columns detected", est.counts.get("column", 0) >= 6, f"got {est.counts.get('column')}")
    check("zones/slabs detected", est.counts.get("slab", 0) >= 4, f"got {est.counts.get('slab')}")
    check("walls detected", est.counts.get("wall", 0) >= 4, f"got {est.counts.get('wall')}")
    check("rebar schedule parsed", est.schedule_rows >= 8, f"got {est.schedule_rows} rows")
    check("rebar grouped by diameter", len(est.rebar) >= 4, f"got {len(est.rebar)} diameter groups")
    check("concrete volume > 0", est.quantities.get("concrete_m3", 0) > 0,
          f"{est.quantities.get('concrete_m3'):.2f} m3")
    check("steel weight > 0", est.quantities.get("steel_kg", 0) > 0,
          f"{est.quantities.get('steel_kg'):.0f} kg")
    check("formwork area > 0", est.quantities.get("formwork_m2", 0) > 0,
          f"{est.quantities.get('formwork_m2'):.1f} m2")
    check("grand total > 0", est.grand_total > 0, f"{est.grand_total:,.0f}")
    check("CSV report written", bool(result.csv_path) and Path(result.csv_path).exists())
    check("PDF report written", bool(result.pdf_path) and Path(result.pdf_path).exists())
    check("annotated pages written", len(result.pages) >= 1 and
          all(p.get("annotated_path") for p in result.pages))

    print()
    failed = 0
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{status}] {name}{('  -> ' + detail) if detail else ''}")

    print(f"\n  {len(checks) - failed}/{len(checks)} checks passed\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
