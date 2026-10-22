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
