"""Generate a realistic vector test drawing.

The produced PDF mimics what a CAD export of a residential plan looks like:
a structural grid, columns, masonry walls, beams, a title block carrying the
drawing scale and a reinforcement schedule table.  It exists so the whole
pipeline can be exercised end to end without proprietary drawings.

    python tools/make_sample_drawing.py [--out sample_drawing.pdf]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A1
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

PAGE_W, PAGE_H = A1  # 594 x 841 mm

# The plan is authored in real-world millimetres and placed on the sheet at
# 1:100, so 6000 mm on the drawing becomes 60 mm of paper.
SHEET_SCALE = 1.0 / 100.0

# drawing is authored in millimetres on the sheet
ORIGIN_X, ORIGIN_Y = 40 * mm, 60 * mm
COL_SPACING_X = 6000        # mm between column centres
COL_SPACING_Y = 4500
COL_SIZE = 300
WALL_T = 230
SLAB_THK = 125
STOREYS = 2
BEAM_W, BEAM_D = 230, 450


def mm_on_sheet(real_mm: float) -> float:
    """Real-world millimetres -> points on the sheet at 1:100."""
    return real_mm * mm * SHEET_SCALE


def col(x_mm: float, y_mm: float) -> tuple[float, float]:
    return ORIGIN_X + mm_on_sheet(x_mm), ORIGIN_Y + mm_on_sheet(y_mm)


def draw_plan(c: canvas.Canvas) -> None:
    c.setLineWidth(0.6)
    c.setStrokeColor(colors.HexColor("#1b3a5c"))
    c.setFillColor(colors.HexColor("#1b3a5c"))

    cols_x = [0, COL_SPACING_X, COL_SPACING_X * 2, COL_SPACING_X * 3]
    cols_y = [0, COL_SPACING_Y, COL_SPACING_Y * 2]

    # ---- structural grid ------------------------------------------- #
    c.setStrokeColor(colors.HexColor("#9a9a9a"))
    c.setLineWidth(0.25)
    c.setDash(4, 2)
    for i, x in enumerate(cols_x):
        px, _ = col(x, 0)
        c.line(px, ORIGIN_Y - 12 * mm, px, ORIGIN_Y + mm_on_sheet(COL_SPACING_Y * 2) + 12 * mm)
        c.setFont("Helvetica", 7)
        c.drawCentredString(px, ORIGIN_Y - 18 * mm, f"({i+1})")
    for j, y in enumerate(cols_y):
        _, py = col(0, y)
        c.line(ORIGIN_X - 12 * mm, py, ORIGIN_X + mm_on_sheet(COL_SPACING_X * 3) + 12 * mm, py)
        c.setFont("Helvetica", 7)
        c.drawRightString(ORIGIN_X - 14 * mm, py - 2, f"({chr(65+j)})")
    c.setDash()

    # ---- slabs / room outlines -------------------------------------- #
    c.setStrokeColor(colors.HexColor("#1b3a5c"))
    c.setLineWidth(0.9)
    for j in range(len(cols_y) - 1):
        for i in range(len(cols_x) - 1):
            x0, y0 = col(cols_x[i], cols_y[j])
            x1, y1 = col(cols_x[i + 1], cols_y[j + 1])
            c.rect(x0, y0, x1 - x0, y1 - y0, stroke=1, fill=0)

    # ---- walls (double line convention) ------------------------------ #
    c.setLineWidth(0.7)
    c.setStrokeColor(colors.HexColor("#7a2020"))
    for j in range(len(cols_y) - 1):
        for i in range(len(cols_x) - 1):
            x0, y0 = col(cols_x[i], cols_y[j])
            x1, y1 = col(cols_x[i + 1], cols_y[j + 1])
            half = mm_on_sheet(WALL_T / 2)
            c.line(x0, y0 + half, x1, y0 + half)
            c.line(x0, y0 - half, x1, y0 - half)
            c.line(x0 + half, y0, x0 + half, y1)
            c.line(x1 - half, y0, x1 - half, y1)

    # ---- columns ------------------------------------------------------ #
    c.setFillColor(colors.HexColor("#d4d4d4"))
    c.setStrokeColor(colors.HexColor("#1b3a5c"))
    c.setLineWidth(0.8)
    for x in cols_x:
        for y in cols_y:
            cx, cy = col(x, y)
            s = mm_on_sheet(COL_SIZE / 2)
            c.rect(cx - s, cy - s, 2 * s, 2 * s, stroke=1, fill=1)

    # ---- beams (single heavy line on the grid) ------------------------ #
    c.setStrokeColor(colors.HexColor("#1f6f3f"))
    c.setLineWidth(2.0)
    for y in cols_y:
        _, py = col(0, y)
        c.line(ORIGIN_X, py, ORIGIN_X + mm_on_sheet(COL_SPACING_X * 3), py)
    for x in cols_x:
        px, _ = col(x, 0)
        c.line(px, ORIGIN_Y, px, ORIGIN_Y + mm_on_sheet(COL_SPACING_Y * 2))

    # ---- dimension lines ---------------------------------------------- #
    c.setStrokeColor(colors.HexColor("#444444"))
    c.setFillColor(colors.HexColor("#444444"))
    c.setFont("Helvetica", 7)
    dim_y = ORIGIN_Y - 26 * mm
    c.line(ORIGIN_X, dim_y, ORIGIN_X + mm_on_sheet(COL_SPACING_X * 3), dim_y)
    for i, x in enumerate(cols_x):
        px, _ = col(x, 0)
        c.setStrokeColor(colors.HexColor("#888888"))
        c.line(px, dim_y - 3 * mm, px, dim_y + 3 * mm)
        c.setFillColor(colors.HexColor("#444444"))
        c.drawCentredString(px, dim_y + 2 * mm, str(COL_SPACING_X))
    dim_x = ORIGIN_X + mm_on_sheet(COL_SPACING_X * 3) + 26 * mm
    c.setStrokeColor(colors.HexColor("#444444"))
    c.line(dim_x, ORIGIN_Y, dim_x, ORIGIN_Y + mm_on_sheet(COL_SPACING_Y * 2))
    for y in cols_y:
        _, py = col(0, y)
        c.setStrokeColor(colors.HexColor("#888888"))
        c.line(dim_x - 3 * mm, py, dim_x + 3 * mm, py)
        c.setFillColor(colors.HexColor("#444444"))
        c.saveState(); c.translate(dim_x + 4 * mm, py); c.rotate(90)
        c.drawCentredString(0, 0, str(COL_SPACING_Y)); c.restoreState()

    # ---- labels --------------------------------------------------------- #
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#333333"))
    for j in range(len(cols_y) - 1):
        for i in range(len(cols_x) - 1):
            x0, y0 = col(cols_x[i], cols_y[j])
            c.drawCentredString((x0 + col(cols_x[i + 1], 0)[0]) / 2, (y0 + col(0, cols_y[j + 1])[1]) / 2,
                                f"ROOM {chr(65+j)}{i+1}")


def draw_title_block(c: canvas.Canvas) -> None:
    bw, bh = 190 * mm, 42 * mm
    x0, y0 = PAGE_W - bw - 12 * mm, 12 * mm
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.0)
    c.rect(x0, y0, bw, bh)
    c.setLineWidth(0.5)
    c.line(x0, y0 + 14 * mm, x0 + bw, y0 + 14 * mm)
    c.line(x0 + 95 * mm, y0, x0 + 95 * mm, y0 + 14 * mm)
    c.line(x0 + 142 * mm, y0, x0 + 142 * mm, y0 + 14 * mm)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0 + 4 * mm, y0 + 30 * mm, "RESIDENTIAL BLOCK - TYPICAL FLOOR PLAN")
    c.setFont("Helvetica", 8)
    c.drawString(x0 + 4 * mm, y0 + 24 * mm, "PROJECT: SAMPLE APARTMENT SCHEME")
    c.drawString(x0 + 4 * mm, y0 + 19 * mm, "DWG NO: CA/FLR/001")

    c.setFont("Helvetica-Bold", 9)
    c.drawString(x0 + 98 * mm, y0 + 9 * mm, "SCALE 1:100")
    c.setFont("Helvetica", 8)
    c.drawString(x0 + 98 * mm, y0 + 4 * mm, "ALL DIMENSIONS IN MM")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x0 + 145 * mm, y0 + 9 * mm, "STOREYS: G+1")
    c.setFont("Helvetica", 8)
    c.drawString(x0 + 145 * mm, y0 + 4 * mm, f"SLAB: {SLAB_THK} MM")


def draw_schedule(c: canvas.Canvas) -> None:
    rows = [
        ("SLAB", "D12", "1500", "240", "150 c/c"),
        ("SLAB", "D16", "4200", "96", "180 c/c"),
        ("BEAM", "D16", "5800", "48", "200 c/c"),
        ("BEAM", "D12", "3600", "72", "150 c/c"),
        ("COLUMN", "D20", "3400", "36", "200 c/c"),
        ("COLUMN", "D12", "3400", "96", "150 c/c"),
        ("FOOTING", "D25", "2600", "24", "200 c/c"),
        ("FOOTING", "D16", "1800", "40", "150 c/c"),
    ]
    headers = ("MEMBER", "DIA", "LENGTH MM", "NOS", "SPACING")
    x0 = 14 * mm
    y0 = PAGE_H - 120 * mm
    col_w = (26, 18, 30, 20, 28)
    row_h = 6 * mm

    total_w = sum(col_w) * mm
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)
    c.rect(x0, y0, total_w, row_h * (len(rows) + 1))

    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.HexColor("#e8e8e8"))
    c.rect(x0, y0 + row_h * len(rows), total_w, row_h, stroke=0, fill=1)
    c.setFillColor(colors.black)
    cx = x0
    for i, h in enumerate(headers):
        c.drawString(cx + 2 * mm, y0 + row_h * len(rows) + 2 * mm, h)
        cx += col_w[i] * mm

    c.setFont("Helvetica", 8)
    for r, row in enumerate(rows):
        ry = y0 + row_h * (len(rows) - 1 - r)
        cx = x0
        for i, cell in enumerate(row):
            c.drawString(cx + 2 * mm, ry + 2 * mm, cell)
            cx += col_w[i] * mm

    cx = x0
    for w in col_w[:-1]:
        cx += w * mm
        c.line(cx, y0, cx, y0 + row_h * (len(rows) + 1))
    for r in range(len(rows) + 1):
        c.line(x0, y0 + r * row_h, x0 + total_w, y0 + r * row_h)

    c.setFont("Helvetica-Bold", 9)
    c.drawString(x0, y0 - 6 * mm, "REBAR SCHEDULE")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a vector test drawing")
    ap.add_argument("--out", default="sample_drawing.pdf")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out), pagesize=A1)
    c.setTitle("Residential Block - Typical Floor Plan")
    draw_plan(c)
    draw_schedule(c)
    draw_title_block(c)
    c.showPage()
    c.save()
    print(f"Wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
