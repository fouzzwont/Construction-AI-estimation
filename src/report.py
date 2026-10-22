"""Client facing reports: CSV take-off sheet and a PDF summary."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .estimator import EstimateResult

logger = logging.getLogger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as RLImage,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    HAVE_REPORTLAB = True
except Exception:  # pragma: no cover
    HAVE_REPORTLAB = False


# --------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------- #
def write_csv(result: EstimateResult, output_path: str | Path) -> str:
    """Write the BOQ, rebar breakdown and page summary into one CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sym = result.currency_symbol

    with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)

        writer.writerow(["CONSTRUCTION AI - QUANTITY TAKE-OFF & COST ESTIMATE"])
        writer.writerow(["Project", result.project_name])
        writer.writerow(["Source drawing", result.source_file])
        writer.writerow(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")])
        writer.writerow(["Drawing scale", result.scale.get("ratio", "n/a")])
        writer.writerow(["Storeys", result.storeys])
        writer.writerow(["Market", result.market])
        writer.writerow([])

        writer.writerow(["BILL OF QUANTITIES"])
        writer.writerow(["Category", "Item", "Quantity", "Unit", "Rate", "Amount", "Details"])
        for line in result.boq:
            writer.writerow(
                [
                    line["category"],
                    line["item"],
                    line["quantity"],
                    line["unit"],
                    f"{sym}{line['rate']:,.2f}",
                    f"{sym}{line['amount']:,.2f}",
                    line.get("detail", ""),
                ]
            )
        writer.writerow([])
        writer.writerow(["Category", "Total"])
        for category, amount in result.totals.items():
            writer.writerow([category, f"{sym}{amount:,.2f}"])
        writer.writerow(["GRAND TOTAL", f"{sym}{result.grand_total:,.2f}"])
        writer.writerow([])

        if result.rebar:
            writer.writerow(["REINFORCEMENT SCHEDULE SUMMARY"])
            writer.writerow(["Diameter (mm)", "Total length (m)", "Count", "Weight (kg)", "Weight (t)"])
            for row in result.rebar:
                writer.writerow(
                    [
                        row["diameter_mm"],
                        row["total_length_m"],
                        row["count"],
                        row["weight_kg"],
                        row["weight_t"],
                    ]
                )
            writer.writerow([])

        writer.writerow(["KEY QUANTITIES"])
        writer.writerow(["Parameter", "Value"])
        for key, value in result.quantities.items():
            writer.writerow([key, value])
        writer.writerow([])

        writer.writerow(["ELEMENT COUNTS"])
        writer.writerow(["Element", "Count"])
        for kind, count in result.counts.items():
            writer.writerow([kind, count])
        writer.writerow([])

        writer.writerow(["ASSUMPTIONS"])
        for note in result.assumptions:
            writer.writerow([note])
        if result.warnings:
            writer.writerow([])
            writer.writerow(["WARNINGS"])
            for note in result.warnings:
                writer.writerow([note])

    return str(output_path)


# --------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------- #
def write_pdf(
    result: EstimateResult,
    output_path: str | Path,
    page_images: list[str] | None = None,
) -> str | None:
    """Render a branded PDF summary of the estimate."""
    if not HAVE_REPORTLAB:  # pragma: no cover
        logger.warning("reportlab not installed - skipping PDF report")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sym = result.currency_symbol

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleX", parent=styles["Title"], fontSize=19, spaceAfter=4, textColor=colors.HexColor("#0f172a")
    )
    sub_style = ParagraphStyle(
        "SubX", parent=styles["Normal"], fontSize=9.5, textColor=colors.HexColor("#64748b"), spaceAfter=12
    )
    head_style = ParagraphStyle(
        "HeadX", parent=styles["Heading2"], fontSize=12, spaceBefore=14, spaceAfter=6,
        textColor=colors.HexColor("#1e293b"),
    )
    warn_style = ParagraphStyle(
        "WarnX", parent=styles["Normal"], fontSize=8.6, textColor=colors.HexColor("#92400e"),
        backColor=colors.HexColor("#fef3c7"), borderPadding=5, spaceAfter=3,
    )

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
    )
    story: list[Any] = []

    story.append(Paragraph("Construction Cost Estimate", title_style))
    story.append(
        Paragraph(
            f"{result.project_name} &nbsp;|&nbsp; {result.source_file} &nbsp;|&nbsp; "
            f"generated {datetime.now().strftime('%d %b %Y, %H:%M')}",
            sub_style,
        )
    )

    # ---- summary tiles ------------------------------------------------ #
    q = result.quantities
    tiles = [
        ["DRAWING SCALE", "STOREYS", "ELEMENTS", "CONFIDENCE"],
        [
            result.scale.get("ratio", "n/a"),
            str(result.storeys),
            str(int(q.get("elements", 0))),
            f"{result.confidence*100:.0f}%",
        ],
    ]
    t = Table(tiles, colWidths=[doc.width / 4] * 4)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 7.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 1), (-1, 1), 12),
                ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#1e293b")),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("TOPPADDING", (0, 1), (-1, 1), 8),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 10))

    # ---- key quantities ------------------------------------------------ #
    story.append(Paragraph("Key Quantities", head_style))
    qty_rows = [
        ["Parameter", "Quantity", "Unit"],
        ["Concrete", f"{q.get('concrete_m3', 0):,.2f}", "m3"],
        ["Reinforcement steel", f"{q.get('steel_t', 0):,.3f}", "t"],
        ["Formwork / shuttering", f"{q.get('formwork_m2', 0):,.2f}", "m2"],
        ["Masonry", f"{q.get('masonry_m3', 0):,.2f}", "m3"],
    ]
    story.append(_table(qty_rows, [doc.width * 0.5, doc.width * 0.25, doc.width * 0.25]))

    # ---- BOQ ----------------------------------------------------------- #
    story.append(Paragraph("Bill of Quantities", head_style))
    if result.boq:
        data = [["Category", "Item", "Qty", "Unit", "Rate", "Amount"]]
        for line in result.boq:
            data.append(
                [
                    line["category"],
                    line["item"],
                    f"{line['quantity']:,.2f}",
                    line["unit"],
                    f"{sym}{line['rate']:,.0f}",
                    f"{sym}{line['amount']:,.0f}",
                ]
            )
        data.append(["", "", "", "", "GRAND TOTAL", f"{sym}{result.grand_total:,.0f}"])
        widths = [doc.width * 0.13, doc.width * 0.31, doc.width * 0.11, doc.width * 0.08, doc.width * 0.17, doc.width * 0.20]
        t = _table(data, widths)
        t.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, len(data) - 1), (-1, len(data) - 1), "Helvetica-Bold"),
                    ("BACKGROUND", (0, len(data) - 1), (-1, len(data) - 1), colors.HexColor("#f1f5f9")),
                    ("ALIGN", (3, 1), (-1, -1), "RIGHT"),
                ]
            )
        )
        story.append(t)
    else:
        story.append(Paragraph("No priced items could be derived from this drawing.", sub_style))

    # ---- rebar --------------------------------------------------------- #
    if result.rebar:
        story.append(Paragraph("Reinforcement Schedule Summary", head_style))
        data = [["Diameter (mm)", "Total length (m)", "Nos", "Weight (t)"]]
        for row in result.rebar:
            data.append(
                [str(row["diameter_mm"]), f"{row['total_length_m']:,.1f}", str(row["count"]), f"{row['weight_t']:,.3f}"]
            )
        story.append(_table(data, [doc.width * 0.25] * 4))

    # ---- element counts ------------------------------------------------ #
    if result.counts:
        story.append(Paragraph("Detected Elements", head_style))
        data = [["Element", "Count"]] + [[k, str(v)] for k, v in result.counts.items()]
        story.append(_table(data, [doc.width * 0.5, doc.width * 0.5]))

    # ---- assumptions ---------------------------------------------------- #
    if result.assumptions:
        story.append(Paragraph("Assumptions & Basis of Estimate", head_style))
        for note in result.assumptions:
            story.append(Paragraph(f"&bull; {note}", styles["Normal"]))

    for note in result.warnings:
        story.append(Paragraph(f"Warning: {note}", warn_style))

    # ---- annotated pages ------------------------------------------------ #
    if page_images:
        story.append(PageBreak())
        story.append(Paragraph("Annotated Drawings", head_style))
        for idx, img in enumerate(page_images, start=1):
            p = Path(img)
            if not p.exists():
                continue
            story.append(Paragraph(f"Page {idx}", styles["Normal"]))
            try:
                img = RLImage(str(p))
                target_w = doc.width
                target_h = target_w * img.imageHeight / img.imageWidth
                max_h = doc.height - 30 * mm
                if target_h > max_h:
                    target_h = max_h
                    target_w = target_h * img.imageWidth / img.imageHeight
                img.drawWidth = target_w
                img.drawHeight = target_h
                story.append(img)
                story.append(Spacer(1, 8))
            except Exception as exc:
                logger.warning("Could not embed %s: %s", p, exc)

    doc.build(story)
    return str(output_path)


def _table(data: list[list[Any]], widths: list[float]):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.4),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    return t
