"""
Report Generation Module - Chapter 4.4
Uses Python ReportLab (Platypus) to generate professional PDF meeting summaries.
Sections: Meeting Overview, Executive Summary, Key Decisions, Action Items, Discussion Highlights.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)


# ─── Color Palette ──────────────────────────────────────────────────────────

PRIMARY = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#16213e")
BLUE = colors.HexColor("#0f3460")
LIGHT_BLUE = colors.HexColor("#e8f4f8")
HEADER_BG = colors.HexColor("#1a1a2e")
ROW_ALT = colors.HexColor("#f5f8fa")
SUCCESS = colors.HexColor("#27ae60")
WARNING = colors.HexColor("#e67e22")
BORDER = colors.HexColor("#d0d7de")
WHITE = colors.white
TEXT_DARK = colors.HexColor("#1a1a2e")
TEXT_MUTED = colors.HexColor("#57606a")


# ─── Styles ─────────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["title"] = ParagraphStyle(
        "title", fontName="Helvetica-Bold", fontSize=22,
        textColor=WHITE, alignment=TA_CENTER, spaceAfter=4
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle", fontName="Helvetica", fontSize=11,
        textColor=colors.HexColor("#a8c8d8"), alignment=TA_CENTER, spaceAfter=2
    )
    styles["section_header"] = ParagraphStyle(
        "section_header", fontName="Helvetica-Bold", fontSize=13,
        textColor=BLUE, spaceBefore=16, spaceAfter=6
    )
    styles["body"] = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=10,
        textColor=TEXT_DARK, leading=15, alignment=TA_JUSTIFY, spaceAfter=6
    )
    styles["bullet"] = ParagraphStyle(
        "bullet", fontName="Helvetica", fontSize=10,
        textColor=TEXT_DARK, leading=14, leftIndent=12, spaceAfter=4,
        bulletIndent=0
    )
    styles["table_header"] = ParagraphStyle(
        "table_header", fontName="Helvetica-Bold", fontSize=10,
        textColor=WHITE, alignment=TA_CENTER
    )
    styles["table_cell"] = ParagraphStyle(
        "table_cell", fontName="Helvetica", fontSize=9,
        textColor=TEXT_DARK, leading=13
    )
    styles["table_cell_muted"] = ParagraphStyle(
        "table_cell_muted", fontName="Helvetica", fontSize=9,
        textColor=TEXT_MUTED, leading=13
    )
    styles["meta"] = ParagraphStyle(
        "meta", fontName="Helvetica", fontSize=9,
        textColor=TEXT_MUTED, alignment=TA_CENTER
    )
    return styles


# ─── PDF Generator ──────────────────────────────────────────────────────────

def generate_pdf_report(report_data: dict, job_id: str, output_folder: str) -> str:
    """
    Generate a professional PDF meeting summary using ReportLab Platypus.

    Args:
        report_data: Compiled output from the LangChain agent pipeline.
        job_id: Unique job identifier (used for filename).
        output_folder: Directory to save the PDF.

    Returns:
        Path to the generated PDF file.
    """
    output_path = os.path.join(output_folder, f"{job_id}_report.pdf")
    styles = build_styles()
    story = []

    page_w, page_h = A4
    margin = 2 * cm

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
        title="Meeting Summary Report",
    )

    content_width = page_w - 2 * margin

    # ── Header Banner ─────────────────────────────────────────────────────
    now = datetime.now().strftime("%B %d, %Y  •  %I:%M %p")
    attendees_str = ", ".join(report_data.get("attendees", []))

    header_data = [[
        Paragraph("MEETING SUMMARY REPORT", styles["title"]),
    ]]
    header_table = Table(header_data, colWidths=[content_width])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HEADER_BG),
        ("TOPPADDING", (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 4))

    # Meta info row
    meta_data = [[
        Paragraph(f"Generated: {now}", styles["meta"]),
        Paragraph(f"Attendees: {attendees_str}", styles["meta"]),
    ]]
    meta_table = Table(meta_data, colWidths=[content_width / 2, content_width / 2])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # ── Executive Summary ─────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", styles["section_header"]))
    story.append(HRFlowable(width=content_width, thickness=1, color=BLUE, spaceAfter=6))
    summary_text = report_data.get("summary", "No summary available.")
    story.append(Paragraph(summary_text, styles["body"]))
    story.append(Spacer(1, 10))

    # ── Key Decisions ─────────────────────────────────────────────────────
    decisions = report_data.get("decisions", [])
    if decisions:
        story.append(Paragraph("Key Decisions", styles["section_header"]))
        story.append(HRFlowable(width=content_width, thickness=1, color=BLUE, spaceAfter=6))

        decision_rows = [
            [
                Paragraph("#", styles["table_header"]),
                Paragraph("Decision", styles["table_header"]),
            ]
        ]
        for i, decision in enumerate(decisions, 1):
            decision_rows.append([
                Paragraph(str(i), styles["table_cell_muted"]),
                Paragraph(decision, styles["table_cell"]),
            ])

        decision_table = Table(
            decision_rows,
            colWidths=[1.2 * cm, content_width - 1.2 * cm],
        )
        decision_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, ROW_ALT]),
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(decision_table)
        story.append(Spacer(1, 14))

    # ── Action Items ──────────────────────────────────────────────────────
    action_items = report_data.get("action_items", [])
    if action_items:
        story.append(Paragraph("Action Items", styles["section_header"]))
        story.append(HRFlowable(width=content_width, thickness=1, color=BLUE, spaceAfter=6))

        col_w = [content_width * 0.50, content_width * 0.25, content_width * 0.25]
        action_rows = [[
            Paragraph("Task", styles["table_header"]),
            Paragraph("Owner", styles["table_header"]),
            Paragraph("Deadline", styles["table_header"]),
        ]]
        for item in action_items:
            action_rows.append([
                Paragraph(item.get("task", ""), styles["table_cell"]),
                Paragraph(item.get("owner", ""), styles["table_cell_muted"]),
                Paragraph(item.get("deadline", "Not specified"), styles["table_cell_muted"]),
            ])

        action_table = Table(action_rows, colWidths=col_w)
        action_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, ROW_ALT]),
            ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(action_table)
        story.append(Spacer(1, 14))

    # ── Cleaned Transcript ────────────────────────────────────────────────
    transcript = report_data.get("transcript", "")
    if transcript:
        story.append(Paragraph("Meeting Transcript", styles["section_header"]))
        story.append(HRFlowable(width=content_width, thickness=1, color=BLUE, spaceAfter=6))

        for line in transcript.strip().split("\n\n")[:30]:  # cap at 30 turns
            if line.strip():
                story.append(Paragraph(line.strip(), styles["bullet"]))

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width=content_width, thickness=0.5, color=BORDER))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Generated by Autonomous Meeting Summarization System  •  "
        "Whisper ASR  |  PyAnnote Diarization  |  LangChain Agents  |  ReportLab PDF",
        styles["meta"]
    ))

    doc.build(story)
    print(f"[ReportLab] PDF report saved: {output_path}")
    return output_path
