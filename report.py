"""
modules/report.py
-----------------
PDF report generation using ReportLab.
Produces a downloadable, professional candidate screening report.
"""

import io
import logging
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)


def generate_pdf_report(
    ranked_candidates: List[Dict],
    jd_text: str,
    jd_skills: Dict,
    stats: Dict,
) -> bytes:
    """
    Generate a comprehensive PDF screening report.
    
    Args:
        ranked_candidates: Sorted list of scored candidate dicts.
        jd_text: Original job description text.
        jd_skills: {skill: weight} extracted from JD.
        stats: Aggregate statistics dict from ranker.compute_aggregate_stats().
        
    Returns:
        PDF as bytes (ready for st.download_button).
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        logger.error("ReportLab not installed. Install with: pip install reportlab")
        raise
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )
    
    # ---------------------------------------------------------------------------
    # Define styles
    # ---------------------------------------------------------------------------
    styles = getSampleStyleSheet()
    
    INDIGO = colors.HexColor("#4F46E5")
    EMERALD = colors.HexColor("#10B981")
    AMBER = colors.HexColor("#F59E0B")
    RED = colors.HexColor("#EF4444")
    LIGHT_GRAY = colors.HexColor("#F3F4F6")
    DARK = colors.HexColor("#1F2937")
    MUTED = colors.HexColor("#6B7280")
    
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=INDIGO,
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=MUTED,
        spaceAfter=16,
        alignment=TA_CENTER,
    )
    section_header_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=INDIGO,
        spaceBefore=14,
        spaceAfter=6,
        borderPad=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        textColor=DARK,
        spaceAfter=4,
        leading=13,
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=8,
        textColor=MUTED,
        spaceAfter=2,
    )
    
    elements = []
    
    # ---------------------------------------------------------------------------
    # Report Header
    # ---------------------------------------------------------------------------
    elements.append(Spacer(1, 0.5*cm))
    elements.append(Paragraph("🎯 AI Resume Screening Report", title_style))
    elements.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | "
        f"{stats.get('total_candidates', 0)} Candidates Evaluated",
        subtitle_style,
    ))
    elements.append(HRFlowable(width="100%", color=INDIGO, thickness=2, spaceAfter=12))
    
    # ---------------------------------------------------------------------------
    # Executive Summary
    # ---------------------------------------------------------------------------
    elements.append(Paragraph("Executive Summary", section_header_style))
    
    summary_data = [
        ["Metric", "Value"],
        ["Total Candidates", str(stats.get("total_candidates", 0))],
        ["Weak Candidates Flagged", str(stats.get("weak_candidates", 0))],
        ["Average Final Score", f"{round(stats.get('avg_final', 0) * 100, 1)}%"],
        ["Highest Score", f"{round(stats.get('max_final', 0) * 100, 1)}%"],
        ["Lowest Score", f"{round(stats.get('min_final', 0) * 100, 1)}%"],
        ["Skills Required", str(len(jd_skills))],
    ]
    
    summary_table = Table(summary_data, colWidths=[8*cm, 9*cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INDIGO),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.5*cm))
    
    # ---------------------------------------------------------------------------
    # Top Candidate Highlight
    # ---------------------------------------------------------------------------
    if ranked_candidates:
        top = ranked_candidates[0]
        top_name = top["name"].replace(".pdf", "").replace(".txt", "")
        elements.append(Paragraph("🏆 Top Candidate", section_header_style))
        elements.append(Paragraph(
            f"<b>{top_name}</b> achieved the highest final score of "
            f"<b>{round(top['final_score'] * 100, 1)}%</b> with "
            f"{len(top['matched_skills'])} matched skills and "
            f"{top['experience_years']} years of experience.",
            body_style,
        ))
        elements.append(Spacer(1, 0.3*cm))
    
    # ---------------------------------------------------------------------------
    # Candidate Rankings Table
    # ---------------------------------------------------------------------------
    elements.append(Paragraph("Candidate Rankings", section_header_style))
    
    rank_headers = [
        "Rank", "Candidate", "Final", "Skill", "Similarity",
        "Experience", "Matched", "Missing", "Status"
    ]
    rank_data = [rank_headers]
    
    for c in ranked_candidates:
        name = c["name"].replace(".pdf", "").replace(".txt", "")
        if len(name) > 18:
            name = name[:15] + "..."
        status = "⚠ Weak" if c.get("is_weak", {}).get("is_weak") else "Good"
        rank_data.append([
            f"#{c['rank']}",
            name,
            f"{round(c['final_score']*100, 1)}%",
            f"{round(c['skill_score']*100, 1)}%",
            f"{round(c['similarity_score']*100, 1)}%",
            f"{c['experience_years']} yrs",
            str(len(c['matched_skills'])),
            str(len(c['missing_skills'])),
            status,
        ])
    
    col_widths = [1.5*cm, 4*cm, 2*cm, 2*cm, 2.5*cm, 2.5*cm, 1.8*cm, 1.8*cm, 1.8*cm]
    rank_table = Table(rank_data, colWidths=col_widths)
    rank_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INDIGO),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (1, 1), (1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(rank_table)
    elements.append(PageBreak())
    
    # ---------------------------------------------------------------------------
    # Individual Candidate Reports
    # ---------------------------------------------------------------------------
    elements.append(Paragraph("Individual Candidate Reports", section_header_style))
    
    for c in ranked_candidates:
        name = c["name"].replace(".pdf", "").replace(".txt", "")
        medal = c.get("medal", "")
        
        elements.append(HRFlowable(width="100%", color=colors.HexColor("#E5E7EB"), thickness=1))
        elements.append(Spacer(1, 0.2*cm))
        
        # Candidate header
        rank_color = INDIGO if c["rank"] <= 3 else MUTED
        elements.append(Paragraph(
            f"{medal} <b>#{c['rank']} — {name}</b>  "
            f"<font color='#{INDIGO.hexval()[2:]}'>Final Score: {round(c['final_score']*100, 1)}%</font>",
            ParagraphStyle("CandHeader", parent=styles["Heading3"],
                           fontSize=12, textColor=DARK, spaceBefore=8, spaceAfter=4)
        ))
        
        # Score breakdown mini-table
        score_data = [
            ["Component", "Raw Score", "Weight", "Contribution"],
            ["Skill Match", f"{round(c['skill_score']*100, 1)}%", "50%", f"{round(c['weighted_skill']*100, 1)}%"],
            ["Semantic Similarity", f"{round(c['similarity_score']*100, 1)}%", "30%", f"{round(c['weighted_similarity']*100, 1)}%"],
            ["Experience", f"{round(c['experience_score']*100, 1)}%", "20%", f"{round(c['weighted_experience']*100, 1)}%"],
            ["FINAL SCORE", "", "", f"{round(c['final_score']*100, 1)}%"],
        ]
        score_table = Table(score_data, colWidths=[5*cm, 3*cm, 3*cm, 3*cm])
        score_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0E7FF")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 4), (-1, 4), INDIGO),
            ("TEXTCOLOR", (0, 4), (-1, 4), colors.white),
            ("FONTNAME", (0, 4), (-1, 4), "Helvetica-Bold"),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(score_table)
        elements.append(Spacer(1, 0.2*cm))
        
        # Explanation
        if c.get("explanation"):
            elements.append(Paragraph("Analysis:", small_style))
            for line in c["explanation"].split("\n"):
                if line.strip():
                    elements.append(Paragraph(line.strip(), body_style))
        
        # Matched skills
        if c["matched_skills"]:
            matched_str = "  •  ".join(c["matched_skills"][:15])
            if len(c["matched_skills"]) > 15:
                matched_str += f"  +{len(c['matched_skills'])-15} more"
            elements.append(Paragraph(f"<b>✔ Matched Skills:</b> {matched_str}", body_style))
        
        # Missing skills
        if c["missing_skills"]:
            missing_str = "  •  ".join(c["missing_skills"][:10])
            if len(c["missing_skills"]) > 10:
                missing_str += f"  +{len(c['missing_skills'])-10} more"
            elements.append(Paragraph(
                f"<b>✘ Missing Skills:</b> <font color='#EF4444'>{missing_str}</font>",
                body_style,
            ))
        
        # Weak candidate warning
        weak_data = c.get("is_weak", {})
        if weak_data.get("is_weak"):
            elements.append(Paragraph(
                f"⚠️ <b>Weak Candidate:</b> {' | '.join(weak_data.get('reasons', []))}",
                ParagraphStyle("Warn", parent=body_style, textColor=AMBER),
            ))
        
        elements.append(Spacer(1, 0.4*cm))
    
    # ---------------------------------------------------------------------------
    # Footer
    # ---------------------------------------------------------------------------
    elements.append(HRFlowable(width="100%", color=INDIGO, thickness=1))
    elements.append(Paragraph(
        "Generated by AI Resume Screening System | Confidential HR Document",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7,
                       textColor=MUTED, alignment=TA_CENTER, spaceBefore=4),
    ))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.read()
