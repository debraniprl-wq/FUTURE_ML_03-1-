"""
modules/visualizer.py
----------------------
Plotly-based chart generation for the resume screening dashboard.
All charts use the premium light-theme color palette.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (matches the app's premium light theme)
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#4F46E5",       # Indigo
    "success": "#10B981",       # Emerald
    "warning": "#F59E0B",       # Amber
    "danger": "#EF4444",        # Red
    "secondary": "#6366F1",     # Lighter indigo
    "light_bg": "#F8F9FF",
    "grid": "#E5E7EB",
    "text": "#1F2937",
    "muted": "#6B7280",
}

SCORE_COLORS = {
    "Skill Score": "#4F46E5",
    "Similarity Score": "#10B981",
    "Experience Score": "#F59E0B",
    "Final Score": "#7C3AED",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor=COLORS["light_bg"],
    font=dict(family="Inter, sans-serif", color=COLORS["text"]),
    # margin is intentionally omitted here so each chart can set its own
)


def candidate_comparison_bar(ranked_candidates: List[Dict]) -> go.Figure:
    """
    Grouped bar chart comparing all candidates across all score dimensions.
    
    Args:
        ranked_candidates: Sorted list of scored candidate dicts.
        
    Returns:
        Plotly Figure object.
    """
    # Prepare data
    names = []
    for c in ranked_candidates:
        name = c["name"].replace(".pdf", "").replace(".txt", "")
        if len(name) > 20:
            name = name[:17] + "..."
        medal = c.get("medal", "")
        names.append(f"{medal} {name}".strip())
    
    skill_scores = [round(c["skill_score"] * 100, 1) for c in ranked_candidates]
    sim_scores = [round(c["similarity_score"] * 100, 1) for c in ranked_candidates]
    exp_scores = [round(c["experience_score"] * 100, 1) for c in ranked_candidates]
    final_scores = [round(c["final_score"] * 100, 1) for c in ranked_candidates]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Skill Score",
        x=names,
        y=skill_scores,
        marker_color=SCORE_COLORS["Skill Score"],
        opacity=0.9,
        text=[f"{s}%" for s in skill_scores],
        textposition="outside",
        textfont=dict(size=10),
    ))
    
    fig.add_trace(go.Bar(
        name="Similarity Score",
        x=names,
        y=sim_scores,
        marker_color=SCORE_COLORS["Similarity Score"],
        opacity=0.9,
        text=[f"{s}%" for s in sim_scores],
        textposition="outside",
        textfont=dict(size=10),
    ))
    
    fig.add_trace(go.Bar(
        name="Experience Score",
        x=names,
        y=exp_scores,
        marker_color=SCORE_COLORS["Experience Score"],
        opacity=0.9,
        text=[f"{s}%" for s in exp_scores],
        textposition="outside",
        textfont=dict(size=10),
    ))
    
    # Overlay line for final score
    fig.add_trace(go.Scatter(
        name="Final Score",
        x=names,
        y=final_scores,
        mode="lines+markers+text",
        line=dict(color=SCORE_COLORS["Final Score"], width=3, dash="dot"),
        marker=dict(size=10, color=SCORE_COLORS["Final Score"]),
        text=[f"{s}%" for s in final_scores],
        textposition="top center",
        textfont=dict(size=11, color=SCORE_COLORS["Final Score"]),
    ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="<b>Candidate Score Comparison</b>",
            x=0.5,
            font=dict(size=16),
        ),
        barmode="group",
        margin=dict(l=20, r=20, t=60, b=20),
        yaxis=dict(
            title="Score (%)",
            range=[0, 115],
            gridcolor=COLORS["grid"],
            gridwidth=1,
        ),
        xaxis=dict(title="Candidates"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        bargap=0.15,
        bargroupgap=0.05,
    )
    
    return fig


def score_breakdown_donut(candidate: Dict) -> go.Figure:
    """
    Donut chart showing the weighted contribution of each score component
    to the final score for a single candidate.
    """
    labels = ["Skill Score (50%)", "Similarity Score (30%)", "Experience Score (20%)"]
    values = [
        candidate["weighted_skill"] * 100,
        candidate["weighted_similarity"] * 100,
        candidate["weighted_experience"] * 100,
    ]
    colors = [
        SCORE_COLORS["Skill Score"],
        SCORE_COLORS["Similarity Score"],
        SCORE_COLORS["Experience Score"],
    ]
    
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>Contribution: %{value:.2f}%<extra></extra>",
    ))
    
    # Annotation in center
    final_pct = round(candidate["final_score"] * 100, 1)
    fig.add_annotation(
        text=f"<b>{final_pct}%</b><br><span style='font-size:10px'>Final</span>",
        x=0.5, y=0.5,
        font=dict(size=18, color=COLORS["primary"]),
        showarrow=False,
        align="center",
    )
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="<b>Score Breakdown</b>",
            x=0.5,
            font=dict(size=14),
        ),
        showlegend=True,
        legend=dict(orientation="v", x=1.0, y=0.5),
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig


def skill_match_horizontal_bar(candidate: Dict, jd_skills: Dict) -> go.Figure:
    """
    Horizontal bar chart showing matched (green) vs missing (red) skills
    for a single candidate, sorted by JD skill weight.
    """
    matched = set(candidate["matched_skills"])
    all_skills = sorted(jd_skills.keys(), key=lambda s: jd_skills[s], reverse=True)
    
    # Limit to top 20 skills for readability
    all_skills = all_skills[:20]
    
    colors = [
        COLORS["success"] if s in matched else COLORS["danger"]
        for s in all_skills
    ]
    
    status = [
        "✔ Matched" if s in matched else "✘ Missing"
        for s in all_skills
    ]
    
    weights = [jd_skills.get(s, 0) * 100 for s in all_skills]
    
    fig = go.Figure(go.Bar(
        x=weights,
        y=all_skills,
        orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=1)),
        text=status,
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.0f}%<br>%{text}<extra></extra>",
    ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="<b>Skill Match Analysis</b>",
            x=0.5,
            font=dict(size=14),
        ),
        xaxis=dict(
            title="Skill Importance (%)",
            range=[0, 130],
            gridcolor=COLORS["grid"],
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=11),
        ),
        height=max(300, len(all_skills) * 28),
        margin=dict(l=10, r=80, t=50, b=20),
    )
    
    return fig


def leaderboard_funnel(ranked_candidates: List[Dict]) -> go.Figure:
    """
    Horizontal funnel-style final score chart for the leaderboard overview.
    """
    names = []
    for c in ranked_candidates:
        name = c["name"].replace(".pdf", "").replace(".txt", "")
        if len(name) > 25:
            name = name[:22] + "..."
        medal = c.get("medal", "")
        names.append(f"#{c['rank']} {medal} {name}".strip())
    
    scores = [round(c["final_score"] * 100, 1) for c in ranked_candidates]
    
    # Color gradient from top to bottom
    n = len(ranked_candidates)
    bar_colors = []
    for i in range(n):
        if i == 0:
            bar_colors.append(COLORS["primary"])
        elif i == 1:
            bar_colors.append(COLORS["secondary"])
        elif i == 2:
            bar_colors.append(COLORS["success"])
        else:
            # Fade to muted for lower ranks
            bar_colors.append(COLORS["muted"])
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=names,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="white", width=1)),
        text=[f"{s}%" for s in scores],
        textposition="outside",
        textfont=dict(size=12, color=COLORS["text"]),
        hovertemplate="<b>%{y}</b><br>Final Score: %{x}%<extra></extra>",
    ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="<b>Candidate Leaderboard — Final Scores</b>",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Final Score (%)",
            range=[0, 120],
            gridcolor=COLORS["grid"],
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        height=max(350, n * 50),
        margin=dict(l=10, r=60, t=60, b=30),
    )
    
    return fig


def experience_distribution(ranked_candidates: List[Dict]) -> go.Figure:
    """Scatter plot of experience years vs final score."""
    names = [c["name"].replace(".pdf", "").replace(".txt", "") for c in ranked_candidates]
    years = [c["experience_years"] for c in ranked_candidates]
    finals = [round(c["final_score"] * 100, 1) for c in ranked_candidates]
    skills = [round(c["skill_score"] * 100, 1) for c in ranked_candidates]
    
    fig = go.Figure(go.Scatter(
        x=years,
        y=finals,
        mode="markers+text",
        marker=dict(
            size=[max(10, s / 5) for s in skills],  # Size proportional to skill score
            color=finals,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Final Score %"),
            line=dict(color="white", width=1),
        ),
        text=names,
        textposition="top center",
        textfont=dict(size=9),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Experience: %{x} years<br>"
            "Final Score: %{y}%<extra></extra>"
        ),
    ))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="<b>Experience vs Final Score</b>",
            x=0.5,
            font=dict(size=14),
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(title="Years of Experience", gridcolor=COLORS["grid"]),
        yaxis=dict(title="Final Score (%)", gridcolor=COLORS["grid"]),
        height=350,
    )
    
    return fig
