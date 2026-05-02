"""
app.py — AI Resume Screening System (Main Entry Point)
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import os

from modules.parser import parse_resume, deduplicate_resumes
from modules.preprocessor import preprocess_resume, preprocess_job_description
from modules.extractor import load_skills_db, build_skills_flat, extract_skills, extract_skills_from_jd, extract_experience_years
from modules.scorer import score_candidate, batch_similarity_scores
from modules.ranker import rank_candidates, get_skill_gap_analysis, get_comparison_table, get_top_candidate, compute_aggregate_stats
from modules import visualizer
from modules.report import generate_pdf_report

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #F8F9FF; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #4F46E5 0%, #7C3AED 100%) !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: white !important; }

/* Textarea — force dark bg so white text is visible */
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] .stTextArea textarea,
[data-testid="stSidebar"] .stTextArea > div > div > textarea {
    background-color: rgba(30, 20, 80, 0.55) !important;
    border: 1.5px solid rgba(255,255,255,0.35) !important;
    color: white !important;
    border-radius: 10px !important;
    caret-color: white !important;
}

/* Placeholder text visible */
[data-testid="stSidebar"] textarea::placeholder {
    color: rgba(255,255,255,0.5) !important;
}

/* File uploader area */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(30, 20, 80, 0.4) !important;
    border: 1.5px dashed rgba(255,255,255,0.35) !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: white !important; }
[data-testid="stSidebar"] label { color: rgba(255,255,255,0.85) !important; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #9333EA 100%);
    border-radius: 20px;
    padding: 36px 40px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 10px 40px rgba(79,70,229,0.3);
}
.hero-banner h1 { font-size: 2.2rem; font-weight: 800; margin: 0 0 6px; }
.hero-banner p { font-size: 1rem; opacity: 0.85; margin: 0; }

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    border: 1px solid #EEF0FF;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(79,70,229,0.15); }
.metric-card .metric-value { font-size: 2rem; font-weight: 800; color: #4F46E5; line-height: 1; }
.metric-card .metric-label { font-size: 0.8rem; color: #6B7280; margin-top: 6px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }

/* Section headers */
.section-header {
    display: flex; align-items: center; gap: 10px;
    font-size: 1.2rem; font-weight: 700; color: #1F2937;
    margin: 28px 0 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid #EEF0FF;
}

/* Rank card */
.rank-card {
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-left: 5px solid #4F46E5;
    transition: all 0.2s;
}
.rank-card:hover { box-shadow: 0 6px 24px rgba(79,70,229,0.18); transform: translateX(3px); }
.rank-card.rank-1 { border-left-color: #F59E0B; }
.rank-card.rank-2 { border-left-color: #9CA3AF; }
.rank-card.rank-3 { border-left-color: #CD7C2F; }
.rank-card.weak { border-left-color: #EF4444; }

/* Score bar */
.score-bar-wrap { background: #F3F4F6; border-radius: 99px; height: 8px; margin: 4px 0 10px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 99px; transition: width 0.6s ease; }

/* Skill chip */
.chip { display: inline-block; padding: 3px 10px; border-radius: 99px; font-size: 0.75rem; font-weight: 500; margin: 3px 2px; }
.chip-green { background: #D1FAE5; color: #065F46; }
.chip-red { background: #FEE2E2; color: #991B1B; }
.chip-amber { background: #FEF3C7; color: #92400E; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background: white; border-radius: 12px; padding: 6px; gap: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.stTabs [data-baseweb="tab"] { border-radius: 8px; font-weight: 600; color: #6B7280; padding: 8px 18px; }
.stTabs [aria-selected="true"] { background: #4F46E5 !important; color: white !important; }

/* Highlight box */
.info-box { background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 12px; padding: 14px 18px; margin: 10px 0; font-size: 0.9rem; color: #3730A3; }
.warning-box { background: #FEF3C7; border: 1px solid #FDE68A; border-radius: 12px; padding: 14px 18px; margin: 10px 0; font-size: 0.9rem; color: #92400E; }
.success-box { background: #D1FAE5; border: 1px solid #A7F3D0; border-radius: 12px; padding: 14px 18px; margin: 10px 0; font-size: 0.9rem; color: #065F46; }

/* Button */
div.stButton > button { background: linear-gradient(135deg, #4F46E5, #7C3AED); color: white; font-weight: 600; border: none; border-radius: 10px; padding: 10px 28px; font-size: 0.95rem; transition: all 0.2s; box-shadow: 0 4px 14px rgba(79,70,229,0.35); }
div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(79,70,229,0.45); }
</style>
""", unsafe_allow_html=True)

# ── Load skills DB ────────────────────────────────────────────────────────────
@st.cache_resource
def load_db():
    db = load_skills_db()
    return db, build_skills_flat(db)

skills_db, skills_flat = load_db()

# ── Helper (defined early so it can be used anywhere below) ──────────────────
def _get_category(skill: str, db: dict) -> str:
    """Return the category name for a skill from the skills DB."""
    for cat, skills in db.items():
        if cat == "synonyms":
            continue
        if isinstance(skills, dict) and skill in skills:
            return cat.replace("_", " ").title()
    return "General"

# ── Session state ─────────────────────────────────────────────────────────────
if "ranked" not in st.session_state:
    st.session_state.ranked = []
if "jd_skills" not in st.session_state:
    st.session_state.jd_skills = {}
if "jd_preprocessed" not in st.session_state:
    st.session_state.jd_preprocessed = {}
if "stats" not in st.session_state:
    st.session_state.stats = {}
if "duplicates" not in st.session_state:
    st.session_state.duplicates = []
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Resume Screener")
    st.markdown("**AI-Powered Candidate Ranking**")
    st.markdown("---")

    st.markdown("### 📂 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT resumes",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload multiple resumes to compare candidates"
    )

    st.markdown("### 📝 Job Description")
    jd_text_input = st.text_area(
        "Paste job description here",
        height=220,
        placeholder="Paste the full job description including required skills, responsibilities, and qualifications...",
        help="The more detailed, the better the matching"
    )

    st.markdown("---")
    run_btn = st.button("🚀 Analyze Candidates", use_container_width=True)
    
    if st.session_state.processing_done:
        st.markdown("---")
        st.markdown("### ⚙️ Scoring Weights")
        st.markdown("""
        | Component | Weight |
        |-----------|--------|
        | Skill Match | **50%** |
        | Similarity | **30%** |
        | Experience | **20%** |
        """)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🎯 AI Resume Screening System</h1>
    <p>Upload resumes · Paste a job description · Get explainable AI-powered candidate rankings instantly</p>
</div>
""", unsafe_allow_html=True)

# ── Processing ────────────────────────────────────────────────────────────────
if run_btn:
    if not uploaded_files:
        st.error("⚠️ Please upload at least one resume.")
    elif not jd_text_input.strip():
        st.error("⚠️ Please paste a job description.")
    else:
        with st.spinner("🔍 Analyzing resumes with AI..."):
            # 1. Parse all resumes
            parsed = []
            parse_errors = []
            for f in uploaded_files:
                result = parse_resume(f)
                if result["error"]:
                    parse_errors.append(f"{result['name']}: {result['error']}")
                else:
                    parsed.append(result)

            # 2. Deduplicate
            unique_resumes, duplicates = deduplicate_resumes(parsed)
            st.session_state.duplicates = duplicates

            # 3. Preprocess JD
            jd_pre = preprocess_job_description(jd_text_input)
            jd_skills = extract_skills_from_jd(jd_pre["expanded"], skills_flat)
            st.session_state.jd_preprocessed = jd_pre
            st.session_state.jd_skills = jd_skills

            # 4. Preprocess + extract from all resumes
            resume_data_list = []
            for r in unique_resumes:
                pre = preprocess_resume(r["text"])
                skills = extract_skills(pre["expanded"], skills_flat, pre["sections"])
                exp = extract_experience_years(pre["cleaned"])
                resume_data_list.append({
                    "name": r["name"],
                    "preprocessed": pre,
                    "skills": skills,
                    "experience": exp,
                })

            # 5. Batch similarity (efficient)
            resume_texts = [rd["preprocessed"]["expanded"] for rd in resume_data_list]
            batch_sims = batch_similarity_scores(resume_texts, jd_pre["expanded"])

            # 6. Score each candidate
            scored = []
            for i, rd in enumerate(resume_data_list):
                result = score_candidate(rd, jd_pre["expanded"], jd_skills, skills_flat)
                # Overwrite similarity with batch-computed value for consistency
                result["similarity_score"] = batch_sims[i]
                scored.append(result)

            # 7. Rank
            ranked = rank_candidates(scored)
            st.session_state.ranked = ranked
            st.session_state.stats = compute_aggregate_stats(ranked)
            st.session_state.processing_done = True

            if parse_errors:
                for err in parse_errors:
                    st.warning(f"⚠️ {err}")
            if duplicates:
                st.info(f"ℹ️ Removed {len(duplicates)} duplicate(s): {', '.join(duplicates)}")

        st.success(f"✅ Analyzed {len(ranked)} candidate(s) successfully!")

# ── Main Dashboard ────────────────────────────────────────────────────────────
if st.session_state.processing_done and st.session_state.ranked:
    ranked = st.session_state.ranked
    stats = st.session_state.stats
    jd_skills = st.session_state.jd_skills
    jd_pre = st.session_state.jd_preprocessed
    top = get_top_candidate(ranked)

    # ── KPI Metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, str(stats["total_candidates"]), "Candidates"),
        (c2, f"{round(stats['max_final']*100,1)}%", "Top Score"),
        (c3, f"{round(stats['avg_final']*100,1)}%", "Avg Score"),
        (c4, str(len(jd_skills)), "Skills Required"),
        (c5, str(stats["weak_candidates"]), "Flagged Weak"),
    ]
    for col, val, label in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top Candidate Highlight ───────────────────────────────────────────────
    if top:
        top_name = top["name"].replace(".pdf","").replace(".txt","")
        st.markdown(f"""
        <div class="success-box">
            🏆 <b>Top Candidate: {top_name}</b> — Final Score: 
            <b>{round(top['final_score']*100,1)}%</b> | 
            Skills Matched: {len(top['matched_skills'])} | 
            Experience: {top['experience_years']} years
        </div>""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Leaderboard", "📊 Analytics", "🔍 Candidate Details", "📥 Download Report"
    ])

    # ─── TAB 1: Leaderboard ───────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">🏆 Candidate Rankings</div>', unsafe_allow_html=True)

        # Sortable comparison table
        table_data = get_comparison_table(ranked)
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">📋 Detailed Score Cards</div>', unsafe_allow_html=True)

        for c in ranked:
            name = c["name"].replace(".pdf","").replace(".txt","")
            medal = c.get("medal","")
            weak = c.get("is_weak",{}).get("is_weak", False)
            card_class = f"rank-card rank-{c['rank']}" + (" weak" if weak else "")
            final_pct = round(c["final_score"]*100,1)
            skill_pct = round(c["skill_score"]*100,1)
            sim_pct   = round(c["similarity_score"]*100,1)
            exp_pct   = round(c["experience_score"]*100,1)

            st.markdown(f"""
            <div class="{card_class}">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <span style="font-size:1.1rem;font-weight:700;color:#1F2937;">
                    {medal} #{c['rank']} — {name}
                  </span>
                  {'<span style="background:#FEE2E2;color:#991B1B;font-size:0.75rem;padding:2px 8px;border-radius:99px;margin-left:8px;font-weight:600;">⚠️ Weak</span>' if weak else '<span style="background:#D1FAE5;color:#065F46;font-size:0.75rem;padding:2px 8px;border-radius:99px;margin-left:8px;font-weight:600;">✅ Good</span>'}
                </div>
                <div style="font-size:1.6rem;font-weight:800;color:#4F46E5;">{final_pct}%</div>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:12px;">
                <div>
                  <div style="font-size:0.72rem;color:#6B7280;font-weight:600;text-transform:uppercase;">Skill Match</div>
                  <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{skill_pct}%;background:#4F46E5;"></div></div>
                  <div style="font-size:0.85rem;font-weight:700;color:#4F46E5;">{skill_pct}%</div>
                </div>
                <div>
                  <div style="font-size:0.72rem;color:#6B7280;font-weight:600;text-transform:uppercase;">Similarity</div>
                  <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{sim_pct}%;background:#10B981;"></div></div>
                  <div style="font-size:0.85rem;font-weight:700;color:#10B981;">{sim_pct}%</div>
                </div>
                <div>
                  <div style="font-size:0.72rem;color:#6B7280;font-weight:600;text-transform:uppercase;">Experience ({c['experience_years']} yrs)</div>
                  <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{exp_pct}%;background:#F59E0B;"></div></div>
                  <div style="font-size:0.85rem;font-weight:700;color:#F59E0B;">{exp_pct}%</div>
                </div>
              </div>
              <div style="margin-top:10px;font-size:0.8rem;color:#6B7280;">
                Matched: <b>{len(c['matched_skills'])}</b> skills &nbsp;|&nbsp;
                Missing: <b>{len(c['missing_skills'])}</b> skills &nbsp;|&nbsp;
                Percentile: <b>{c.get('percentile',0)}th</b>
              </div>
            </div>""", unsafe_allow_html=True)

    # ─── TAB 2: Analytics ────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">📊 Visual Analytics</div>', unsafe_allow_html=True)

        # Bar chart
        fig_bar = visualizer.candidate_comparison_bar(ranked)
        st.plotly_chart(fig_bar, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_funnel = visualizer.leaderboard_funnel(ranked)
            st.plotly_chart(fig_funnel, use_container_width=True)
        with col_b:
            if len(ranked) > 1:
                fig_exp = visualizer.experience_distribution(ranked)
                st.plotly_chart(fig_exp, use_container_width=True)
            else:
                st.info("Add more candidates to see experience distribution.")

        # JD Skills breakdown
        st.markdown('<div class="section-header">🔑 Job Description — Required Skills</div>', unsafe_allow_html=True)
        if jd_skills:
            jd_df = pd.DataFrame([
                {"Skill": s, "Importance": round(w*100,0), "Category": _get_category(s, skills_db)}
                for s, w in sorted(jd_skills.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(jd_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No recognized skills found in the job description. Try a more detailed JD.")

    # ─── TAB 3: Candidate Details ─────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">🔍 Candidate Detail View</div>', unsafe_allow_html=True)

        candidate_names = [c["name"].replace(".pdf","").replace(".txt","") for c in ranked]
        selected_name = st.selectbox("Select a candidate", candidate_names, index=0)

        sel = next((c for c in ranked if c["name"].replace(".pdf","").replace(".txt","") == selected_name), None)

        if sel:
            col_left, col_right = st.columns([1, 1])

            with col_left:
                # Score donut
                fig_donut = visualizer.score_breakdown_donut(sel)
                st.plotly_chart(fig_donut, use_container_width=True)

                # Explanation
                st.markdown("#### 🧠 AI Explanation")
                st.markdown(f"""
                <div class="info-box" style="white-space:pre-line;">{sel['explanation']}</div>
                """, unsafe_allow_html=True)

                if sel.get("is_weak",{}).get("is_weak"):
                    reasons = " • ".join(sel["is_weak"]["reasons"])
                    st.markdown(f"""
                    <div class="warning-box">⚠️ <b>Flagged as Weak Candidate</b><br>{reasons}</div>
                    """, unsafe_allow_html=True)

            with col_right:
                # Skill match chart
                fig_skill = visualizer.skill_match_horizontal_bar(sel, jd_skills)
                st.plotly_chart(fig_skill, use_container_width=True)

            # Skill Gap Details
            gap = get_skill_gap_analysis(
                sel["matched_skills"], sel["missing_skills"],
                sel["matched_weights"], jd_skills
            )

            st.markdown("#### ✔ Matched Skills")
            if gap["matched_skills"]:
                chips = " ".join(f'<span class="chip chip-green">✔ {s}</span>' for s in gap["matched_skills"])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.markdown("_No matching skills found._")

            col_c, col_i, col_o = st.columns(3)
            with col_c:
                st.markdown("#### 🔴 Critical Missing")
                if gap["critical_missing"]:
                    chips = " ".join(f'<span class="chip chip-red">✘ {s}</span>' for s in gap["critical_missing"])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.markdown("_None_")
            with col_i:
                st.markdown("#### 🟡 Important Missing")
                if gap["important_missing"]:
                    chips = " ".join(f'<span class="chip chip-amber">✘ {s}</span>' for s in gap["important_missing"])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.markdown("_None_")
            with col_o:
                st.markdown("#### ⚪ Optional Missing")
                if gap["optional_missing"]:
                    chips = " ".join(f'<span class="chip chip-red" style="background:#F3F4F6;color:#6B7280;">✘ {s}</span>' for s in gap["optional_missing"])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.markdown("_None_")

            # Experience details
            st.markdown("#### 📅 Experience Details")
            exp_info = sel.get("experience_method","none")
            exp_conf = sel.get("experience_confidence","low")
            st.markdown(f"""
            <div class="info-box">
              <b>Years:</b> {sel['experience_years']} &nbsp;|&nbsp;
              <b>Detection method:</b> {exp_info} &nbsp;|&nbsp;
              <b>Confidence:</b> {exp_conf}
            </div>""", unsafe_allow_html=True)

    # ─── TAB 4: Download ─────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">📥 Download Screening Report</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          📄 The PDF report includes: executive summary, full rankings table, 
          per-candidate score breakdown, skill gap analysis, and AI explanations.
        </div>""", unsafe_allow_html=True)

        if st.button("📄 Generate PDF Report"):
            with st.spinner("Generating report..."):
                try:
                    pdf_bytes = generate_pdf_report(ranked, jd_text_input, jd_skills, stats)
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name="resume_screening_report.pdf",
                        mime="application/pdf",
                    )
                    st.success("✅ Report ready! Click the button above to download.")
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

        # CSV download
        st.markdown("---")
        st.markdown("#### 📊 Download Raw Data (CSV)")
        csv_data = get_comparison_table(ranked)
        df_csv = pd.DataFrame(csv_data)
        st.download_button(
            "⬇️ Download CSV",
            data=df_csv.to_csv(index=False),
            file_name="candidate_scores.csv",
            mime="text/csv",
        )

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
      <div style="font-size:4rem;">🎯</div>
      <h2 style="color:#4F46E5;font-weight:700;">Ready to Screen Candidates</h2>
      <p style="color:#6B7280;font-size:1.05rem;max-width:500px;margin:0 auto;">
        Upload resumes and paste a job description in the sidebar, 
        then click <b>Analyze Candidates</b> to get AI-powered rankings.
      </p>
      <div style="margin-top:32px;display:grid;grid-template-columns:repeat(3,1fr);gap:20px;max-width:600px;margin:32px auto 0;">
        <div class="metric-card"><div style="font-size:2rem;">📄</div><div style="font-weight:600;color:#4F46E5;margin-top:8px;">Upload Resumes</div></div>
        <div class="metric-card"><div style="font-size:2rem;">📝</div><div style="font-weight:600;color:#4F46E5;margin-top:8px;">Paste JD</div></div>
        <div class="metric-card"><div style="font-size:2rem;">🚀</div><div style="font-weight:600;color:#4F46E5;margin-top:8px;">Get Rankings</div></div>
      </div>
    </div>""", unsafe_allow_html=True)


