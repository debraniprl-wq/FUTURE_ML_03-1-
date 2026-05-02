"""
modules/scorer.py
-----------------
Multi-factor scoring engine for resume-job description matching.

Final Score Formula:
    Final = 0.5 * Skill Score + 0.3 * Similarity Score + 0.2 * Experience Score

All sub-scores are in the range [0, 1].
Every score is fully explainable — no black box outputs.
"""

import math
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score weights (easily configurable)
# ---------------------------------------------------------------------------
SKILL_WEIGHT = 0.5
SIMILARITY_WEIGHT = 0.3
EXPERIENCE_WEIGHT = 0.2

# ---------------------------------------------------------------------------
# Experience scoring lookup table
# Maps years of experience → normalized score [0, 1]
# ---------------------------------------------------------------------------
_EXPERIENCE_SCORE_TABLE = [
    (0, 0.0),
    (1, 0.15),
    (2, 0.30),
    (3, 0.50),
    (4, 0.62),
    (5, 0.75),
    (6, 0.83),
    (7, 0.90),
    (8, 0.95),
    (10, 1.0),
]


def normalize_experience_score(years: float) -> float:
    """
    Convert years of experience to a normalized score [0, 1].
    Uses linear interpolation between breakpoints in the lookup table.
    
    Args:
        years: Years of experience (can be float).
        
    Returns:
        Score in [0, 1].
    """
    if years <= 0:
        return 0.0
    if years >= _EXPERIENCE_SCORE_TABLE[-1][0]:
        return 1.0
    
    # Linear interpolation
    for i in range(len(_EXPERIENCE_SCORE_TABLE) - 1):
        y0, s0 = _EXPERIENCE_SCORE_TABLE[i]
        y1, s1 = _EXPERIENCE_SCORE_TABLE[i + 1]
        if y0 <= years <= y1:
            t = (years - y0) / (y1 - y0)
            return round(s0 + t * (s1 - s0), 4)
    
    return 1.0


def compute_skill_score(
    resume_skills: dict,
    jd_skills: dict,
    section_boost: float = 0.2,
) -> dict:
    """
    Compute a weighted skill match score.
    
    Skill Score = Σ(weight_i * boost_i for matched skills) / Σ(weight_i for all JD skills)
    
    Where boost_i = 1.0 + section_boost if skill found in dedicated Skills section,
    else 1.0 (capped so score doesn't exceed 1.0).
    
    Args:
        resume_skills: {skill: {"weight": float, "in_skills_section": bool}}
        jd_skills: {skill: weight} — skills required by the JD
        section_boost: Extra multiplier for skills in the Skills section header.
        
    Returns:
        dict with:
            - score: float [0, 1]
            - matched_skills: list of matched skill names
            - missing_skills: list of missing skill names
            - matched_weights: {skill: effective_weight}
            - total_jd_weight: sum of all JD skill weights
    """
    if not jd_skills:
        return {
            "score": 0.0,
            "matched_skills": [],
            "missing_skills": [],
            "matched_weights": {},
            "total_jd_weight": 0.0,
        }
    
    total_jd_weight = sum(jd_skills.values())
    matched_weight_sum = 0.0
    matched_skills = []
    missing_skills = []
    matched_weights = {}
    
    for skill, jd_weight in jd_skills.items():
        if skill in resume_skills:
            resume_skill_data = resume_skills[skill]
            # Apply section boost if skill appears in dedicated skills section
            boost = 1.0 + section_boost if resume_skill_data.get("in_skills_section") else 1.0
            effective_weight = jd_weight * boost
            matched_weight_sum += effective_weight
            matched_skills.append(skill)
            matched_weights[skill] = round(effective_weight, 3)
        else:
            missing_skills.append(skill)
    
    # Cap score at 1.0 (boosted weights can push sum over total)
    raw_score = matched_weight_sum / total_jd_weight if total_jd_weight > 0 else 0.0
    score = min(round(raw_score, 4), 1.0)
    
    return {
        "score": score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_weights": matched_weights,
        "total_jd_weight": round(total_jd_weight, 3),
    }


def compute_similarity_score(resume_text: str, jd_text: str) -> dict:
    """
    Compute semantic similarity between resume and JD using TF-IDF cosine similarity.
    
    TF-IDF captures term importance while cosine similarity measures directional
    alignment regardless of document length.
    
    Args:
        resume_text: Cleaned/expanded resume text.
        jd_text: Cleaned/expanded job description text.
        
    Returns:
        dict with:
            - score: float [0, 1]
            - method: 'tfidf_cosine'
    """
    try:
        # Build TF-IDF matrix from both documents
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),          # Unigrams and bigrams
            stop_words="english",
            max_features=10000,
            sublinear_tf=True,           # Dampen term frequency counts
            min_df=1,
        )
        
        corpus = [jd_text, resume_text]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Cosine similarity between JD (index 0) and resume (index 1)
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        score = float(sim[0][0])
        score = round(min(max(score, 0.0), 1.0), 4)
        
    except Exception as e:
        logger.warning(f"TF-IDF similarity failed: {e}")
        score = 0.0
    
    return {
        "score": score,
        "method": "tfidf_cosine",
    }


def compute_experience_score(experience_data: dict) -> dict:
    """
    Compute a normalized experience score.
    
    Args:
        experience_data: Output from extractor.extract_experience_years()
        
    Returns:
        dict with:
            - score: float [0, 1]
            - years: float
            - method: extraction method used
            - confidence: confidence level
    """
    years = experience_data.get("years", 0)
    method = experience_data.get("method", "none")
    confidence = experience_data.get("confidence", "low")
    
    score = normalize_experience_score(years)
    
    return {
        "score": score,
        "years": years,
        "method": method,
        "confidence": confidence,
    }


def compute_final_score(
    skill_score: float,
    similarity_score: float,
    experience_score: float,
) -> dict:
    """
    Compute the weighted final score.
    
    Final = 0.5 * Skill + 0.3 * Similarity + 0.2 * Experience
    
    Returns:
        dict with final_score and the weighted contributions of each component.
    """
    weighted_skill = SKILL_WEIGHT * skill_score
    weighted_similarity = SIMILARITY_WEIGHT * similarity_score
    weighted_experience = EXPERIENCE_WEIGHT * experience_score
    
    final = weighted_skill + weighted_similarity + weighted_experience
    final = round(min(max(final, 0.0), 1.0), 4)
    
    return {
        "final_score": final,
        "weighted_skill": round(weighted_skill, 4),
        "weighted_similarity": round(weighted_similarity, 4),
        "weighted_experience": round(weighted_experience, 4),
        "weights": {
            "skill": SKILL_WEIGHT,
            "similarity": SIMILARITY_WEIGHT,
            "experience": EXPERIENCE_WEIGHT,
        },
    }


def score_candidate(
    resume_data: dict,
    jd_text: str,
    jd_skills: dict,
    skills_flat: dict,
) -> dict:
    """
    Run the full scoring pipeline for a single candidate.
    
    Args:
        resume_data: Parsed + preprocessed resume dict containing:
            - name: filename
            - preprocessed: output of preprocessor.preprocess_resume()
            - experience: output of extractor.extract_experience_years()
            - skills: output of extractor.extract_skills()
        jd_text: Cleaned/expanded job description text.
        jd_skills: {skill: weight} extracted from JD.
        skills_flat: Flat skills DB for reference.
        
    Returns:
        Complete scoring dict for this candidate.
    """
    preprocessed = resume_data["preprocessed"]
    resume_skills = resume_data["skills"]
    experience_data = resume_data["experience"]
    
    # --- Compute sub-scores ---
    skill_result = compute_skill_score(resume_skills, jd_skills)
    similarity_result = compute_similarity_score(preprocessed["expanded"], jd_text)
    experience_result = compute_experience_score(experience_data)
    
    # --- Compute final score ---
    final_result = compute_final_score(
        skill_result["score"],
        similarity_result["score"],
        experience_result["score"],
    )
    
    # --- Generate explanation ---
    explanation = _generate_explanation(
        resume_data["name"],
        skill_result,
        similarity_result,
        experience_result,
        final_result,
    )
    
    return {
        "name": resume_data["name"],
        "skill_score": skill_result["score"],
        "similarity_score": similarity_result["score"],
        "experience_score": experience_result["score"],
        "final_score": final_result["final_score"],
        "weighted_skill": final_result["weighted_skill"],
        "weighted_similarity": final_result["weighted_similarity"],
        "weighted_experience": final_result["weighted_experience"],
        "matched_skills": skill_result["matched_skills"],
        "missing_skills": skill_result["missing_skills"],
        "matched_weights": skill_result["matched_weights"],
        "experience_years": experience_result["years"],
        "experience_method": experience_result["method"],
        "experience_confidence": experience_result["confidence"],
        "similarity_method": similarity_result["method"],
        "explanation": explanation,
        "is_weak": _flag_weak_candidate(skill_result, experience_result, final_result),
    }


def _generate_explanation(
    name: str,
    skill_result: dict,
    similarity_result: dict,
    experience_result: dict,
    final_result: dict,
) -> str:
    """
    Generate a human-readable explanation of why the candidate scored as they did.
    This makes the system fully transparent and not a black box.
    """
    lines = []
    
    # Skill explanation
    n_matched = len(skill_result["matched_skills"])
    n_missing = len(skill_result["missing_skills"])
    skill_pct = round(skill_result["score"] * 100, 1)
    
    if n_matched == 0:
        lines.append(f"❌ No required skills matched (skill score: {skill_pct}%).")
    elif n_missing == 0:
        lines.append(f"✅ All required skills matched ({n_matched}/{n_matched}) — perfect skill coverage ({skill_pct}%).")
    else:
        lines.append(
            f"⚡ {n_matched} of {n_matched + n_missing} required skills matched ({skill_pct}% weighted coverage)."
        )
        top_missing = skill_result["missing_skills"][:3]
        if top_missing:
            lines.append(f"   Key missing skills: {', '.join(top_missing)}.")
    
    # Similarity explanation
    sim_pct = round(similarity_result["score"] * 100, 1)
    if similarity_result["score"] >= 0.7:
        lines.append(f"📄 Resume content is highly aligned with the JD (similarity: {sim_pct}%).")
    elif similarity_result["score"] >= 0.4:
        lines.append(f"📄 Resume partially aligns with JD language (similarity: {sim_pct}%).")
    else:
        lines.append(
            f"📄 Low content similarity to JD — resume may lack relevant keywords ({sim_pct}%)."
        )
    
    # Experience explanation
    years = experience_result["years"]
    exp_method = experience_result["method"]
    if exp_method == "none":
        lines.append("📅 No extractable work experience found.")
    elif exp_method == "explicit":
        lines.append(f"📅 Explicitly stated {years} year(s) of experience.")
    else:
        lines.append(f"📅 ~{years} year(s) estimated from employment date ranges.")
    
    # Final score summary
    final = round(final_result["final_score"] * 100, 1)
    lines.append(
        f"\n🏆 Final Score: {final}% "
        f"(Skill ×{final_result['weights']['skill']} + "
        f"Similarity ×{final_result['weights']['similarity']} + "
        f"Experience ×{final_result['weights']['experience']})"
    )
    
    return "\n".join(lines)


def _flag_weak_candidate(
    skill_result: dict,
    experience_result: dict,
    final_result: dict,
) -> dict:
    """
    Flag weak candidates and explain why.
    
    Returns:
        dict with:
            - is_weak: bool
            - reasons: list of strings
    """
    reasons = []
    
    if skill_result["score"] < 0.3:
        reasons.append("Very low skill match (< 30%)")
    if experience_result["years"] == 0 and experience_result["method"] == "none":
        reasons.append("No work experience detected")
    if final_result["final_score"] < 0.25:
        reasons.append("Overall score below 25%")
    if len(skill_result["matched_skills"]) == 0:
        reasons.append("Zero required skills matched")
    
    return {
        "is_weak": len(reasons) > 0,
        "reasons": reasons,
    }


def batch_similarity_scores(resume_texts: list, jd_text: str) -> list:
    """
    Compute TF-IDF similarity for all resumes at once (more efficient than one-by-one).
    
    Args:
        resume_texts: List of resume texts.
        jd_text: Job description text.
        
    Returns:
        List of similarity scores (floats) in same order as resume_texts.
    """
    if not resume_texts:
        return []
    
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=10000,
            sublinear_tf=True,
            min_df=1,
        )
        
        all_docs = [jd_text] + resume_texts
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        
        jd_vector = tfidf_matrix[0:1]
        resume_vectors = tfidf_matrix[1:]
        similarities = cosine_similarity(jd_vector, resume_vectors)[0]
        
        return [round(float(s), 4) for s in similarities]
    
    except Exception as e:
        logger.error(f"Batch similarity failed: {e}")
        return [0.0] * len(resume_texts)
