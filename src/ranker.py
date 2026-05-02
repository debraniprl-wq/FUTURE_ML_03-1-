"""
modules/ranker.py
-----------------
Ranks candidates by final score, detects weak candidates,
computes skill gaps, and provides comparison utilities.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def rank_candidates(scored_candidates: List[Dict]) -> List[Dict]:
    """
    Sort candidates from best to worst by final_score.
    Adds rank, medal, and percentile fields.
    
    Args:
        scored_candidates: List of scoring dicts from scorer.score_candidate().
        
    Returns:
        Sorted list with added 'rank', 'medal', and 'percentile' fields.
    """
    if not scored_candidates:
        return []
    
    # Sort descending by final_score, then by skill_score as tiebreaker
    sorted_candidates = sorted(
        scored_candidates,
        key=lambda c: (c["final_score"], c["skill_score"]),
        reverse=True,
    )
    
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    total = len(sorted_candidates)
    
    for i, candidate in enumerate(sorted_candidates):
        rank = i + 1
        candidate["rank"] = rank
        candidate["medal"] = medals.get(rank, "")
        
        # Percentile: rank 1 = 100th percentile
        if total == 1:
            candidate["percentile"] = 100.0
        else:
            candidate["percentile"] = round(((total - i) / total) * 100, 1)
    
    return sorted_candidates


def get_skill_gap_analysis(
    matched_skills: List[str],
    missing_skills: List[str],
    matched_weights: Dict[str, float],
    jd_skills: Dict[str, float],
) -> Dict:
    """
    Produce a detailed skill gap analysis for a candidate.
    
    Args:
        matched_skills: Skills present in both resume and JD.
        missing_skills: JD skills not found in resume.
        matched_weights: {skill: effective_weight} for matched skills.
        jd_skills: {skill: weight} from JD.
        
    Returns:
        dict with categorized matched/missing skills and coverage metrics.
    """
    total_jd_skills = len(jd_skills)
    n_matched = len(matched_skills)
    n_missing = len(missing_skills)
    
    # Categorize missing skills by importance
    critical_missing = []     # weight >= 0.9
    important_missing = []    # weight >= 0.7
    optional_missing = []     # weight < 0.7
    
    for skill in missing_skills:
        weight = jd_skills.get(skill, 0.5)
        if weight >= 0.9:
            critical_missing.append(skill)
        elif weight >= 0.7:
            important_missing.append(skill)
        else:
            optional_missing.append(skill)
    
    # Sort matched skills by effective weight (descending)
    matched_sorted = sorted(
        matched_skills,
        key=lambda s: matched_weights.get(s, jd_skills.get(s, 0)),
        reverse=True,
    )
    
    return {
        "total_jd_skills": total_jd_skills,
        "matched_count": n_matched,
        "missing_count": n_missing,
        "coverage_pct": round((n_matched / total_jd_skills * 100) if total_jd_skills > 0 else 0, 1),
        "matched_skills": matched_sorted,
        "critical_missing": critical_missing,
        "important_missing": important_missing,
        "optional_missing": optional_missing,
        "all_missing": missing_skills,
    }


def get_comparison_table(ranked_candidates: List[Dict]) -> List[Dict]:
    """
    Build a clean comparison table for all candidates.
    
    Returns:
        List of dicts suitable for display as a DataFrame.
    """
    rows = []
    for c in ranked_candidates:
        # Truncate filename for display
        display_name = c["name"].replace(".pdf", "").replace(".txt", "")
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."
        
        rows.append({
            "Rank": f"{c.get('medal', '')} #{c['rank']}",
            "Candidate": display_name,
            "Final Score": f"{round(c['final_score'] * 100, 1)}%",
            "Skill Score": f"{round(c['skill_score'] * 100, 1)}%",
            "Similarity": f"{round(c['similarity_score'] * 100, 1)}%",
            "Experience": f"{c['experience_years']} yrs",
            "Matched Skills": len(c['matched_skills']),
            "Missing Skills": len(c['missing_skills']),
            "Status": "⚠️ Weak" if c.get("is_weak", {}).get("is_weak") else "✅ Good",
        })
    
    return rows


def get_top_candidate(ranked_candidates: List[Dict]) -> Dict:
    """Return the highest-ranked candidate."""
    return ranked_candidates[0] if ranked_candidates else None


def compute_aggregate_stats(ranked_candidates: List[Dict]) -> Dict:
    """
    Compute aggregate statistics across all candidates.
    
    Returns:
        dict with avg/min/max scores, etc.
    """
    if not ranked_candidates:
        return {}
    
    finals = [c["final_score"] for c in ranked_candidates]
    skills = [c["skill_score"] for c in ranked_candidates]
    sims = [c["similarity_score"] for c in ranked_candidates]
    exps = [c["experience_score"] for c in ranked_candidates]
    
    weak_count = sum(1 for c in ranked_candidates if c.get("is_weak", {}).get("is_weak"))
    
    return {
        "total_candidates": len(ranked_candidates),
        "weak_candidates": weak_count,
        "avg_final": round(sum(finals) / len(finals), 4),
        "max_final": round(max(finals), 4),
        "min_final": round(min(finals), 4),
        "avg_skill": round(sum(skills) / len(skills), 4),
        "avg_similarity": round(sum(sims) / len(sims), 4),
        "avg_experience": round(sum(exps) / len(exps), 4),
    }
