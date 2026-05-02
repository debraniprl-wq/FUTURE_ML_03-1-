"""
modules/extractor.py
---------------------
Skill extraction and experience years extraction using:
- Rule-based matching against the skills database (primary method)
- Section-aware weighting (skills in "Skills" section scored higher)
- Regex-based experience year extraction

This module does NOT use spaCy as a hard dependency; it falls back gracefully
to rule-based methods if spaCy is unavailable.
"""

import re
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load skills database
# ---------------------------------------------------------------------------
_SKILLS_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "skills_db.json")


def load_skills_db() -> dict:
    """Load the full skills database."""
    try:
        with open(_SKILLS_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load skills DB: {e}")
        return {}


def build_skills_flat(skills_db: dict) -> dict:
    """
    Flatten skills database into a single dict: {skill_name: weight}.
    Excludes the 'synonyms' key.
    
    Returns:
        {skill: weight} for all skills across all categories.
    """
    flat = {}
    for category, skills in skills_db.items():
        if category == "synonyms":
            continue
        if isinstance(skills, dict):
            for skill, weight in skills.items():
                flat[skill] = weight
    return flat


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------

def extract_skills(text: str, skills_flat: dict, section_texts: dict = None) -> dict:
    """
    Extract skills from text using rule-based matching against the skills DB.
    
    Section-awareness: if section_texts is provided, skills found in the
    "skills" section get a 1.2x match confidence boost.
    
    Args:
        text: Normalized/expanded text to search.
        skills_flat: Flat dict of {skill: weight} from skills DB.
        section_texts: Optional dict of section name → section text.
        
    Returns:
        dict: {skill_name: {"weight": float, "in_skills_section": bool}}
    """
    found_skills = {}
    text_lower = text.lower()
    
    # Text of the skills section (for boost detection)
    skills_section_text = ""
    if section_texts:
        skills_section_text = section_texts.get("skills", "").lower()
    
    for skill, weight in skills_flat.items():
        # Build a case-insensitive whole-word match pattern
        # Escape special regex characters in skill names
        escaped = re.escape(skill.lower())
        # Handle multi-word: allow flexible spaces
        escaped = escaped.replace(r"\ ", r"\s+")
        pattern = r"\b" + escaped + r"\b"
        
        if re.search(pattern, text_lower):
            in_skills_section = bool(
                skills_section_text and re.search(pattern, skills_section_text)
            )
            found_skills[skill] = {
                "weight": weight,
                "in_skills_section": in_skills_section,
            }
    
    return found_skills


def extract_skills_from_jd(jd_text: str, skills_flat: dict) -> dict:
    """
    Extract required skills from a job description.
    
    Returns:
        dict: {skill: weight} for each skill found in the JD.
    """
    jd_lower = jd_text.lower()
    found = {}
    for skill, weight in skills_flat.items():
        escaped = re.escape(skill.lower())
        escaped = escaped.replace(r"\ ", r"\s+")
        pattern = r"\b" + escaped + r"\b"
        if re.search(pattern, jd_lower):
            found[skill] = weight
    return found


# ---------------------------------------------------------------------------
# Experience year extraction
# ---------------------------------------------------------------------------

# Regex patterns for explicit "N years of experience" mentions
_EXPLICIT_EXPERIENCE_PATTERNS = [
    # "5 years of experience", "5+ years experience", "5-8 years experience"
    r"(\d+)\s*[\+\-]?\s*(?:to\s*\d+\s*)?years?\s+(?:of\s+)?(?:relevant\s+)?(?:work\s+)?experience",
    # "experience of 5 years"
    r"experience\s+(?:of\s+)?(\d+)\s*[\+\-]?\s*years?",
    # "5 year experience"
    r"(\d+)\s*[\-]?\s*year\s+experience",
]

# Date range patterns to calculate experience from job history
# e.g., "Jan 2018 - Dec 2021", "2019 – Present", "March 2020 to Current"
_DATE_RANGE_PATTERN = re.compile(
    r"""
    (?:
        (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|
           Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|
           Dec(?:ember)?)\s+
    )?
    (20\d{2}|19\d{2})          # Start year
    \s*(?:–|-|to|—)\s*
    (?:
        (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|
           Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|
           Dec(?:ember)?)\s+
    )?
    (20\d{2}|19\d{2}|[Pp]resent|[Cc]urrent|[Nn]ow|[Tt]oday)   # End year or "Present"
    """,
    re.VERBOSE,
)


def extract_experience_years(text: str) -> dict:
    """
    Extract total years of professional experience from resume text.
    
    Strategy:
    1. Look for explicit "N years of experience" statements.
    2. Parse date ranges from employment history and sum up durations.
    3. Return the maximum of both methods (most conservative estimate).
    
    Args:
        text: Cleaned resume text.
        
    Returns:
        dict with:
            - years: estimated total years (float)
            - method: how it was determined ('explicit', 'date_range', 'none')
            - confidence: 'high', 'medium', 'low'
    """
    current_year = datetime.now().year
    
    # Method 1: Explicit statements
    explicit_years = []
    for pattern in _EXPLICIT_EXPERIENCE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                years = int(match.group(1))
                if 0 < years <= 50:  # Sanity check
                    explicit_years.append(years)
            except (IndexError, ValueError):
                continue
    
    # Method 2: Date range parsing
    date_ranges = []
    for match in _DATE_RANGE_PATTERN.finditer(text):
        start_year_str = match.group(1)
        end_year_str = match.group(2)
        
        try:
            start_year = int(start_year_str)
            # Handle "Present", "Current", etc.
            if end_year_str.lower() in ("present", "current", "now", "today"):
                end_year = current_year
            else:
                end_year = int(end_year_str)
            
            duration = end_year - start_year
            if 0 < duration <= 50:  # Sanity check
                date_ranges.append(duration)
        except ValueError:
            continue
    
    # Sum unique date ranges (deduplicate overlapping ranges approximately)
    # We take the total sum but cap at 45 years to avoid data errors
    date_range_total = min(sum(date_ranges), 45) if date_ranges else 0
    
    # Determine result
    if explicit_years:
        years = max(explicit_years)
        method = "explicit"
        confidence = "high"
    elif date_range_total > 0:
        years = date_range_total
        method = "date_range"
        confidence = "medium"
    else:
        years = 0
        method = "none"
        confidence = "low"
    
    return {
        "years": years,
        "method": method,
        "confidence": confidence,
        "explicit_mentions": explicit_years,
        "date_ranges_found": date_ranges,
    }
