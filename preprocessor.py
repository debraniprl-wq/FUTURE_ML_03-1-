"""
modules/preprocessor.py
-----------------------
Text cleaning, normalization, and synonym expansion.
Prepares raw resume and job description text for NLP processing.
"""

import re
import string
import json
import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load synonym map from skills_db.json
# ---------------------------------------------------------------------------
_SKILLS_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "skills_db.json")

def _load_synonyms() -> dict:
    """Load the acronym/synonym mapping from the skills database."""
    try:
        with open(_SKILLS_DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
        return db.get("synonyms", {})
    except Exception as e:
        logger.warning(f"Could not load synonyms: {e}")
        return {}

SYNONYMS = _load_synonyms()

# ---------------------------------------------------------------------------
# Resume section headers (for section-aware parsing)
# ---------------------------------------------------------------------------
SECTION_PATTERNS = {
    "skills": [
        r"(?:technical\s+)?skills?",
        r"core\s+competenc(?:y|ies)",
        r"technologies",
        r"tools?\s+&\s+technologies",
        r"expertise",
        r"proficienc(?:y|ies)",
    ],
    "experience": [
        r"(?:work\s+|professional\s+)?experience",
        r"employment(?:\s+history)?",
        r"work\s+history",
        r"career\s+history",
        r"positions?\s+held",
        r"professional\s+background",
    ],
    "education": [
        r"education(?:al\s+background)?",
        r"academic(?:\s+background)?",
        r"qualifications?",
        r"degrees?",
        r"certifications?",
    ],
    "projects": [
        r"projects?",
        r"personal\s+projects?",
        r"key\s+projects?",
        r"academic\s+projects?",
    ],
    "summary": [
        r"(?:professional\s+)?summary",
        r"(?:career\s+)?objective",
        r"about\s+(?:me|myself)",
        r"profile",
        r"overview",
    ],
}

# Compile section header patterns
_SECTION_REGEX = {
    section: re.compile(
        r"(?:^|\n)\s*(?:" + "|".join(pats) + r")\s*[:\-]?\s*\n",
        re.IGNORECASE | re.MULTILINE,
    )
    for section, pats in SECTION_PATTERNS.items()
}


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Normalize unicode characters
    - Remove excessive whitespace and special formatting characters
    - Preserve punctuation needed for NLP
    
    Args:
        text: Raw extracted text.
        
    Returns:
        Cleaned text string.
    """
    if not text:
        return ""
    
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Remove null bytes and other control characters (except newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    
    # Replace multiple spaces with single space (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    
    # Remove excessive newlines (more than 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Remove bullet point decorators but keep content
    text = re.sub(r"^[\s]*[•·▪▸►◆➤✓✔➔→\-\*]+\s*", "", text, flags=re.MULTILINE)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def expand_acronyms(text: str) -> str:
    """
    Expand known acronyms and synonyms in text.
    Example: 'ML' → 'ML Machine Learning', so both forms are searchable.
    
    We append the expansion rather than replace, so the original term
    is still present (important for exact skill matching).
    
    Args:
        text: Input text.
        
    Returns:
        Text with expansions appended inline.
    """
    for acronym, expansion in SYNONYMS.items():
        # Match the acronym as a whole word (case-insensitive)
        pattern = r"\b" + re.escape(acronym) + r"\b"
        # Check if expansion is already present (avoid duplicating)
        if re.search(pattern, text, re.IGNORECASE):
            if expansion.lower() not in text.lower():
                # Append expansion next to acronym
                text = re.sub(
                    pattern,
                    lambda m: m.group() + f" ({expansion})",
                    text,
                    flags=re.IGNORECASE,
                )
    return text


def normalize_for_matching(text: str) -> str:
    """
    Normalize text specifically for skill matching:
    - Lowercase
    - Remove punctuation except hyphens (for terms like 'scikit-learn')
    - Expand acronyms
    
    Args:
        text: Input text.
        
    Returns:
        Normalized lowercase text.
    """
    text = text.lower()
    # Keep hyphens and forward slashes (common in tech terms: 'ci/cd', 'scikit-learn')
    text = re.sub(r"[^\w\s\-\/\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_sections(text: str) -> dict:
    """
    Detect resume sections and split text accordingly.
    Section-aware parsing lets us weight skills found in the "Skills"
    section more heavily than skills mentioned in passing elsewhere.
    
    Args:
        text: Cleaned resume text.
        
    Returns:
        dict mapping section names to their text content.
        Always includes 'full' key with the complete text.
    """
    sections = {"full": text}
    
    # Find all section header positions
    found_sections = []
    for section_name, pattern in _SECTION_REGEX.items():
        for match in pattern.finditer(text):
            found_sections.append((match.start(), match.end(), section_name))
    
    # Sort by position
    found_sections.sort(key=lambda x: x[0])
    
    # Extract text between section headers
    for i, (start, end, section_name) in enumerate(found_sections):
        # Section ends at the next header or end of text
        if i + 1 < len(found_sections):
            next_start = found_sections[i + 1][0]
        else:
            next_start = len(text)
        
        section_text = text[end:next_start].strip()
        
        # If same section appears multiple times, concatenate
        if section_name in sections:
            sections[section_name] += "\n" + section_text
        else:
            sections[section_name] = section_text
    
    return sections


def preprocess_resume(text: str) -> dict:
    """
    Full preprocessing pipeline for a resume.
    
    Args:
        text: Raw extracted text.
        
    Returns:
        dict with:
            - cleaned: cleaned text
            - expanded: text with acronym expansions
            - normalized: normalized lowercase text for matching
            - sections: dict of detected sections
    """
    cleaned = clean_text(text)
    expanded = expand_acronyms(cleaned)
    normalized = normalize_for_matching(expanded)
    sections = extract_sections(cleaned)
    
    return {
        "cleaned": cleaned,
        "expanded": expanded,
        "normalized": normalized,
        "sections": sections,
    }


def preprocess_job_description(text: str) -> dict:
    """
    Full preprocessing pipeline for a job description.
    Same as resume but without section extraction.
    
    Args:
        text: Raw job description text.
        
    Returns:
        dict with cleaned, expanded, normalized keys.
    """
    cleaned = clean_text(text)
    expanded = expand_acronyms(cleaned)
    normalized = normalize_for_matching(expanded)
    
    return {
        "cleaned": cleaned,
        "expanded": expanded,
        "normalized": normalized,
    }
