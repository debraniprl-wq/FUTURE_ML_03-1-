"""
modules/parser.py
-----------------
Handles extraction of raw text from uploaded resume files.
Supports PDF (via PyMuPDF) and plain text (.txt) formats.
Includes MD5-based duplicate detection to avoid processing the same resume twice.
"""

import hashlib
import io
import logging

logger = logging.getLogger(__name__)


def _compute_hash(file_bytes: bytes) -> str:
    """Compute MD5 hash of file bytes for duplicate detection."""
    return hashlib.md5(file_bytes).hexdigest()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF file using PyMuPDF (fitz).
    
    Args:
        file_bytes: Raw bytes of the PDF file.
        
    Returns:
        Concatenated text from all pages.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pages_text.append(page.get_text("text"))
        doc.close()
        return "\n".join(pages_text)
    except ImportError:
        logger.error("PyMuPDF not installed. Install with: pip install pymupdf")
        raise
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""


def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Extract text from a plain text file.
    
    Args:
        file_bytes: Raw bytes of the text file.
        
    Returns:
        Decoded text string.
    """
    # Try UTF-8 first, fall back to latin-1
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def parse_resume(uploaded_file) -> dict:
    """
    Parse a resume file (PDF or TXT) and return structured data.
    
    Args:
        uploaded_file: Streamlit UploadedFile object.
        
    Returns:
        dict with keys:
            - name: filename (str)
            - text: extracted raw text (str)
            - hash: MD5 hash for deduplication (str)
            - file_type: 'pdf' or 'txt'
            - error: error message if parsing failed (str or None)
    """
    file_bytes = uploaded_file.read()
    file_hash = _compute_hash(file_bytes)
    file_name = uploaded_file.name
    file_ext = file_name.lower().split(".")[-1]
    
    result = {
        "name": file_name,
        "text": "",
        "hash": file_hash,
        "file_type": file_ext,
        "error": None,
    }
    
    try:
        if file_ext == "pdf":
            result["text"] = extract_text_from_pdf(file_bytes)
        elif file_ext in ("txt", "text"):
            result["text"] = extract_text_from_txt(file_bytes)
        else:
            result["error"] = f"Unsupported file type: .{file_ext}"
    except Exception as e:
        result["error"] = str(e)
    
    # Validate that we got meaningful text
    if not result["error"] and len(result["text"].strip()) < 50:
        result["error"] = "Could not extract sufficient text from this file."
    
    return result


def deduplicate_resumes(parsed_resumes: list) -> tuple[list, list]:
    """
    Remove duplicate resumes based on MD5 hash.
    
    Args:
        parsed_resumes: List of parsed resume dicts.
        
    Returns:
        Tuple of (unique_resumes, duplicate_names) where duplicate_names
        is a list of filenames that were removed as duplicates.
    """
    seen_hashes = set()
    unique = []
    duplicates = []
    
    for resume in parsed_resumes:
        if resume["hash"] in seen_hashes:
            duplicates.append(resume["name"])
        else:
            seen_hashes.add(resume["hash"])
            unique.append(resume)
    
    return unique, duplicates
