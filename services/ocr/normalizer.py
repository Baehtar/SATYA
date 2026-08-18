"""
services/ocr/normalizer.py — OCR Text Cleaning & Multilingual Normalization Layer.
Preserves raw OCR output while creating cleaned, transliterated, and English retrieval forms.
"""
import re
import structlog
from typing import Dict, Any
from src.pipelines.text.language_detector import detect_language, normalize_text
from src.models.schemas import LanguageCode

log = structlog.get_logger(__name__)

# Noise phrases to strip from OCR headline text
NOISE_PATTERNS = [
    r"(?i)\bBREAKING NEWS\b",
    r"(?i)\bURGENT\b",
    r"(?i)\bPLEASE SHARE\b",
    r"(?i)\bFORWARDED AS RECEIVED\b",
    r"(?i)\bEXCLUSIVE\b",
    r"(?i)\bJUST IN\b",
    r"(?i)\bWATCH\b",
    r"(?i)\bSHARE THIS EVERYWHERE\b",
]


def detect_script(text: str) -> str:
    """Detects the primary script of the OCR text."""
    devanagari_count = len(re.findall(r'[\u0900-\u097F]', text))
    tamil_count = len(re.findall(r'[\u0B80-\u0BFF]', text))
    latin_count = len(re.findall(r'[a-zA-Z]', text))

    total = devanagari_count + tamil_count + latin_count
    if total == 0:
        return "Unknown"

    if devanagari_count / total > 0.40:
        return "Devanagari"
    if tamil_count / total > 0.40:
        return "Tamil"
    if latin_count / total > 0.60:
        return "Latin"

    return "Mixed"


def normalize_ocr_result(raw_text: str) -> Dict[str, Any]:
    """
    Cleans OCR output and generates multi-variant representations.
    Returns structured dict with raw_text, cleaned_text, language, script, etc.
    """
    if not raw_text or not raw_text.strip():
        return {
            "raw_text": "",
            "cleaned_text": "",
            "language": LanguageCode.EN.value,
            "script": "Unknown",
            "normalized_text": "",
            "transliterated_text": "",
            "english_text": "",
            "has_readable_text": False,
        }

    # Step 1: Clean OCR noise
    cleaned = raw_text
    for pattern in NOISE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)

    # Clean excessive whitespace and newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Step 2: Language & Script Detection
    original, normalized, urls = normalize_text(cleaned)
    lang = detect_language(normalized)
    script = detect_script(cleaned)

    has_readable_text = len(normalized) >= 10

    # Filter out generic filler phrases when no text is found in image
    filler_patterns = [
        r"no (visible )?text",
        r"no (factual )?claim",
        r"no text provided",
        r"no text found",
        r"no headline",
        r"there is no"
    ]
    if any(re.search(pat, cleaned, re.IGNORECASE) for pat in filler_patterns) and len(cleaned) < 80:
        has_readable_text = False

    log.info(
        "ocr_normalized",
        lang=lang.value,
        script=script,
        raw_len=len(raw_text),
        clean_len=len(cleaned),
        has_readable_text=has_readable_text
    )

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "language": lang.value,
        "script": script,
        "normalized_text": normalized,
        "transliterated_text": cleaned,  # Preserved original script/transliteration
        "english_text": normalized if lang == LanguageCode.EN else "",
        "urls": urls,
        "has_readable_text": has_readable_text,
    }
