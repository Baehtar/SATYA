"""
src/pipelines/text/language_detector.py — Multilingual detection & text normalization.
Handles EN, HI_DEVANAGARI, HI_ROMAN (Hinglish), TA_TAMIL, TA_ROMAN (Tanglish), MIXED.
"""
import re
import unicodedata
import structlog
from typing import Tuple
from src.models.schemas import LanguageCode

log = structlog.get_logger(__name__)

# Common Hinglish indicators
HINGLISH_KEYWORDS = {
    "ne", "ko", "ka", "ki", "ke", "hai", "hain", "tha", "thi", "the", "gaya", "gaye",
    "gayi", "karo", "karo!", "bhej", "raha", "rahi", "rahe", "batao", "sach", "jhoot",
    "par", "se", "aur", "ya", "magar", "lekin", "apne", "humara", "kya", "kyun", "kab"
}

# Common Tanglish indicators
TANGLISH_KEYWORDS = {
    "nethu", "naikku", "pannitaar", "pannanga", "irukku", "illai", "varum", "solranga",
    "sonnaa", "aachu", "yen", "yaru", "parunga", "share", "potrukanga", "nadandhadhu"
}


def detect_language(text: str) -> LanguageCode:
    """
    Classifies text into EN, HI_DEVANAGARI, HI_ROMAN, TA_TAMIL, TA_ROMAN, or MIXED.
    Does NOT assume Roman script is purely English.
    """
    if not text or not text.strip():
        return LanguageCode.EN

    # Count script characters
    devanagari_count = len(re.findall(r'[\u0900-\u097F]', text))
    tamil_count = len(re.findall(r'[\u0B80-\u0BFF]', text))
    latin_count = len(re.findall(r'[a-zA-Z]', text))

    total_chars = len(text)

    # Script predominance
    if devanagari_count > 0 and devanagari_count > latin_count and devanagari_count > tamil_count:
        if latin_count > 5:
            return LanguageCode.MIXED
        return LanguageCode.HI_DEVANAGARI

    if tamil_count > 0 and tamil_count > latin_count and tamil_count > devanagari_count:
        if latin_count > 5:
            return LanguageCode.MIXED
        return LanguageCode.TA_TAMIL

    if latin_count > 0:
        words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
        
        hinglish_matches = words.intersection(HINGLISH_KEYWORDS)
        tanglish_matches = words.intersection(TANGLISH_KEYWORDS)

        if len(hinglish_matches) >= 2:
            return LanguageCode.HI_ROMAN
        if len(tanglish_matches) >= 2:
            return LanguageCode.TA_ROMAN

        # If both script types present
        if devanagari_count > 0 or tamil_count > 0:
            return LanguageCode.MIXED

        return LanguageCode.EN

    return LanguageCode.EN


def normalize_text(text: str) -> Tuple[str, str, list]:
    """
    Normalizes text without destroying original source of truth.
    Returns: (original_text, normalized_text, extracted_urls)
    """
    original_text = text

    # Extract URLs
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)

    # Unicode normalization
    normalized = unicodedata.normalize('NFKC', text)

    # Remove URLs for clean search string
    normalized_clean = re.sub(url_pattern, '', normalized)

    # Clean whitespace while preserving words, numbers, and dates
    normalized_clean = re.sub(r'\s+', ' ', normalized_clean).strip()

    return original_text, normalized_clean, urls
