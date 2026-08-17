"""
src/pipelines/text/claim_extractor.py — Core claim extraction using google.genai with response_schema & automatic retry logic.
Extracts factual claims, entities, search queries, and checkability into strict Pydantic schemas.
"""
import asyncio
import structlog
from typing import Dict, Any
from google import genai
from google.genai import types
from src.config import settings
from src.pipelines.text.language_detector import detect_language, normalize_text
from src.models.schemas import ClaimType, LanguageCode, ClaimExtractionSchema

log = structlog.get_logger(__name__)


def _get_client():
    if not settings.gemini_api_key:
        return None
    return genai.Client(api_key=settings.gemini_api_key)


CLAIM_EXTRACTION_PROMPT = """You are an expert fact-checker specialising in Indian misinformation and viral messages.

Analyse the following message forwarded on social media/WhatsApp.
The message may be in English, Hindi (Devanagari or Hinglish), Tamil (Tamil script or Tanglish), or mixed.

ORIGINAL MESSAGE:
---
{text}
---

DETECTED LANGUAGE CODE: {lang}

Rules:
- is_checkable = false ONLY for pure personal opinions, general greetings, or uncheckable predictions.
- Always extract a clear factual proposition in English (and Hindi/Tamil if applicable).
- Generate 3 compact search queries suitable for Google/Fact-check search.
"""


async def extract_claim(text: str) -> Dict[str, Any]:
    """
    Normalizes text, detects language, and extracts structured claim info asynchronously with retries.
    """
    original_text, normalized_text, urls = normalize_text(text)
    lang_code = detect_language(normalized_text)

    log.info("extracting_claim", lang=lang_code.value, text_len=len(text))

    prompt = CLAIM_EXTRACTION_PROMPT.format(
        text=normalized_text[:3000],
        lang=lang_code.value
    )

    client = _get_client()
    if not client:
        return _fallback_extraction(original_text, normalized_text, lang_code, urls)

    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ClaimExtractionSchema,
                    temperature=0.1,
                    max_output_tokens=600,
                ),
            )

            extracted: ClaimExtractionSchema = getattr(response, "parsed", None)
            if not extracted or not getattr(extracted, "claim", None):
                break

            result = {
                "original_text": original_text,
                "normalized_text": normalized_text,
                "language": lang_code,
                "claim": extracted.claim,
                "claim_hindi": extracted.claim_hindi,
                "claim_tamil": extracted.claim_tamil,
                "claim_type": extracted.claim_type,
                "is_checkable": extracted.is_checkable,
                "checkability_reason": extracted.checkability_reason,
                "entities": {"people": [], "places": [], "dates": [], "organisations": []},
                "keywords": extracted.keywords,
                "search_queries": extracted.search_queries or [extracted.claim],
                "urls": urls,
            }

            log.info("claim_extracted_successfully", claim=result.get("claim", "")[:80])
            return result

        except Exception as e:
            if ("429" in str(e) or "503" in str(e)) and attempt < 2:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            log.warning("gemini_claim_extraction_failed_fallback_used", error=str(e))
            break

    return _fallback_extraction(original_text, normalized_text, lang_code, urls)


def _fallback_extraction(original_text: str, normalized_text: str, lang_code: LanguageCode, urls: list) -> dict:
    return {
        "original_text": original_text,
        "normalized_text": normalized_text,
        "language": lang_code,
        "claim": normalized_text[:200],
        "claim_hindi": normalized_text[:200],
        "claim_tamil": "",
        "claim_type": "other",
        "is_checkable": True,
        "checkability_reason": "Automated extraction fallback.",
        "entities": {"people": [], "places": [], "dates": [], "organisations": []},
        "keywords": [normalized_text[:50]],
        "search_queries": [normalized_text[:150]],
        "urls": urls,
    }
