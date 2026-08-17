"""
services/ocr/engine.py — Modular Multilingual OCR Engine.
Supports English, Hindi (Devanagari & Roman), Tamil (Tamil script & Roman), and Mixed text graphics.
Uses Gemini Vision API as primary high-accuracy OCR engine with local fallback capabilities.
"""
import os
import asyncio
import structlog
from PIL import Image
from typing import Dict, Any, Optional
from google import genai
from google.genai import types
from src.config import settings

log = structlog.get_logger(__name__)


def _get_client():
    if not settings.gemini_api_key:
        return None
    return genai.Client(api_key=settings.gemini_api_key)


OCR_PROMPT = """You are a precision OCR engine for news verification.
Extract ALL visible text from this image exactly as written.

The text may be in English, Hindi (Devanagari script), Hindi (Roman/Hinglish), Tamil (Tamil script), Tamil (Roman/Tanglish), or mixed languages.

Instructions:
1. Extract ALL readable headline text, body text, subheadings, and captions.
2. Preserve original words, numbers, and dates.
3. Ignore background noise or tiny watermarks if illegible.
4. Output ONLY the raw extracted text. Do not add conversational commentary like "Here is the text".
"""


async def extract_text_from_image(image_path: str) -> Dict[str, Any]:
    """
    Extracts text from an image using multilingual OCR.
    Returns a dictionary containing raw_text, success status, and metadata.
    """
    if not os.path.exists(image_path):
        log.error("ocr_image_not_found", path=image_path)
        return {
            "success": False,
            "raw_text": "",
            "engine": "none",
            "error": "Image file does not exist."
        }

    log.info("starting_ocr", image_path=image_path)

    # ── Primary Engine: Gemini Vision API ────────────────────────────────────
    client = _get_client()
    if client:
        try:
            with Image.open(image_path) as img:
                # Run vision API call asynchronously
                response = await client.aio.models.generate_content(
                    model=settings.gemini_model,
                    contents=[img, OCR_PROMPT],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=1000,
                    ),
                )

                raw_text = response.text.strip() if response.text else ""
                log.info("ocr_completed_gemini", text_len=len(raw_text))

                return {
                    "success": True,
                    "raw_text": raw_text,
                    "engine": "gemini_vision",
                    "error": None
                }

        except Exception as e:
            log.warning("gemini_vision_ocr_failed_fallback_next", error=str(e))

    # ── Local Fallback: Pytesseract / EasyOCR if available ────────────────────
    try:
        import pytesseract
        with Image.open(image_path) as img:
            raw_text = await asyncio.to_thread(pytesseract.image_to_string, img)
            raw_text = raw_text.strip()
            log.info("ocr_completed_pytesseract", text_len=len(raw_text))

            return {
                "success": True,
                "raw_text": raw_text,
                "engine": "pytesseract",
                "error": None
            }
    except Exception as e:
        log.warning("local_ocr_unavailable", error=str(e))

    return {
        "success": False,
        "raw_text": "",
        "engine": "fallback",
        "error": "No OCR engine available or no text could be extracted."
    }
