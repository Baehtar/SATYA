"""
src/pipelines/screenshot/chyron_detector.py — News chyron tampering detector.
Owned by: Person 4

Detects news channel screenshots and checks if the chyron text
matches actual verified broadcasts.
"""
import asyncio
import functools
import structlog
from pathlib import Path
from src.models.schemas import CheckRequest, ScreenshotAnalysis

log = structlog.get_logger(__name__)

# Known Indian news channel template patterns (logo region + colour palette)
KNOWN_CHANNELS = {
    "NDTV": {"primary_color": (255, 0, 0), "keywords": ["ndtv", "we the people"]},
    "Aaj Tak": {"primary_color": (255, 0, 0), "keywords": ["aaj tak", "आज तक"]},
    "Republic TV": {"primary_color": (0, 0, 200), "keywords": ["republic", "arnab"]},
    "Zee News": {"primary_color": (0, 100, 200), "keywords": ["zee news", "ज़ी न्यूज़"]},
    "India Today": {"primary_color": (255, 100, 0), "keywords": ["india today"]},
    "ABP News": {"primary_color": (255, 0, 0), "keywords": ["abp news", "एबीपी"]},
}


def _detect_channel_from_image(image_path: str) -> tuple[str | None, float]:
    """Detects the news channel from a screenshot using template matching."""
    try:
        import cv2
        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        # Check the bottom 20% of image (where chyrons typically are)
        h, w = img_array.shape[:2]
        bottom_strip = img_array[int(h * 0.75):, :]

        best_channel = None
        best_confidence = 0.0

        # Simple colour-dominance based channel detection
        for channel, config in KNOWN_CHANNELS.items():
            target_r, target_g, target_b = config["primary_color"]
            mask = (
                (np.abs(bottom_strip[:, :, 0].astype(int) - target_r) < 40) &
                (np.abs(bottom_strip[:, :, 1].astype(int) - target_g) < 40) &
                (np.abs(bottom_strip[:, :, 2].astype(int) - target_b) < 40)
            )
            pixel_ratio = mask.sum() / mask.size
            if pixel_ratio > best_confidence:
                best_confidence = pixel_ratio
                best_channel = channel

        if best_confidence > 0.05:  # at least 5% of pixels match
            return best_channel, min(1.0, best_confidence * 10)

        return None, 0.0

    except Exception as e:
        log.warning("channel_detection_failed", error=str(e))
        return None, 0.0


async def _extract_chyron_text(image_path: str) -> str:
    """Uses Gemini Vision to OCR the chyron/ticker text from the screenshot."""
    try:
        import google.generativeai as genai
        from src.config import settings

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        import PIL.Image
        img = PIL.Image.open(image_path)

        response = await model.generate_content_async([
            img,
            "Extract ONLY the news ticker/chyron text from the bottom of this news channel screenshot. "
            "Return just the text, no explanation. If this is not a news screenshot, return 'NOT_NEWS_SCREENSHOT'."
        ])

        text = response.text.strip()
        if text == "NOT_NEWS_SCREENSHOT":
            return ""
        return text

    except Exception as e:
        log.warning("chyron_ocr_failed", error=str(e))
        return ""


async def _verify_chyron(channel: str, chyron_text: str) -> tuple[bool, str]:
    """
    Cross-references chyron text with known broadcast records.
    Currently uses a Google search to verify.
    Returns (verified: bool, details: str).
    TODO: Integrate with NewsAPI or a broadcast archive for production.
    """
    if not chyron_text or not channel:
        return False, ""

    try:
        import httpx
        from src.config import settings

        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(
                "https://serpapi.com/search",
                params={
                    "engine": "google",
                    "q": f'"{chyron_text}" site:{channel.lower().replace(" ", "")}.com OR "{chyron_text}" news',
                    "api_key": settings.serpapi_key,
                    "num": 3,
                    "gl": "in",
                },
            )
            data = r.json()

        results = data.get("organic_results", [])
        if results:
            return True, f"Found in {len(results)} sources"
        return False, "Chyron text not found in any online news source"

    except Exception as e:
        log.warning("chyron_verification_failed", error=str(e))
        return False, str(e)


def _full_screenshot_analysis(image_path: str) -> dict:
    channel, channel_conf = _detect_channel_from_image(image_path)
    return {"channel": channel, "channel_confidence": channel_conf}


async def run_screenshot_pipeline(request: CheckRequest) -> ScreenshotAnalysis:
    """Entry point for screenshot analysis pipeline."""
    import time
    start = time.monotonic()

    if not request.image_path or not Path(request.image_path).exists():
        return ScreenshotAnalysis(error="No image provided")

    try:
        # Detect channel (sync, in thread)
        loop = asyncio.get_running_loop()
        sync_result = await loop.run_in_executor(
            None,
            functools.partial(_full_screenshot_analysis, request.image_path)
        )

        channel = sync_result.get("channel")
        channel_conf = sync_result.get("channel_confidence", 0.0)

        if not channel:
            return ScreenshotAnalysis(
                pipeline_latency_ms=int((time.monotonic() - start) * 1000),
            )

        # Extract chyron text (async, Gemini Vision)
        chyron_text = await _extract_chyron_text(request.image_path)

        # Verify chyron
        verified, verify_details = await _verify_chyron(channel, chyron_text)
        tampering = not verified and bool(chyron_text)

        return ScreenshotAnalysis(
            detected_channel=channel,
            channel_confidence=channel_conf,
            extracted_chyron_text=chyron_text,
            chyron_verified=verified,
            tampering_detected=tampering,
            tampering_details=f"'{chyron_text[:80]}' — {verify_details}" if tampering else "",
            pipeline_latency_ms=int((time.monotonic() - start) * 1000),
        )

    except Exception as e:
        log.error("screenshot_pipeline_failed", error=str(e))
        return ScreenshotAnalysis(
            error=str(e),
            pipeline_latency_ms=int((time.monotonic() - start) * 1000),
        )
