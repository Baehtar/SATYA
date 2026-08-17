"""
src/bot/router.py — Message type detection and pipeline dispatcher.
Owned by: Person 3

Classifies incoming Telegram messages and dispatches to the correct pipeline(s).
For IMAGE_WITH_CAPTION, runs both image and text pipelines in parallel.
"""
import asyncio
import time
import tempfile
import os
import structlog
from telegram import Update

from src.config import settings
from src.models.schemas import (
    CheckRequest, MessageType, EvidenceBundle, VerdictCard,
)
from src.verdict.aggregator import aggregate_evidence
from src.verdict.confidence import calibrate_confidence
from src.verdict.card_generator import generate_card

log = structlog.get_logger(__name__)


async def route_message(update: Update, request_id: str) -> VerdictCard:
    """
    Main dispatch function. Detects message type, runs the correct pipeline(s)
    in parallel, aggregates evidence, and returns a VerdictCard.
    """
    start_time = time.monotonic()
    message = update.message

    # ── Build CheckRequest ────────────────────────────────────────────────────
    request = await _build_check_request(update, request_id)
    log.info("routing_message", request_id=request_id, type=request.message_type)

    # ── Dispatch to pipelines ─────────────────────────────────────────────────
    try:
        async with asyncio.timeout(settings.total_timeout):
            bundle = await _dispatch(request)
    finally:
        # Cleanup temp files
        _cleanup_temp_files(request)

    bundle.total_latency_ms = int((time.monotonic() - start_time) * 1000)

    # ── Verdict Engine ────────────────────────────────────────────────────────
    evidence = aggregate_evidence(bundle)
    confidence = calibrate_confidence(evidence)
    card = await generate_card(evidence, confidence, request_id)
    card.total_latency_ms = bundle.total_latency_ms

    log.info(
        "verdict_generated",
        request_id=request_id,
        verdict=card.verdict,
        confidence=card.confidence_score,
        latency_ms=card.total_latency_ms,
    )
    return card


async def _dispatch(request: CheckRequest) -> EvidenceBundle:
    """Runs the appropriate pipeline(s) based on message type."""
    # Import pipelines lazily to avoid circular imports
    from src.pipelines.image.pipeline import run_image_pipeline
    from src.pipelines.text.pipeline import run_text_pipeline
    from src.pipelines.audio.voice_analyzer import run_audio_pipeline
    from src.pipelines.screenshot.chyron_detector import run_screenshot_pipeline

    bundle = EvidenceBundle(
        request_id=request.request_id,
        message_type=request.message_type,
    )

    match request.message_type:
        case MessageType.IMAGE:
            bundle.image_analysis = await run_image_pipeline(request)

        case MessageType.TEXT:
            bundle.claim_analysis = await run_text_pipeline(request)

        case MessageType.IMAGE_WITH_CAPTION:
            # Run image + text pipelines in parallel
            image_task = asyncio.create_task(run_image_pipeline(request))
            text_task = asyncio.create_task(run_text_pipeline(request))
            results = await asyncio.gather(image_task, text_task, return_exceptions=True)
            bundle.image_analysis = results[0] if not isinstance(results[0], Exception) else None
            bundle.claim_analysis = results[1] if not isinstance(results[1], Exception) else None

        case MessageType.VOICE:
            bundle.audio_analysis = await run_audio_pipeline(request)
            # If transcription succeeded, also run text pipeline
            if bundle.audio_analysis and bundle.audio_analysis.transcription:
                text_request = request.model_copy(update={
                    "message_type": MessageType.TEXT,
                    "text_content": bundle.audio_analysis.transcription,
                })
                bundle.claim_analysis = await run_text_pipeline(text_request)

        case MessageType.SCREENSHOT:
            # Run image pipeline + screenshot-specific detector in parallel
            image_task = asyncio.create_task(run_image_pipeline(request))
            ss_task = asyncio.create_task(run_screenshot_pipeline(request))
            results = await asyncio.gather(image_task, ss_task, return_exceptions=True)
            bundle.image_analysis = results[0] if not isinstance(results[0], Exception) else None
            bundle.screenshot_analysis = results[1] if not isinstance(results[1], Exception) else None

        case _:
            # Unknown type — try text if there's any text content
            if request.text_content:
                bundle.claim_analysis = await run_text_pipeline(request)

    return bundle


async def _build_check_request(update: Update, request_id: str) -> CheckRequest:
    """Downloads media and builds a CheckRequest from a Telegram Update."""
    message = update.message
    message_type = MessageType.UNKNOWN
    image_path = None
    text_content = None
    audio_path = None

    if message.photo:
        # Highest resolution photo
        photo = message.photo[-1]
        image_path = await _download_file(update, photo.file_id, ".jpg")
        text_content = message.caption  # may be None

        # Detect if it looks like a news screenshot
        message_type = (
            MessageType.SCREENSHOT if _looks_like_screenshot(message)
            else MessageType.IMAGE_WITH_CAPTION if text_content
            else MessageType.IMAGE
        )

    elif message.document and message.document.mime_type.startswith("image/"):
        image_path = await _download_file(update, message.document.file_id, ".jpg")
        text_content = message.caption
        message_type = MessageType.IMAGE_WITH_CAPTION if text_content else MessageType.IMAGE

    elif message.voice or message.audio:
        file_id = (message.voice or message.audio).file_id
        audio_path = await _download_file(update, file_id, ".ogg")
        message_type = MessageType.VOICE

    elif message.text:
        text_content = message.text
        message_type = MessageType.TEXT

    return CheckRequest(
        request_id=request_id,
        message_type=message_type,
        user_id=update.effective_user.id,
        chat_id=update.effective_chat.id,
        image_path=image_path,
        text_content=text_content,
        audio_path=audio_path,
    )


async def _download_file(update: Update, file_id: str, suffix: str) -> str:
    """Downloads a Telegram file to a temp path and returns the path."""
    file = await update.get_bot().get_file(file_id)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    await file.download_to_drive(tmp.name)
    return tmp.name


def _looks_like_screenshot(message) -> bool:
    """Heuristic: is the image likely a news-channel screenshot?"""
    caption = (message.caption or "").lower()
    keywords = ["ndtv", "aaj tak", "republic", "zee news", "india today",
                "abp", "news18", "breaking", "live", "exclusive"]
    return any(kw in caption for kw in keywords)


def _cleanup_temp_files(request: CheckRequest) -> None:
    """Removes downloaded temp files after processing."""
    for path in [request.image_path, request.audio_path]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
