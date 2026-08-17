"""
src/api/routes/check.py — All fact-check endpoints.

Endpoints:
    POST /check/text        — JSON body with text claim
    POST /check/image       — multipart file upload (image)
    POST /check/audio       — multipart file upload (voice note)
    POST /check             — unified: auto-detects from multipart fields present
    GET  /check/verdict/{id} — retrieve a cached verdict by request_id
"""
import uuid
import time
import tempfile
import os
import asyncio
import structlog

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from src.models.schemas import (
    CheckRequest, MessageType, EvidenceBundle, VerdictCard,
)
from src.verdict.aggregator import aggregate_evidence
from src.verdict.confidence import calibrate_confidence
from src.verdict.card_generator import generate_card
from src.config import settings

log = structlog.get_logger(__name__)
router = APIRouter()

# In-memory verdict cache for the demo (keyed by request_id)
# In production: replace with Redis or DB lookup
_verdict_cache: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────
#  Request / Response models
# ─────────────────────────────────────────────────────────────

class TextCheckRequest(BaseModel):
    text: str
    claimed_date: Optional[str] = None
    user_id: Optional[int] = 0
    chat_id: Optional[int] = 0


class VerdictResponse(BaseModel):
    request_id: str
    verdict: str
    confidence_level: str
    confidence_score: float
    explanation_english: str
    explanation_hindi: str
    signals_used: list[str]
    sources: list[dict]
    source_urls: list[str]
    blind_spot_warning: Optional[str]
    is_adversarial_suspected: bool
    total_latency_ms: int
    pipeline_breakdown: dict    # per-pipeline latency for transparency


# ─────────────────────────────────────────────────────────────
#  Helper: run verdict engine on any bundle
# ─────────────────────────────────────────────────────────────

async def _run_verdict(bundle: EvidenceBundle, request_id: str, start: float) -> VerdictCard:
    evidence = aggregate_evidence(bundle)
    score = calibrate_confidence(evidence)
    card = await generate_card(evidence, score, request_id)
    card.total_latency_ms = int((time.monotonic() - start) * 1000)
    return card


def _card_to_response(card: VerdictCard, breakdown: dict) -> dict:
    return {
        "request_id": card.request_id,
        "verdict": card.verdict.value,
        "confidence_level": card.confidence_level.value,
        "confidence_score": round(card.confidence_score, 3),
        "explanation_english": card.explanation_english,
        "explanation_hindi": card.explanation_hindi,
        "signals_used": card.signals_used,
        "sources": [
            {
                "source_name": s.source_name,
                "source_url": s.source_url,
                "verdict": s.fact_check_verdict,
                "snippet": s.snippet,
                "match_confidence": round(s.match_confidence, 3),
            }
            for s in card.sources
        ],
        "source_urls": card.source_urls,
        "blind_spot_warning": card.blind_spot_warning,
        "is_adversarial_suspected": card.is_adversarial_suspected,
        "total_latency_ms": card.total_latency_ms,
        "pipeline_breakdown": breakdown,
    }


async def _log_to_db(card: VerdictCard, message_type: str, claim: str = "") -> None:
    """Background task: log the result to DB."""
    try:
        from src.db.database import log_check
        await log_check({
            "request_id": card.request_id,
            "user_id": 0,
            "message_type": message_type,
            "verdict": card.verdict.value,
            "confidence_score": card.confidence_score,
            "confidence_level": card.confidence_level.value,
            "extracted_claim": claim[:500],
            "latency_ms": card.total_latency_ms,
        })
    except Exception as e:
        log.warning("db_log_failed", error=str(e))


# ─────────────────────────────────────────────────────────────
#  POST /check/text
# ─────────────────────────────────────────────────────────────

@router.post("/text", response_model=None, summary="Fact-check a text forward")
async def check_text(body: TextCheckRequest, background_tasks: BackgroundTasks):
    """
    Submit a text forward for fact-checking.

    **Body:**
    ```json
    {
      "text": "BREAKING: Government announces free petrol for all Indians!",
      "claimed_date": "2024-08-15",   // optional
      "user_id": 12345                // optional, for logging
    }
    ```

    **Returns:** VerdictResponse with verdict, confidence, bilingual explanation, and sources.
    """
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text field cannot be empty")

    request_id = str(uuid.uuid4())[:12]
    start = time.monotonic()
    log.info("check_text_start", request_id=request_id, text_len=len(body.text))

    try:
        async with asyncio.timeout(settings.total_timeout):
            from src.pipelines.text.pipeline import run_text_pipeline

            request = CheckRequest(
                request_id=request_id,
                message_type=MessageType.TEXT,
                user_id=body.user_id or 0,
                chat_id=body.chat_id or 0,
                text_content=body.text,
                claimed_date=body.claimed_date,
            )

            claim_analysis = await run_text_pipeline(request)
            text_latency = claim_analysis.pipeline_latency_ms

            bundle = EvidenceBundle(
                request_id=request_id,
                message_type=MessageType.TEXT,
                claim_analysis=claim_analysis,
            )

            card = await _run_verdict(bundle, request_id, start)

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Pipeline timed out after {settings.total_timeout}s")
    except Exception as e:
        log.error("check_text_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    breakdown = {"text_pipeline_ms": text_latency, "verdict_engine_ms": card.total_latency_ms - text_latency}
    result = _card_to_response(card, breakdown)

    # Cache and log
    _verdict_cache[request_id] = result
    background_tasks.add_task(
        _log_to_db, card, "text",
        claim_analysis.extracted_claim if claim_analysis else ""
    )

    log.info("check_text_done", request_id=request_id, verdict=card.verdict.value, latency_ms=card.total_latency_ms)
    return JSONResponse(content=result)


# ─────────────────────────────────────────────────────────────
#  POST /check/image
# ─────────────────────────────────────────────────────────────

@router.post("/image", response_model=None, summary="Fact-check an image")
async def check_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file (JPEG/PNG/WEBP)"),
    caption: Optional[str] = Form(default=None, description="Caption or text accompanying the image"),
    claimed_date: Optional[str] = Form(default=None, description="Date mentioned in the forward (YYYY-MM-DD)"),
    user_id: Optional[int] = Form(default=0),
):
    """
    Submit an image (and optional caption) for fact-checking.

    Runs in parallel:
    - AI-generation detection (HuggingFace ViT)
    - Manipulation detection (ELA + EXIF + noise)
    - Reverse image search + date comparison (SerpAPI Google Lens)

    If a caption is provided, also runs the text fact-check pipeline simultaneously.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail=f"File must be an image, got: {file.content_type}")

    request_id = str(uuid.uuid4())[:12]
    start = time.monotonic()
    log.info("check_image_start", request_id=request_id, filename=file.filename, has_caption=bool(caption))

    # Save upload to temp file
    suffix = "." + (file.filename or "img.jpg").rsplit(".", 1)[-1]
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        image_path = tmp.name
    finally:
        tmp.close()

    try:
        async with asyncio.timeout(settings.total_timeout):
            from src.pipelines.image.pipeline import run_image_pipeline
            from src.pipelines.text.pipeline import run_text_pipeline

            message_type = MessageType.IMAGE_WITH_CAPTION if caption else MessageType.IMAGE

            request = CheckRequest(
                request_id=request_id,
                message_type=message_type,
                user_id=user_id or 0,
                chat_id=0,
                image_path=image_path,
                text_content=caption,
                claimed_date=claimed_date,
            )

            # Run image pipeline (+ optional text pipeline) in parallel
            if caption:
                img_task = asyncio.create_task(run_image_pipeline(request))
                txt_task = asyncio.create_task(run_text_pipeline(request))
                img_result, txt_result = await asyncio.gather(img_task, txt_task, return_exceptions=True)
                image_analysis = img_result if not isinstance(img_result, Exception) else None
                claim_analysis = txt_result if not isinstance(txt_result, Exception) else None
            else:
                image_analysis = await run_image_pipeline(request)
                claim_analysis = None

            bundle = EvidenceBundle(
                request_id=request_id,
                message_type=message_type,
                image_analysis=image_analysis,
                claim_analysis=claim_analysis,
            )

            card = await _run_verdict(bundle, request_id, start)

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Pipeline timed out after {settings.total_timeout}s")
    except Exception as e:
        log.error("check_image_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

    breakdown = {
        "image_pipeline_ms": image_analysis.pipeline_latency_ms if image_analysis else 0,
        "text_pipeline_ms": claim_analysis.pipeline_latency_ms if claim_analysis else 0,
        "verdict_engine_ms": card.total_latency_ms - (image_analysis.pipeline_latency_ms if image_analysis else 0),
    }
    result = _card_to_response(card, breakdown)
    _verdict_cache[request_id] = result
    background_tasks.add_task(_log_to_db, card, message_type.value)

    log.info("check_image_done", request_id=request_id, verdict=card.verdict.value, latency_ms=card.total_latency_ms)
    return JSONResponse(content=result)


# ─────────────────────────────────────────────────────────────
#  POST /check/audio
# ─────────────────────────────────────────────────────────────

@router.post("/audio", response_model=None, summary="Fact-check a voice note")
async def check_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file (OGG/MP3/WAV/M4A)"),
    user_id: Optional[int] = Form(default=0),
):
    """
    Submit a voice note for fact-checking.

    Pipeline:
    1. Whisper STT → transcription (GPU-accelerated)
    2. Voice clone detection (spectral/pitch analysis)
    3. Transcription → text fact-check pipeline
    """
    request_id = str(uuid.uuid4())[:12]
    start = time.monotonic()
    log.info("check_audio_start", request_id=request_id, filename=file.filename)

    suffix = "." + (file.filename or "audio.ogg").rsplit(".", 1)[-1]
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(await file.read())
        tmp.flush()
        audio_path = tmp.name
    finally:
        tmp.close()

    try:
        async with asyncio.timeout(settings.audio_pipeline_timeout):
            from src.pipelines.audio.voice_analyzer import run_audio_pipeline
            from src.pipelines.text.pipeline import run_text_pipeline

            request = CheckRequest(
                request_id=request_id,
                message_type=MessageType.VOICE,
                user_id=user_id or 0,
                chat_id=0,
                audio_path=audio_path,
            )

            audio_analysis = await run_audio_pipeline(request)
            claim_analysis = None

            # If transcription succeeded, run text fact-check pipeline
            if audio_analysis.transcription.strip():
                text_request = request.model_copy(update={
                    "message_type": MessageType.TEXT,
                    "text_content": audio_analysis.transcription,
                })
                claim_analysis = await run_text_pipeline(text_request)

            bundle = EvidenceBundle(
                request_id=request_id,
                message_type=MessageType.VOICE,
                audio_analysis=audio_analysis,
                claim_analysis=claim_analysis,
            )

            card = await _run_verdict(bundle, request_id, start)

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Audio pipeline timed out after {settings.audio_pipeline_timeout}s")
    except Exception as e:
        log.error("check_audio_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    breakdown = {
        "audio_pipeline_ms": audio_analysis.pipeline_latency_ms if audio_analysis else 0,
        "text_pipeline_ms": claim_analysis.pipeline_latency_ms if claim_analysis else 0,
        "verdict_engine_ms": card.total_latency_ms - (audio_analysis.pipeline_latency_ms if audio_analysis else 0),
        "transcription": audio_analysis.transcription[:200] if audio_analysis else "",
    }
    result = _card_to_response(card, breakdown)
    _verdict_cache[request_id] = result
    background_tasks.add_task(_log_to_db, card, "voice")

    log.info("check_audio_done", request_id=request_id, verdict=card.verdict.value, latency_ms=card.total_latency_ms)
    return JSONResponse(content=result)


# ─────────────────────────────────────────────────────────────
#  POST /check/screenshot
# ─────────────────────────────────────────────────────────────

@router.post("/screenshot", response_model=None, summary="Fact-check a news screenshot")
async def check_screenshot(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Screenshot image (JPEG/PNG)"),
    user_id: Optional[int] = Form(default=0),
):
    """
    Submit a news channel screenshot for tampering detection.

    Pipeline:
    1. Channel detection (colour + keyword heuristics)
    2. Chyron/ticker OCR via Gemini Vision
    3. Chyron cross-reference against online sources
    4. Image manipulation detection
    """
    request_id = str(uuid.uuid4())[:12]
    start = time.monotonic()
    log.info("check_screenshot_start", request_id=request_id)

    suffix = "." + (file.filename or "screenshot.jpg").rsplit(".", 1)[-1]
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(await file.read())
        tmp.flush()
        image_path = tmp.name
    finally:
        tmp.close()

    try:
        async with asyncio.timeout(settings.image_pipeline_timeout):
            from src.pipelines.screenshot.chyron_detector import run_screenshot_pipeline
            from src.pipelines.image.pipeline import run_image_pipeline

            request = CheckRequest(
                request_id=request_id,
                message_type=MessageType.SCREENSHOT,
                user_id=user_id or 0,
                chat_id=0,
                image_path=image_path,
            )

            # Run image + screenshot pipelines in parallel
            img_task = asyncio.create_task(run_image_pipeline(request))
            ss_task = asyncio.create_task(run_screenshot_pipeline(request))
            img_result, ss_result = await asyncio.gather(img_task, ss_task, return_exceptions=True)

            bundle = EvidenceBundle(
                request_id=request_id,
                message_type=MessageType.SCREENSHOT,
                image_analysis=img_result if not isinstance(img_result, Exception) else None,
                screenshot_analysis=ss_result if not isinstance(ss_result, Exception) else None,
            )

            card = await _run_verdict(bundle, request_id, start)

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Screenshot pipeline timed out")
    except Exception as e:
        log.error("check_screenshot_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

    breakdown = {
        "image_pipeline_ms": img_result.pipeline_latency_ms if not isinstance(img_result, Exception) else 0,
        "screenshot_pipeline_ms": ss_result.pipeline_latency_ms if not isinstance(ss_result, Exception) else 0,
    }
    result = _card_to_response(card, breakdown)
    _verdict_cache[request_id] = result
    background_tasks.add_task(_log_to_db, card, "screenshot")

    log.info("check_screenshot_done", request_id=request_id, verdict=card.verdict.value, latency_ms=card.total_latency_ms)
    return JSONResponse(content=result)


# ─────────────────────────────────────────────────────────────
#  POST /check  — unified auto-detect endpoint
# ─────────────────────────────────────────────────────────────

@router.post("/", response_model=None, summary="Unified fact-check (auto-detects type)")
async def check_unified(
    background_tasks: BackgroundTasks,
    text: Optional[str] = Form(default=None),
    claimed_date: Optional[str] = Form(default=None),
    user_id: Optional[int] = Form(default=0),
    file: Optional[UploadFile] = File(default=None),
):
    """
    Unified endpoint — the bot can always call this and let the backend decide.

    Detection logic:
    - file = image → /check/image
    - file = audio → /check/audio
    - file = image + text → /check/image (with caption)
    - text only → /check/text
    """
    if file:
        ct = file.content_type or ""
        if ct.startswith("audio/") or file.filename.endswith((".ogg", ".mp3", ".wav", ".m4a")):
            return await check_audio(background_tasks, file, user_id)
        elif ct.startswith("image/"):
            return await check_image(background_tasks, file, text, claimed_date, user_id)
        else:
            raise HTTPException(status_code=422, detail=f"Unsupported file type: {ct}")
    elif text:
        body = TextCheckRequest(text=text, claimed_date=claimed_date, user_id=user_id)
        return await check_text(body, background_tasks)
    else:
        raise HTTPException(status_code=422, detail="Provide either a file upload or a text field")


# ─────────────────────────────────────────────────────────────
#  GET /check/verdict/{id}
# ─────────────────────────────────────────────────────────────

@router.get("/verdict/{request_id}", summary="Retrieve a past verdict")
async def get_verdict(request_id: str):
    """Retrieve a cached verdict by request_id (in-memory cache, survives process lifetime)."""
    if request_id not in _verdict_cache:
        raise HTTPException(status_code=404, detail=f"Verdict '{request_id}' not found. It may have expired.")
    return JSONResponse(content=_verdict_cache[request_id])
