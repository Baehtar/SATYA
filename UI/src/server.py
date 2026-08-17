"""
UI/src/server.py — FastAPI server for the Satya web UI.

This is a *thin* layer: it accepts an upload, calls the same analysis functions
the Telegram bot calls (services/ml_service.py), streams progress over SSE, and
returns the verdict card built by UI/src/adapter.py. All fact-checking logic
lives in the shared backend — nothing is duplicated here.

Run from the repo root:
    python -m UI.run              # or: uvicorn UI.src.server:app --port 8000

Endpoints:
    GET  /                        → the single-page app
    GET  /api/health              → liveness + which API keys are configured
    POST /api/check               → multipart {text?, image?, audio?} → {"id": ...}
    GET  /api/check/{id}/stream   → SSE: progress* → (verdict | failed) → done
"""
import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

import structlog
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from services.ml_service import check_image, check_mixed, check_text, check_voice
from services.audio.convert import extract_audio_from_video
from UI.src.adapter import build_card

logging.basicConfig(level=settings.log_level)
log = structlog.get_logger(__name__)

# Paths are resolved from this file, so the server works from any CWD.
UI_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = UI_DIR / "frontend"
UPLOAD_DIR = UI_DIR / "uploads"

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024
JOB_TTL_SECONDS = 600

# Pipeline step ids (services/ml_service.py) → the three steps the UI shows.
STEP_MAP = {
    "image_analysis": "analyze",
    "text_analysis": "analyze",
    "audio_analysis": "analyze",
    "video_analysis": "analyze",
    "fact_check": "search",
    "generating_verdict": "verdict",
}
UI_STEPS = ["analyze", "search", "verdict"]

# What the user asked for. FAKE_NEWS verifies a claim (text, or text read out of an
# image); AI_IMAGE only asks the detector whether a picture is synthetic — no OCR,
# no claim extraction, no source search.
MODE_FAKE_NEWS = "fake_news"
MODE_AI_IMAGE = "ai_image"
VALID_MODES = {MODE_FAKE_NEWS, MODE_AI_IMAGE}

app = FastAPI(title="Satya Web UI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  In-flight checks
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Job:
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: Optional[asyncio.Task] = None
    created_at: float = field(default_factory=time.monotonic)


_jobs: Dict[str, Job] = {}


def _prune_jobs() -> None:
    """Drops abandoned jobs (client never connected to the stream)."""
    now = time.monotonic()
    for job_id, job in list(_jobs.items()):
        if now - job.created_at > JOB_TTL_SECONDS:
            if job.task and not job.task.done():
                job.task.cancel()
            _jobs.pop(job_id, None)
            log.info("job_pruned", job_id=job_id)


@app.on_event("startup")
async def on_startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    log.info(
        "satya_web_ui_started",
        frontend=str(FRONTEND_DIR),
        gemini_configured=bool(settings.gemini_api_key),
        hf_configured=bool(os.getenv("HF_API_KEY", "")),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Static front-end
# ─────────────────────────────────────────────────────────────────────────────

app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    """Reports which capabilities are actually usable — the fastest way to see
    why results are degraded (a missing key silently downgrades a pipeline)."""
    from services.audio import whisper_stt

    whisper_config = whisper_stt.resolve_api_config()
    return {
        "status": "ok",
        "gemini_configured": bool(settings.gemini_api_key),     # OCR, claims, transcription
        "hf_configured": bool(os.getenv("HF_API_KEY", "")),     # AI-image detection
        "google_factcheck_configured": bool(settings.google_factcheck_api_key),
        "serpapi_configured": bool(settings.serpapi_key),
        "ffmpeg_available": bool(shutil.which("ffmpeg")),       # voice-note transcoding
        "stt_engine": settings.stt_engine,                      # voice-note speech-to-text
        "whisper_configured": whisper_stt.is_configured(),
        "whisper_model": whisper_config[2] if whisper_config else None,
        # Reverse image search. Vision reads local files; Lens additionally needs
        # a public image URL, so it is usually off — see services/image/serpapi_lens.py.
        "reverse_search_enabled": settings.reverse_search_enabled,
        "google_vision_configured": bool(settings.google_vision_api_key),
        "serpapi_lens_configured": bool(settings.serpapi_key) and bool(
            settings.public_image_base_url or settings.serpapi_lens_allow_upload
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Analysis
# ─────────────────────────────────────────────────────────────────────────────

async def _run_analysis(
    job_id: str,
    queue: asyncio.Queue,
    text: str,
    image_path: Optional[str],
    audio_path: Optional[str],
    mode: str,
    extra_paths: Optional[list] = None,
) -> None:
    """Runs the shared pipeline, streaming progress into `queue`."""
    start = time.monotonic()
    started_steps: list[str] = []
    current_step: Optional[str] = None

    async def progress(message: str, step: str = "") -> None:
        """Adapts the backend's (message, step) hook to stepper events."""
        nonlocal current_step
        ui_step = STEP_MAP.get(step, "analyze")

        # check_mixed() runs the image and text pipelines concurrently, so their
        # updates interleave. The stepper only ever moves forward.
        if current_step and UI_STEPS.index(ui_step) < UI_STEPS.index(current_step):
            return

        if ui_step != current_step:
            if current_step:
                await queue.put(("progress", {"step": current_step, "status": "completed", "message": ""}))
            current_step = ui_step
            if ui_step not in started_steps:
                started_steps.append(ui_step)
        await queue.put(("progress", {"step": ui_step, "status": "running", "message": message}))

    try:
        async with asyncio.timeout(settings.total_timeout):
            if mode == MODE_AI_IMAGE:
                # Authenticity only — the caption, OCR and fact-check search are
                # deliberately skipped so the answer is just about the picture.
                result = await check_image(image_path, progress_callback=progress, mode=MODE_AI_IMAGE)
            elif image_path and text:
                result = await check_mixed(image_path, text, progress_callback=progress)
            elif image_path:
                result = await check_image(image_path, progress_callback=progress)
            elif audio_path:
                result = await check_voice(audio_path, progress_callback=progress)
            else:
                result = await check_text(text, progress_callback=progress)

        if current_step:
            await queue.put(("progress", {"step": current_step, "status": "completed", "message": ""}))

        # Steps a given flow never needs (e.g. no fact-check search for a photo
        # with no readable text) are marked skipped, not silently left spinning.
        # The verdict step always runs — the card is written below.
        started_steps.append("verdict")
        for step in UI_STEPS:
            if step not in started_steps:
                await queue.put(("progress", {"step": step, "status": "skipped", "message": "Not needed"}))

        await queue.put(("progress", {"step": "verdict", "status": "running", "message": "Writing the verdict…"}))
        latency_ms = int((time.monotonic() - start) * 1000)
        card = await build_card(result, submitted_text=text, latency_ms=latency_ms, mode=mode)
        await queue.put(("progress", {"step": "verdict", "status": "completed", "message": "Verdict ready"}))
        await queue.put(("verdict", card))
        log.info("check_complete", job_id=job_id, verdict=card["verdict"], latency_ms=latency_ms)

    except asyncio.TimeoutError:
        log.warning("check_timeout", job_id=job_id, budget_s=settings.total_timeout)
        await queue.put(("failed", {
            "error": f"The check took longer than {settings.total_timeout}s. Please try again."
        }))
    except asyncio.CancelledError:
        log.info("check_cancelled", job_id=job_id)
        raise
    except Exception as e:
        log.error("check_failed", job_id=job_id, error=str(e), exc_info=True)
        await queue.put(("failed", {"error": str(e)}))
    finally:
        # The footer promises uploads are deleted after analysis — keep that promise.
        all_paths = [image_path, audio_path] + (extra_paths or [])
        for path in all_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    log.warning("upload_cleanup_failed", path=path, error=str(e))
        await queue.put(("done", {}))


async def _save_upload(upload: UploadFile, job_id: str, max_bytes: int) -> str:
    contents = await upload.read()
    if len(contents) > max_bytes:
        raise ValueError(f"File is too large ({len(contents) // 1024} KB). Limit is {max_bytes // (1024 * 1024)} MB.")

    suffix = Path(upload.filename or "").suffix.lower() or ".bin"
    path = UPLOAD_DIR / f"{job_id}{suffix}"
    path.write_bytes(contents)
    return str(path)


@app.post("/api/check")
async def create_check(
    text: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
    mode: str = Form(default=MODE_FAKE_NEWS),
) -> JSONResponse:
    """Accepts the submission and starts the analysis; returns a job id to stream."""
    _prune_jobs()

    if mode not in VALID_MODES:
        return JSONResponse({"error": f"Unknown mode '{mode}'."}, status_code=422)

    text = (text or "").strip()
    if not text and image is None and audio is None and video is None:
        return JSONResponse({"error": "Send text, an image, a voice recording, or a video."}, status_code=422)

    if mode == MODE_AI_IMAGE:
        if image is None:
            return JSONResponse({"error": "AI-image detection needs an image."}, status_code=422)
        # Nothing else is analysed in this mode; don't pretend otherwise.
        text = ""
        audio = None
        video = None

    job_id = str(uuid.uuid4())
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    extracted_audio_path: Optional[str] = None

    try:
        if image is not None:
            if not (image.content_type or "").startswith("image/"):
                return JSONResponse(
                    {"error": f"Expected an image, got {image.content_type or 'unknown type'}."},
                    status_code=422,
                )
            image_path = await _save_upload(image, job_id, MAX_IMAGE_BYTES)

        if audio is not None:
            if not (audio.content_type or "").startswith("audio/"):
                return JSONResponse(
                    {"error": f"Expected audio, got {audio.content_type or 'unknown type'}."},
                    status_code=422,
                )
            audio_path = await _save_upload(audio, job_id + "_audio", MAX_AUDIO_BYTES)

        if video is not None:
            if not (video.content_type or "").startswith("video/"):
                return JSONResponse(
                    {"error": f"Expected a video file, got {video.content_type or 'unknown type'}."},
                    status_code=422,
                )
            video_path = await _save_upload(video, job_id + "_video", MAX_VIDEO_BYTES)
            # Extract audio from the video; the video itself is not needed by the pipeline.
            extracted_audio_path = await extract_audio_from_video(video_path)
            if not extracted_audio_path:
                return JSONResponse(
                    {"error": "Could not extract audio from the video. Make sure the file contains an audio track and that ffmpeg is installed."},
                    status_code=422,
                )
            # Use the extracted WAV as the audio path for the pipeline.
            audio_path = extracted_audio_path

    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=413)

    job = Job()
    job.task = asyncio.create_task(
        _run_analysis(job_id, job.queue, text, image_path, audio_path, mode,
                      extra_paths=[video_path, extracted_audio_path])
    )
    _jobs[job_id] = job

    log.info("check_started", job_id=job_id, mode=mode, has_text=bool(text),
             has_image=bool(image_path), has_audio=bool(audio_path),
             has_video=bool(video_path))
    return JSONResponse({"id": job_id})


@app.get("/api/check/{job_id}/stream")
async def stream_check(job_id: str):
    """Server-sent events: progress* → (verdict | failed) → done."""
    job = _jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "Unknown or expired check id."}, status_code=404)

    async def events() -> AsyncIterator[str]:
        try:
            while True:
                event, payload = await job.queue.get()
                yield f"event: {event}\ndata: {json.dumps(payload)}\n\n"
                if event == "done":
                    break
        finally:
            # Covers both a clean finish and the browser closing mid-check.
            _jobs.pop(job_id, None)
            if job.task and not job.task.done():
                job.task.cancel()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # don't let a proxy buffer the stream
        },
    )
