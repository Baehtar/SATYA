"""
services/audio/whisper_stt.py — Whisper speech-to-text.

Two backends, selected by WHISPER_BACKEND:

  api   (default) — any OpenAI-compatible /audio/transcriptions endpoint.
                    Works with OpenAI (whisper-1) and Groq (whisper-large-v3)
                    out of the box; point WHISPER_API_BASE at anything else
                    that speaks the same protocol (e.g. a self-hosted
                    faster-whisper server). Needs only httpx, already a dep.
  local           — the `openai-whisper` pip package running on this machine.
                    No API key, but pulls in torch and a multi-GB checkpoint.

Returns the same dict shape as the Gemini engine in transcribe.py, plus the
language and per-segment data Whisper gives us for free:
    {success, text, language, confidence, segments, engine, error}
"""
import asyncio
import functools
import math
import os
import structlog
from typing import Any, Dict, Optional, Tuple

import httpx

from src.config import settings
from services.audio.convert import to_wav

log = structlog.get_logger(__name__)

# Containers the Whisper API accepts directly. Anything else is transcoded.
SUPPORTED_CONTAINERS = {
    ".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".oga",
    ".ogg", ".opus", ".wav", ".webm",
}

# Provider defaults, used when WHISPER_API_BASE / WHISPER_API_MODEL are blank.
_PROVIDER_DEFAULTS = {
    "openai": ("https://api.openai.com/v1", "whisper-1"),
    "groq": ("https://api.groq.com/openai/v1", "whisper-large-v3"),
}

# OpenAI rejects uploads above 25 MB; Groq's limit is the same on the free tier.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

_local_model = None


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

def resolve_api_config() -> Optional[Tuple[str, str, str]]:
    """
    Works out which Whisper endpoint to call from the keys present in .env.
    Precedence: WHISPER_API_KEY → OPENAI_API_KEY → GROQ_API_KEY.
    Returns (api_key, base_url, model) or None when no key is configured.
    """
    if settings.whisper_api_key:
        key, provider = settings.whisper_api_key, "openai"
    elif settings.openai_api_key:
        key, provider = settings.openai_api_key, "openai"
    elif settings.groq_api_key:
        key, provider = settings.groq_api_key, "groq"
    else:
        return None

    default_base, default_model = _PROVIDER_DEFAULTS[provider]
    base = (settings.whisper_api_base or default_base).rstrip("/")
    model = settings.whisper_api_model or default_model
    return key, base, model


def is_configured() -> bool:
    """True when Whisper can actually run — an API key, or the local package."""
    if settings.whisper_backend == "local":
        return True  # availability is only known once we try to import it
    return resolve_api_config() is not None


def _fail(error: str, engine: str = "whisper") -> Dict[str, Any]:
    return {
        "success": False, "text": "", "language": "", "confidence": 0.0,
        "segments": [], "engine": engine, "error": error,
    }


def _confidence_from_segments(segments: list) -> float:
    """
    Whisper reports no per-transcript confidence, but verbose_json segments
    carry avg_logprob (mean token log-probability). exp() maps it back to a
    rough 0–1 likelihood; averaging over segments gives a usable signal.
    """
    logprobs = [
        s["avg_logprob"] for s in segments
        if isinstance(s, dict) and isinstance(s.get("avg_logprob"), (int, float))
    ]
    if not logprobs:
        return 0.9  # engine gave us nothing to go on — match the old default
    return round(min(1.0, math.exp(sum(logprobs) / len(logprobs))), 3)


# ─────────────────────────────────────────────────────────────────────────────
#  API backend
# ─────────────────────────────────────────────────────────────────────────────

async def _transcribe_via_api(audio_path: str, language: Optional[str]) -> Dict[str, Any]:
    config = resolve_api_config()
    if not config:
        return _fail(
            "No Whisper API key configured. Set WHISPER_API_KEY (or OPENAI_API_KEY / "
            "GROQ_API_KEY) in .env."
        )
    api_key, base, model = config

    ext = os.path.splitext(audio_path)[1].lower()
    send_path, temp_path = audio_path, None

    if ext not in SUPPORTED_CONTAINERS:
        temp_path = await to_wav(audio_path)
        if temp_path:
            send_path = temp_path
        else:
            log.warning("unsupported_audio_container_sent_as_is", ext=ext)

    try:
        size = os.path.getsize(send_path)
        if size > MAX_UPLOAD_BYTES:
            return _fail(
                f"Recording is {size // (1024 * 1024)} MB — above the "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB Whisper upload limit. "
                "Send a shorter clip."
            )

        with open(send_path, "rb") as f:
            audio_bytes = f.read()

        data = {"model": model, "response_format": "verbose_json", "temperature": "0"}
        # Blank language = auto-detect, which is what we want for Hindi/Hinglish/Tamil.
        if language:
            data["language"] = language

        log.info("starting_whisper_transcription", model=model, base=base, bytes=size)

        async with httpx.AsyncClient(timeout=settings.whisper_timeout) as client:
            response = await client.post(
                f"{base}/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (os.path.basename(send_path), audio_bytes, "application/octet-stream")},
                data=data,
            )

        if response.status_code != 200:
            detail = response.text[:300]
            log.warning("whisper_api_error", status=response.status_code, detail=detail)
            return _fail(f"Whisper API returned {response.status_code}: {detail}")

        payload = response.json()
        text = (payload.get("text") or "").strip()
        segments = payload.get("segments") or []

        if not text:
            log.info("whisper_no_speech")
            return _fail("No intelligible speech found in the recording.")

        log.info("whisper_transcription_completed", chars=len(text),
                 language=payload.get("language"))
        return {
            "success": True,
            "text": text,
            "language": payload.get("language") or "",
            "confidence": _confidence_from_segments(segments),
            "segments": segments,
            "engine": "whisper",
            "error": None,
        }

    except Exception as e:
        log.warning("whisper_transcription_failed", error=str(e))
        return _fail(str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Local backend
# ─────────────────────────────────────────────────────────────────────────────

def _get_local_model():
    global _local_model
    if _local_model is None:
        import whisper  # optional dep — only needed for WHISPER_BACKEND=local
        log.info("loading_local_whisper", size=settings.whisper_model_size)
        _local_model = whisper.load_model(
            settings.whisper_model_size,
            device="cuda" if settings.use_gpu else "cpu",
        )
        log.info("local_whisper_loaded")
    return _local_model


def _run_local(audio_path: str, language: Optional[str]) -> Dict[str, Any]:
    result = _get_local_model().transcribe(
        audio_path,
        language=language or None,  # None = auto-detect
        task="transcribe",
        fp16=settings.use_gpu,
    )
    text = (result.get("text") or "").strip()
    segments = result.get("segments") or []
    if not text:
        return _fail("No intelligible speech found in the recording.", engine="whisper-local")
    return {
        "success": True,
        "text": text,
        "language": result.get("language") or "",
        "confidence": _confidence_from_segments(segments),
        "segments": segments,
        "engine": "whisper-local",
        "error": None,
    }


async def _transcribe_via_local(audio_path: str, language: Optional[str]) -> Dict[str, Any]:
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(_run_local, audio_path, language)
        )
    except ImportError:
        return _fail(
            "WHISPER_BACKEND=local but the `openai-whisper` package is not installed. "
            "Run `pip install openai-whisper`, or switch to WHISPER_BACKEND=api.",
            engine="whisper-local",
        )
    except Exception as e:
        log.warning("local_whisper_failed", error=str(e))
        return _fail(str(e), engine="whisper-local")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def transcribe_with_whisper(
    audio_path: str, language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transcribes a voice note with Whisper.
    `language` overrides auto-detect with an ISO-639-1 code ("hi", "en", "ta").
    Returns {success, text, language, confidence, segments, engine, error}.
    """
    if not audio_path or not os.path.exists(audio_path):
        return _fail("Audio file does not exist.")

    language = language or settings.whisper_language or None

    if settings.whisper_backend == "local":
        return await _transcribe_via_local(audio_path, language)
    return await _transcribe_via_api(audio_path, language)
