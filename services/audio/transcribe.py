"""
services/audio/transcribe.py — Speech-to-text for voice notes.

Two engines, chosen by STT_ENGINE:

  whisper — Whisper via an OpenAI-compatible API (or the local package).
            Best multilingual accuracy; see services/audio/whisper_stt.py.
  gemini  — the Gemini API the rest of Satya already depends on (OCR + claim
            extraction), so voice notes work with no extra key at all.

STT_ENGINE=auto (the default) picks Whisper when a Whisper key is configured
and falls back to Gemini otherwise — and if the chosen engine errors out, the
other one is tried before giving up.

Telegram sends OGG/Opus and browsers send WebM/Opus. Gemini accepts OGG but not
WebM, so unsupported containers are transcoded to 16 kHz mono WAV with ffmpeg
when it is installed. Without ffmpeg the file is sent as-is and, if the API
rejects it, the caller gets an honest failure instead of a fabricated verdict.
"""
import os
import structlog
from typing import Any, Dict
from google import genai
from google.genai import types
from src.config import settings
from services.audio.convert import to_wav
from services.audio import whisper_stt

log = structlog.get_logger(__name__)

# Containers Gemini accepts directly → their MIME types.
SUPPORTED_AUDIO_MIME = {
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".m4a": "audio/aac",
    ".aiff": "audio/aiff",
}

TRANSCRIPTION_PROMPT = """You are a precision speech-to-text engine for news verification.

Transcribe the speech in this audio recording verbatim.

Instructions:
1. The speaker may use English, Hindi, Hinglish, Tamil or Tanglish. Transcribe in the language actually spoken, keeping the original script.
2. Preserve names, numbers, dates and amounts exactly as spoken.
3. Do not translate, summarise, correct or add commentary.
4. Output ONLY the transcript. If the recording contains no intelligible speech, output exactly NO_SPEECH.
"""


def _get_client():
    if not settings.gemini_api_key:
        return None
    return genai.Client(api_key=settings.gemini_api_key)


async def _transcribe_with_gemini(audio_path: str) -> Dict[str, Any]:
    """Transcribes a voice note with Gemini. Returns {success, text, engine, error}."""
    client = _get_client()
    if not client:
        return {"success": False, "text": "", "engine": "none", "error": "GEMINI_API_KEY is not configured."}

    ext = os.path.splitext(audio_path)[1].lower()
    send_path = audio_path
    temp_path = None

    if ext not in SUPPORTED_AUDIO_MIME:
        temp_path = await to_wav(audio_path)
        if temp_path:
            send_path = temp_path
        else:
            log.warning("unsupported_audio_container_sent_as_is", ext=ext)

    mime = SUPPORTED_AUDIO_MIME.get(os.path.splitext(send_path)[1].lower(), "audio/ogg")
    log.info("starting_transcription", path=send_path, mime=mime)

    try:
        with open(send_path, "rb") as f:
            audio_bytes = f.read()

        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                TRANSCRIPTION_PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1000,
            ),
        )

        text = (response.text or "").strip()
        if text == "NO_SPEECH" or not text:
            log.info("transcription_no_speech")
            return {"success": False, "text": "", "engine": "gemini", "error": "No intelligible speech found in the recording."}

        log.info("transcription_completed", chars=len(text))
        return {"success": True, "text": text, "engine": "gemini", "error": None}

    except Exception as e:
        log.warning("transcription_failed", error=str(e))
        return {"success": False, "text": "", "engine": "gemini", "error": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _normalise(result: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantees every caller sees the same keys whichever engine ran."""
    return {
        "success": result.get("success", False),
        "text": result.get("text", ""),
        "language": result.get("language", ""),
        "confidence": result.get("confidence", 0.9 if result.get("success") else 0.0),
        "segments": result.get("segments", []),
        "engine": result.get("engine", "none"),
        "error": result.get("error"),
    }


def _engine_order() -> list[str]:
    """
    Resolves STT_ENGINE into the engines to try, in order.
    'auto' prefers Whisper when a key is configured, and always keeps the other
    engine as a fallback so a single provider outage doesn't kill voice notes.
    """
    engine = (settings.stt_engine or "auto").strip().lower()
    if engine == "whisper":
        return ["whisper", "gemini"]
    if engine == "gemini":
        return ["gemini", "whisper"]
    return ["whisper", "gemini"] if whisper_stt.is_configured() else ["gemini", "whisper"]


async def transcribe_audio(audio_path: str, language: str | None = None) -> Dict[str, Any]:
    """
    Transcribes a voice note with whichever STT engine is configured.
    `language` forces an ISO-639-1 code; blank/None auto-detects.
    Returns {success, text, language, confidence, segments, engine, error}.
    """
    if not audio_path or not os.path.exists(audio_path):
        return _normalise({"error": "Audio file does not exist."})

    if not whisper_stt.is_configured() and not settings.gemini_api_key:
        return _normalise({"error": (
            "No speech-to-text engine is configured. Set WHISPER_API_KEY (or "
            "OPENAI_API_KEY / GROQ_API_KEY) for Whisper, or GEMINI_API_KEY, in .env."
        )})

    first_error = None

    for engine in _engine_order():
        if engine == "whisper":
            if not whisper_stt.is_configured():
                first_error = first_error or (
                    "No Whisper API key configured. Set WHISPER_API_KEY (or OPENAI_API_KEY / "
                    "GROQ_API_KEY) in .env."
                )
                continue
            result = await whisper_stt.transcribe_with_whisper(audio_path, language)
        else:
            result = await _transcribe_with_gemini(audio_path)

        if result.get("success"):
            return _normalise(result)

        first_error = first_error or result.get("error")
        log.info("stt_engine_failed_trying_next", engine=engine, error=result.get("error"))

    return _normalise({"error": first_error or "No speech-to-text engine is configured."})
