"""
services/audio/transcribe.py — Speech-to-text for voice notes.

Uses the Gemini API that the rest of Satya already depends on (OCR + claim
extraction), so voice notes work without pulling in Whisper/torch.

Telegram sends OGG/Opus and browsers send WebM/Opus. Gemini accepts OGG but not
WebM, so unsupported containers are transcoded to 16 kHz mono WAV with ffmpeg
when it is installed. Without ffmpeg the file is sent as-is and, if the API
rejects it, the caller gets an honest failure instead of a fabricated verdict.
"""
import asyncio
import os
import shutil
import tempfile
import structlog
from typing import Any, Dict
from google import genai
from google.genai import types
from src.config import settings

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


async def _to_wav(audio_path: str) -> str | None:
    """Transcodes any ffmpeg-readable audio to 16 kHz mono WAV. Returns the new path."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    process = await asyncio.create_subprocess_exec(
        ffmpeg, "-y", "-loglevel", "error", "-i", audio_path,
        "-ac", "1", "-ar", "16000", out_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()

    if process.returncode != 0 or not os.path.getsize(out_path):
        log.warning("ffmpeg_transcode_failed", error=stderr.decode()[:200])
        if os.path.exists(out_path):
            os.remove(out_path)
        return None

    return out_path


async def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """
    Transcribes a voice note.
    Returns {success, text, engine, error}.
    """
    if not audio_path or not os.path.exists(audio_path):
        return {"success": False, "text": "", "engine": "none", "error": "Audio file does not exist."}

    client = _get_client()
    if not client:
        return {"success": False, "text": "", "engine": "none", "error": "GEMINI_API_KEY is not configured."}

    ext = os.path.splitext(audio_path)[1].lower()
    send_path = audio_path
    temp_path = None

    if ext not in SUPPORTED_AUDIO_MIME:
        temp_path = await _to_wav(audio_path)
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
