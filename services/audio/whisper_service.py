"""
services/audio/whisper_service.py — Lazy-loaded singleton Hugging Face Whisper ASR pipeline.

Model: openai/whisper-large-v3-turbo
Supports GPU (CUDA) with CPU fallback, automatic chunking, language preservation,
and SHA-256 hash caching for identical audio files.
"""
import asyncio
import hashlib
import os
import time
import structlog
from typing import Dict, Any, Optional

from services.audio.audio_preprocessor import preprocess_audio, cleanup_temp_audio
from src.config import settings

log = structlog.get_logger(__name__)

WHISPER_MODEL_NAME = "openai/whisper-large-v3-turbo"


class WhisperService:
    """Singleton service manager for openai/whisper-large-v3-turbo model."""

    _instance: Optional["WhisperService"] = None

    def __init__(self):
        self._pipeline = None
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        self._device = "cuda" if has_cuda and getattr(settings, "use_gpu", True) else "cpu"
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "WhisperService":
        if cls._instance is None:
            cls._instance = WhisperService()
        return cls._instance

    def _load_pipeline(self):
        """Loads Hugging Face ASR pipeline once."""
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from transformers import pipeline

        log.info("loading_whisper_model", model=WHISPER_MODEL_NAME, device=self._device)
        start_time = time.monotonic()

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or getattr(settings, "hf_token", "") or getattr(settings, "hf_api_key", "")
        token_kwargs = {"token": hf_token} if hf_token else {}

        torch_dtype = torch.float16 if self._device == "cuda" else torch.float32

        try:
            device_id = 0 if self._device == "cuda" else -1
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL_NAME,
                torch_dtype=torch_dtype,
                device=device_id,
                chunk_length_s=30,
                **token_kwargs
            )
            elapsed = time.monotonic() - start_time
            log.info("whisper_model_loaded_successfully", device=self._device, load_time_seconds=round(elapsed, 2))
        except Exception as e:
            log.error("whisper_model_load_failed", error=str(e))
            raise RuntimeError(f"Failed to load Whisper model '{WHISPER_MODEL_NAME}': {e}") from e

        return self._pipeline

    def _calculate_audio_hash(self, audio_path: str) -> str:
        """Calculates SHA-256 hash of an audio file for caching."""
        hasher = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _sync_transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Synchronous transcription execution running inside thread executor."""
        pipe = self._load_pipeline()

        start_time = time.monotonic()

        # Run pipeline with return_timestamps and task="transcribe" to preserve original language
        result = pipe(
            audio_path,
            return_timestamps=True,
            generate_kwargs={"task": "transcribe"}
        )

        elapsed = time.monotonic() - start_time

        raw_text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()

        # Extract language if available from pipeline output chunks or chunks metadata
        language = "unknown"
        if isinstance(result, dict) and "chunks" in result and result["chunks"]:
            first_chunk = result["chunks"][0]
            if isinstance(first_chunk, dict) and "language" in first_chunk:
                language = first_chunk["language"]

        # Approximate audio duration from timestamps if chunk data present
        duration_seconds = 0.0
        if isinstance(result, dict) and "chunks" in result and result["chunks"]:
            last_chunk = result["chunks"][-1]
            if isinstance(last_chunk, dict) and "timestamp" in last_chunk and last_chunk["timestamp"]:
                timestamps = last_chunk["timestamp"]
                if len(timestamps) > 1 and timestamps[1] is not None:
                    duration_seconds = float(timestamps[1])

        return {
            "text": raw_text,
            "language": language,
            "duration_seconds": round(duration_seconds, 2),
            "processing_time_seconds": round(elapsed, 2),
            "device": self._device
        }

    async def _transcribe_with_gemini(self, audio_path: str) -> Optional[Dict[str, Any]]:
        gemini_key = os.getenv("GEMINI_API_KEY") or getattr(settings, "gemini_api_key", "")
        if not gemini_key:
            return None

        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            start_time = time.monotonic()

            ext = os.path.splitext(audio_path)[1].lower()
            mime_map = {
                ".mp3": "audio/mp3",
                ".wav": "audio/wav",
                ".ogg": "audio/ogg",
                ".m4a": "audio/m4a",
                ".aac": "audio/aac",
                ".flac": "audio/flac",
                ".mpeg": "audio/mpeg",
                ".mp4": "audio/mp4",
                ".3gp": "audio/3gpp",
                ".amr": "audio/amr"
            }
            mime_type = mime_map.get(ext, "audio/mp3")

            log.info("uploading_audio_to_gemini", path=audio_path, mime_type=mime_type)
            uploaded_file = await asyncio.to_thread(
                client.files.upload,
                file=audio_path,
                config={"mime_type": mime_type}
            )

            prompt = (
                "Transcribe the spoken audio recording accurately into text in its original language "
                "(such as English, Hindi, Tamil, Telugu, Malayalam, etc.). "
                "Output ONLY the transcribed spoken text. Do NOT add preamble, bullet points, titles, or notes."
            )

            response = None
            models_to_try = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-2.5-flash"]

            for m_name in models_to_try:
                try:
                    resp = await asyncio.to_thread(
                        client.models.generate_content,
                        model=m_name,
                        contents=[prompt, uploaded_file]
                    )
                    if resp and resp.text:
                        response = resp
                        log.info("gemini_audio_model_selected", model=m_name)
                        break
                except Exception as model_err:
                    log.warning("gemini_audio_model_failed", model=m_name, error=str(model_err))

            try:
                await asyncio.to_thread(client.files.delete, name=uploaded_file.name)
            except Exception:
                pass

            if response and response.text:
                text = response.text.strip()
                elapsed = time.monotonic() - start_time
                log.info("gemini_audio_transcription_successful", text_len=len(text), elapsed=round(elapsed, 2))
                return {
                    "text": text,
                    "language": "auto",
                    "duration_seconds": 0.0,
                    "processing_time_seconds": round(elapsed, 2),
                    "device": "gemini-flash"
                }
        except Exception as e:
            log.warning("gemini_audio_transcription_failed", error=str(e))

        return None

    async def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Asynchronously transcribes audio file.
        Uses audio hash cache, Gemini API engine, and local Whisper fallback.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        # SHA-256 audio hash cache check
        audio_hash = await asyncio.to_thread(self._calculate_audio_hash, audio_path)
        if audio_hash in self._cache:
            log.info("whisper_transcription_cache_hit", hash=audio_hash[:10])
            return self._cache[audio_hash]

        # 1. Try Gemini API Audio Engine (Fast, handles all formats & languages without local ffmpeg)
        gemini_res = await self._transcribe_with_gemini(audio_path)
        if gemini_res and gemini_res.get("text"):
            self._cache[audio_hash] = gemini_res
            return gemini_res

        # 2. Local Whisper / Hugging Face Fallback Engine
        effective_path, is_temp = await asyncio.to_thread(preprocess_audio, audio_path)

        try:
            res = await asyncio.to_thread(self._sync_transcribe, effective_path)
            self._cache[audio_hash] = res
            return res
        finally:
            cleanup_temp_audio(effective_path, is_temp)


def get_whisper_service() -> WhisperService:
    """Helper function to get WhisperService singleton instance."""
    return WhisperService.get_instance()


async def transcribe(audio_path: str) -> Dict[str, Any]:
    """
    Public entry point for transcribing audio.
    
    Returns:
    {
        "text": str,
        "language": str,
        "duration_seconds": float,
        "processing_time_seconds": float,
        "device": str
    }
    """
    service = get_whisper_service()
    return await service.transcribe(audio_path)
