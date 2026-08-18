"""
services/audio/audio_preprocessor.py — Audio inspection, format validation & optional conversion.

Handles Telegram voice notes (OGG/Opus) and standard audio formats (MP3, WAV, M4A, FLAC).
Provides safe cleanup of transient preprocessed files.
"""
import os
import shutil
import subprocess
import tempfile
import structlog
from typing import Tuple, Optional

log = structlog.get_logger(__name__)


def check_ffmpeg_available() -> bool:
    """Checks whether system ffmpeg executable is installed and available in PATH."""
    return shutil.which("ffmpeg") is not None


def preprocess_audio(audio_path: str) -> Tuple[str, bool]:
    """
    Validates and preprocesses an input audio file for Whisper.
    
    Returns:
        Tuple[str, bool]: (effective_path, is_temp_file)
        If conversion/copy was performed, effective_path will be a temp file (is_temp_file=True).
        Otherwise returns (original_path, False).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_size = os.path.getsize(audio_path)
    if file_size == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {audio_path}")

    ext = os.path.splitext(audio_path)[1].lower()

    # Standard formats natively readable by soundfile/librosa/whisper
    # Standard Telegram voice OGG/Opus is often readable directly by librosa/soundfile or transformers
    # If conversion is needed and ffmpeg is installed, convert to mono 16kHz WAV for optimal Whisper input.
    if ext in [".wav", ".mp3", ".flac"]:
        return audio_path, False

    # For OGG/M4A/AAC/WMA or unverified extensions, attempt conversion if FFmpeg is available
    if check_ffmpeg_available():
        try:
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav.close()

            cmd = [
                "ffmpeg",
                "-y",
                "-i", audio_path,
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                temp_wav.name
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False
            )
            if result.returncode == 0 and os.path.exists(temp_wav.name) and os.path.getsize(temp_wav.name) > 0:
                log.info("ffmpeg_conversion_successful", original=audio_path, converted=temp_wav.name)
                return temp_wav.name, True
            else:
                log.warning("ffmpeg_conversion_non_zero_exit", stderr=result.stderr.decode("utf-8", errors="ignore"))
                if os.path.exists(temp_wav.name):
                    os.unlink(temp_wav.name)
        except Exception as e:
            log.warning("ffmpeg_conversion_failed_falling_back_to_raw", error=str(e))

    # If FFmpeg not present or conversion failed, return original path for transformers/soundfile/librosa
    return audio_path, False


def cleanup_temp_audio(path: str, is_temp: bool) -> None:
    """Safely deletes a temporary audio file if is_temp is True."""
    if is_temp and path and os.path.exists(path):
        try:
            os.unlink(path)
            log.info("cleaned_up_temp_audio", path=path)
        except Exception as e:
            log.warning("failed_to_cleanup_temp_audio", path=path, error=str(e))
