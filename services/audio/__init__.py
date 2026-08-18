"""
services/audio package — Audio preprocessing and Whisper speech recognition service.
"""
from services.audio.whisper_service import transcribe, get_whisper_service
from services.audio.audio_preprocessor import preprocess_audio, check_ffmpeg_available

__all__ = [
    "transcribe",
    "get_whisper_service",
    "preprocess_audio",
    "check_ffmpeg_available",
]
