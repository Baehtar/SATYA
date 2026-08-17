"""
services/audio/convert.py — ffmpeg helpers shared by the STT engines.

Telegram sends OGG/Opus and browsers send WebM/Opus. Each engine accepts a
different set of containers, so anything unsupported is transcoded to 16 kHz
mono WAV — the format every engine reads and the sample rate Whisper uses
internally anyway.
"""
import asyncio
import os
import shutil
import tempfile
import structlog

log = structlog.get_logger(__name__)


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


async def to_wav(audio_path: str) -> str | None:
    """
    Transcodes any ffmpeg-readable audio to 16 kHz mono WAV.
    Returns the new temp path, or None when ffmpeg is missing or fails.
    The caller owns the returned file and must delete it.
    """
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


async def extract_audio_from_video(video_path: str) -> str | None:
    """
    Extracts the audio track from any ffmpeg-readable video file and writes it
    as a 16 kHz mono WAV.  Returns the temp path, or None when ffmpeg is
    missing, the file has no audio track, or the extraction fails.
    The caller owns the returned file and must delete it.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    process = await asyncio.create_subprocess_exec(
        ffmpeg, "-y", "-loglevel", "error",
        "-i", video_path,
        "-vn",            # drop the video stream
        "-ac", "1", "-ar", "16000",
        out_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()

    if process.returncode != 0 or not os.path.exists(out_path) or not os.path.getsize(out_path):
        log.warning("ffmpeg_video_audio_extraction_failed",
                    video=video_path, error=stderr.decode()[:200])
        if os.path.exists(out_path):
            os.remove(out_path)
        return None

    log.info("video_audio_extracted", video=video_path, wav=out_path)
    return out_path
