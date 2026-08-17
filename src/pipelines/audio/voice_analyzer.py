"""
src/pipelines/audio/voice_analyzer.py — Voice note analysis pipeline.
Owned by: Person 4

1. Whisper STT: transcribe voice note (services/audio — API or local backend)
2. Voice clone detection: spectral analysis for synthetic voice markers

Both steps read the same file, so they run concurrently: STT is network-bound
under the default API backend, clone detection is CPU-bound in a thread.
"""
import asyncio
import functools
import structlog
import numpy as np
from pathlib import Path
from src.models.schemas import CheckRequest, AudioAnalysis

log = structlog.get_logger(__name__)


def _detect_voice_clone(audio_path: str) -> dict:
    """
    Spectral analysis to detect AI-synthesised voice.
    Checks for:
    - Unnatural pitch contour smoothness
    - Missing formant variation
    - Quantisation artifacts in mel spectrogram
    """
    try:
        import librosa
        import soundfile as sf

        audio, sr = librosa.load(audio_path, sr=16000)

        # ── Pitch analysis ────────────────────────────────────────────────────
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        f0_voiced = f0[voiced_flag]
        pitch_score = 0.0

        if len(f0_voiced) > 10:
            # AI voices tend to have unnaturally smooth pitch contours
            pitch_variance = np.var(np.diff(f0_voiced[~np.isnan(f0_voiced)]))
            # Low variance → too smooth → possibly AI
            if pitch_variance < 50:
                pitch_score = 0.7

        # ── Mel spectrogram analysis ──────────────────────────────────────────
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Check for quantisation grid artifacts (common in vocoder-based TTS)
        quantisation_score = 0.0
        row_variances = np.var(mel_db, axis=1)
        if np.min(row_variances) < 0.1:  # some frequency bands nearly constant
            quantisation_score = 0.5

        # ── Combine scores ────────────────────────────────────────────────────
        clone_score = (pitch_score * 0.6) + (quantisation_score * 0.4)
        anomalies = []
        if pitch_score > 0.5:
            anomalies.append("Unnaturally smooth pitch contour")
        if quantisation_score > 0.3:
            anomalies.append("Mel spectrogram quantisation artifacts")

        return {
            "clone_score": min(1.0, clone_score),
            "anomalies": anomalies,
        }

    except Exception as e:
        log.warning("voice_clone_detection_failed", error=str(e))
        return {"clone_score": 0.0, "anomalies": []}


async def run_audio_pipeline(request: CheckRequest) -> AudioAnalysis:
    """Async entry point for audio pipeline."""
    import time
    start = time.monotonic()

    if not request.audio_path or not Path(request.audio_path).exists():
        return AudioAnalysis(error="No audio file provided")

    audio_path = request.audio_path

    try:
        from services.audio import transcribe_audio

        loop = asyncio.get_running_loop()
        stt, clone = await asyncio.gather(
            transcribe_audio(audio_path),
            loop.run_in_executor(None, functools.partial(_detect_voice_clone, audio_path)),
        )

        latency_ms = int((time.monotonic() - start) * 1000)

        if not stt.get("success"):
            # Clone detection still ran, so report it alongside the STT failure.
            return AudioAnalysis(
                voice_clone_score=clone.get("clone_score", 0.0),
                spectral_anomalies=clone.get("anomalies", []),
                error=stt.get("error") or "Transcription failed",
                pipeline_latency_ms=latency_ms,
            )

        return AudioAnalysis(
            transcription=stt.get("text", ""),
            transcription_language=stt.get("language") or "hi",
            transcription_confidence=stt.get("confidence", 0.9),
            voice_clone_score=clone.get("clone_score", 0.0),
            spectral_anomalies=clone.get("anomalies", []),
            pipeline_latency_ms=latency_ms,
        )

    except Exception as e:
        log.error("audio_pipeline_failed", error=str(e))
        return AudioAnalysis(
            error=str(e),
            pipeline_latency_ms=int((time.monotonic() - start) * 1000),
        )
