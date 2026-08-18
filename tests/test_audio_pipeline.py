"""
tests/test_audio_pipeline.py — Comprehensive unit and end-to-end integration tests for Audio/Whisper pipeline.
Tests Whisper transcription, preprocessor, claim extraction, fact check integration, and error handling.
"""
import os
import unittest
import tempfile
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from services.audio.audio_preprocessor import (
    preprocess_audio,
    cleanup_temp_audio,
    check_ffmpeg_available,
)
from services.audio.whisper_service import WhisperService, transcribe, WHISPER_MODEL_NAME
from services.ml_service import check_voice


class TestAudioPipeline(unittest.TestCase):

    def setUp(self):
        # Create a temporary dummy WAV file for testing
        self.temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_wav.write(b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 100)
        self.temp_wav.close()

        # Create an empty corrupt file
        self.temp_corrupt = tempfile.NamedTemporaryFile(suffix=".invalid", delete=False)
        self.temp_corrupt.close()

    def tearDown(self):
        if os.path.exists(self.temp_wav.name):
            os.unlink(self.temp_wav.name)
        if os.path.exists(self.temp_corrupt.name):
            os.unlink(self.temp_corrupt.name)

    def test_audio_preprocessor_dummy_wav(self):
        """Verifies preprocessor returns file path and handles WAV natively."""
        effective_path, is_temp = preprocess_audio(self.temp_wav.name)
        self.assertTrue(os.path.exists(effective_path))
        self.assertFalse(is_temp)
        cleanup_temp_audio(effective_path, is_temp)

    def test_audio_preprocessor_empty_file(self):
        """Verifies preprocessor raises ValueError on empty audio file."""
        with self.assertRaises(ValueError):
            preprocess_audio(self.temp_corrupt.name)

    def test_audio_preprocessor_missing_file(self):
        """Verifies preprocessor raises FileNotFoundError on missing file."""
        with self.assertRaises(FileNotFoundError):
            preprocess_audio("non_existent_audio_path.wav")

    def test_whisper_service_caching(self):
        """Verifies WhisperService SHA-256 hash caching returns cached response on repeated audio."""
        async def run_test():
            mock_pipeline_output = {
                "text": "UPI transactions above 5000 require Aadhaar OTP.",
                "chunks": [{"timestamp": (0.0, 4.5), "language": "en"}]
            }

            service = WhisperService()
            
            with patch.object(service, "_load_pipeline") as mock_load:
                mock_pipe = MagicMock()
                mock_pipe.return_value = mock_pipeline_output
                mock_load.return_value = mock_pipe

                # First call
                res1 = await service.transcribe(self.temp_wav.name)
                self.assertEqual(res1["text"], "UPI transactions above 5000 require Aadhaar OTP.")
                self.assertEqual(res1["language"], "en")

                # Second call (cache hit)
                res2 = await service.transcribe(self.temp_wav.name)
                self.assertEqual(res2["text"], "UPI transactions above 5000 require Aadhaar OTP.")

                # Model should only have been loaded once and executed once
                self.assertEqual(mock_pipe.call_count, 1)

        asyncio.run(run_test())

    def test_check_voice_pipeline_end_to_end(self):
        """Tests complete end-to-end check_voice pipeline from audio to verdict."""
        async def run_test():
            mock_transcription = {
                "text": "Government announced that all UPI transactions above ₹5,000 will require Aadhaar OTP.",
                "language": "en",
                "duration_seconds": 5.2,
                "processing_time_seconds": 0.8,
                "device": "cpu"
            }

            mock_analysis = MagicMock()
            mock_analysis.extracted_claim = "UPI transactions above ₹5,000 require mandatory Aadhaar OTP."
            mock_analysis.text_verdict.value = "LIKELY_FALSE"
            mock_analysis.text_verdict_confidence = 0.88
            mock_analysis.language.value = "EN"
            mock_analysis.matches = [
                MagicMock(source_name="PIB Fact Check", source_url="https://factcheck.pib.gov.in/claim1", original_claim="Fake UPI Aadhaar OTP rumor", fact_check_verdict="FALSE"),
                MagicMock(source_name="BOOM Live", source_url="https://boomlive.in/fact-check/upi-aadhaar", original_claim="UPI OTP viral claim", fact_check_verdict="FALSE")
            ]

            progress_messages = []

            async def mock_progress(msg: str):
                progress_messages.append(msg)

            with patch("services.audio.whisper_service.transcribe", AsyncMock(return_value=mock_transcription)), \
                 patch("src.pipelines.text.pipeline.run_text_pipeline", AsyncMock(return_value=mock_analysis)):

                result = await check_voice(self.temp_wav.name, progress_callback=mock_progress)

                self.assertEqual(result["type"], "voice")
                self.assertEqual(result["verdict"], "LIKELY_FALSE")
                self.assertEqual(result["confidence"], 0.88)
                self.assertEqual(result["transcript"], mock_transcription["text"])
                self.assertEqual(result["extracted_claim"], mock_analysis.extracted_claim)
                self.assertEqual(len(result["sources"]), 2)
                self.assertIn("🎙️ Transcribing audio...", progress_messages)
                self.assertIn("🔎 Checking the claim...", progress_messages)
                self.assertIn("⚖️ Comparing evidence...", progress_messages)
                self.assertIn("✅ Analysis complete", progress_messages)

        asyncio.run(run_test())

    def test_check_voice_empty_transcription(self):
        """Tests check_voice handling when Whisper returns empty transcript."""
        async def run_test():
            mock_transcription = {
                "text": "",
                "language": "unknown",
                "duration_seconds": 0.0,
                "processing_time_seconds": 0.1,
                "device": "cpu"
            }

            with patch("services.audio.whisper_service.transcribe", AsyncMock(return_value=mock_transcription)):
                result = await check_voice(self.temp_wav.name)

                self.assertEqual(result["type"], "voice")
                self.assertEqual(result["verdict"], "UNVERIFIABLE")
                self.assertEqual(result["confidence"], 0.0)
                self.assertIn("couldn't transcribe", result["explanation"])

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
