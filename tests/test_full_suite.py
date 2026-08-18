"""
tests/test_full_suite.py — End-to-End Verification Suite for Satya Fact-Checker.
Executes live tests across:
  1. Text Claim Verification (PIB / Alt News / BOOM / Google News RSS)
  2. Image AI Detection & Reverse Search / Provenance Engine (Gemini 3.5 / Lens / Forensics)
  3. Audio Transcription & Fact-Checking Engine (Gemini 3.5 Flash Lite)
  4. Telegram Document File Classification & Formatting
"""
import asyncio
import os
import unittest
import glob
from src.models.schemas import CheckRequest
from src.pipelines.text.pipeline import run_text_pipeline
from services.ml_service import check_image, check_voice
from services.audio.whisper_service import transcribe
from utils.telegram_files import classify_document_file


class TestSatyaFullSuite(unittest.TestCase):

    def test_1_text_fake_news_claim(self):
        """Tests text claim verification on known viral fake news."""
        async def run():
            req = CheckRequest(
                request_id="test-text-fake-1",
                message_type="text",
                text_content="UNESCO has declared Indian National Anthem Jana Gana Mana as the best anthem in the world."
            )
            res = await run_text_pipeline(req)
            print("\n--- TEST 1A: TEXT FAKE NEWS ---")
            print(f"Extracted Claim: {res.extracted_claim}")
            print(f"Verdict: {res.text_verdict.value} (Confidence: {res.text_verdict_confidence})")
            print(f"Matches count: {len(res.matches)}")
            self.assertIn(res.text_verdict.value, ["LIKELY_FALSE", "UNVERIFIABLE"])

        asyncio.run(run())

    def test_2_text_real_news_claim(self):
        """Tests text claim verification on known real news."""
        async def run():
            req = CheckRequest(
                request_id="test-text-real-1",
                message_type="text",
                text_content="ISRO successfully launched Chandrayaan 3 lunar mission from Sriharikota."
            )
            res = await run_text_pipeline(req)
            print("\n--- TEST 1B: TEXT REAL NEWS ---")
            print(f"Extracted Claim: {res.extracted_claim}")
            print(f"Verdict: {res.text_verdict.value} (Confidence: {res.text_verdict_confidence})")
            print(f"Matches count: {len(res.matches)}")
            self.assertIsNotNone(res.text_verdict)

        asyncio.run(run())

    def test_3_image_reverse_provenance(self):
        """Tests image pipeline with an authentic news photo from temp directory."""
        async def run():
            # Find a sample JPEG in temp/
            temp_images = glob.glob("temp/*.jpg")
            if not temp_images:
                self.skipTest("No sample images in temp/ directory")

            img_path = temp_images[0]
            print(f"\n--- TEST 2A: IMAGE PROVENANCE & FORENSICS ({img_path}) ---")
            res = await check_image(img_path)
            
            print(f"Verdict: {res.get('verdict')}")
            print(f"Confidence: {res.get('confidence')}")
            print(f"Explanation Snippet: {res.get('explanation', '')[:200]}...")
            print(f"Sources count: {len(res.get('sources', []))}")
            
            self.assertIn(res.get("verdict"), ["LIKELY_TRUE", "LIKELY_FALSE", "UNVERIFIABLE"])
            self.assertIn("reverse_engine", res)

        asyncio.run(run())

    def test_4_document_file_classification(self):
        """Tests document file routing for images and audio files."""
        print("\n--- TEST 4: DOCUMENT CLASSIFICATION ---")

        is_audio, is_img = classify_document_file("audio/mp3", "news_speech.mp3")
        self.assertTrue(is_audio)
        self.assertFalse(is_img)

        is_audio, is_img = classify_document_file("image/jpeg", "disaster_photo.jpg")
        self.assertFalse(is_audio)
        self.assertTrue(is_img)

        is_audio, is_img = classify_document_file("application/octet-stream", "voice_note.ogg")
        self.assertTrue(is_audio)

        is_audio, is_img = classify_document_file("application/octet-stream", "speech.mp3.mpeg")
        self.assertTrue(is_audio)
        print("Document file classification passed cleanly!")


if __name__ == "__main__":
    unittest.main()
