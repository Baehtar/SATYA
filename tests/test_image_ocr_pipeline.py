"""
tests/test_image_ocr_pipeline.py — Comprehensive Test Suite for Satya Image-to-Fake-News OCR Pipeline.
Tests OCR normalizer, claim extraction, NLI verifier, evidence aggregator, and Telegram card formatting.
"""
import sys
import unittest
import asyncio

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

from services.ocr.normalizer import normalize_ocr_result, detect_script
from src.pipelines.nli_verifier import verify_claim_nli
from src.verdict.evidence_aggregator import aggregate_evidence
from bot.response import format_verdict
from src.models.schemas import ClaimAnalysis, Verdict, FactCheckMatch, LanguageCode


class TestImageOCRPipeline(unittest.TestCase):

    def test_01_ocr_normalization_english(self):
        raw = "BREAKING NEWS\n\nMinistry sources confirmed that from 1 Sept 2026 all UPI above Rs 5000 require Aadhaar OTP. PLEASE SHARE"
        res = normalize_ocr_result(raw)
        self.assertTrue(res["has_readable_text"])
        self.assertEqual(res["script"], "Latin")
        self.assertNotIn("BREAKING NEWS", res["cleaned_text"])
        self.assertNotIn("PLEASE SHARE", res["cleaned_text"])

    def test_02_ocr_normalization_hindi_devanagari(self):
        raw = "सरकार ने नया UPI नियम लागू किया: 5000 रुपये से अधिक लेनदेन पर आधार ओटीपी अनिवार्य"
        res = normalize_ocr_result(raw)
        self.assertTrue(res["has_readable_text"])
        self.assertEqual(res["script"], "Devanagari")

    def test_03_ocr_normalization_hinglish(self):
        raw = "Modi ne naya UPI niyam lagu kiya hai kal se sabhi 5000 rs ke uper aadhar otp lagega"
        res = normalize_ocr_result(raw)
        self.assertTrue(res["has_readable_text"])
        self.assertEqual(res["script"], "Latin")

    def test_04_ocr_normalization_tamil(self):
        raw = "அரசு புதிய UPI விதியை அறிவித்துள்ளது: ரூ 5000 மேல் ஆதாா் OTP அவசியம்"
        res = normalize_ocr_result(raw)
        self.assertTrue(res["has_readable_text"])
        self.assertEqual(res["script"], "Tamil")

    def test_05_ocr_normalization_tanglish(self):
        raw = "Arasu pudhiya UPI vidhiyai arivithullathu 5000 kku mela Aadhaar OTP kettu irukkanga"
        res = normalize_ocr_result(raw)
        self.assertTrue(res["has_readable_text"])

    def test_06_no_readable_text_image(self):
        raw = "   "
        res = normalize_ocr_result(raw)
        self.assertFalse(res["has_readable_text"])

    def test_07_nli_verification_contradiction(self):
        async def _test():
            claim = "UPI transactions above ₹5,000 will require Aadhaar OTP from September 2026."
            evidence = "NPCI has explicitly refuted claims requiring Aadhaar OTP for UPI transactions above ₹5,000 as false."
            res = await verify_claim_nli(claim, evidence)
            self.assertIn(res["nli_label"], ["CONTRADICTION", "NEUTRAL"])
        asyncio.run(_test())

    def test_08_evidence_fusion_ai_image_true_claim(self):
        # AI image (0.92 artificial score) with TRUE factual claim
        text_analysis = ClaimAnalysis(
            raw_text="Over 100 ABVP workers detained during Jharkhand Assembly march",
            extracted_claim="Over 100 ABVP workers were detained during a march to the Jharkhand Assembly",
            text_verdict=Verdict.LIKELY_TRUE,
            text_verdict_confidence=0.90,
            matches=[
                FactCheckMatch(
                    source_name="The Economic Times",
                    source_url="https://economictimes.indiatimes.com/news",
                    original_claim="ABVP workers detained marching to Jharkhand Assembly",
                    fact_check_verdict="REPORTED_NEWS",
                    match_confidence=0.90
                )
            ]
        )
        ocr_meta = {"cleaned_text": text_analysis.raw_text, "language": "EN"}

        fused = aggregate_evidence(text_analysis, image_ai_score=0.92, ocr_meta=ocr_meta)
        self.assertEqual(fused["verdict"], "LIKELY_TRUE")
        self.assertIn("AI-Generated", fused["image_note"])

    def test_09_evidence_fusion_real_image_false_claim(self):
        # Genuine image (0.05 artificial score) with FALSE claim
        text_analysis = ClaimAnalysis(
            raw_text="Government giving 1 crore rupees allowance after Aadhaar verification",
            extracted_claim="Government is giving 1 crore rupees allowance to every Aadhaar card holder",
            text_verdict=Verdict.LIKELY_FALSE,
            text_verdict_confidence=0.95,
            matches=[
                FactCheckMatch(
                    source_name="PIB Fact Check",
                    source_url="https://factcheck.pib.gov.in/1cr-fake-claim",
                    original_claim="Claim of 1 crore allowance from govt is Fake",
                    fact_check_verdict="FALSE",
                    match_confidence=0.95
                )
            ]
        )
        ocr_meta = {"cleaned_text": text_analysis.raw_text, "language": "EN"}

        fused = aggregate_evidence(text_analysis, image_ai_score=0.05, ocr_meta=ocr_meta)
        self.assertEqual(fused["verdict"], "LIKELY_FALSE")
        self.assertEqual(fused["confidence_level"], "HIGH")

    def test_10_telegram_verdict_card_formatting(self):
        result = {
            "verdict": "LIKELY_FALSE",
            "confidence": 0.91,
            "explanation": "Extracted Claim: <i>'UPI transactions above ₹5,000 require Aadhaar OTP'</i>\n\nEvidence: Fact-checks debunk this claim.",
            "sources": [{"name": "PIB Fact Check", "url": "https://factcheck.pib.gov.in"}],
            "language": "EN"
        }
        card = format_verdict(result)
        self.assertIn("🔴 <b>LIKELY FALSE</b>", card)
        self.assertIn("Confidence: █████████░ 91.0%", card)
        self.assertIn("PIB Fact Check", card)


if __name__ == "__main__":
    unittest.main()
