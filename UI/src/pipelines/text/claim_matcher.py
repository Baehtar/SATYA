"""Claim matcher: evaluates degree of match between user claim and fact-check reports."""
import logging
import asyncio
import json
from typing import List
from google import genai
from src.config import settings
from src.models.schemas import ClaimAnalysis, FactCheckMatch, VerdictLevel, ClaimType

logger = logging.getLogger(__name__)


async def match_claims(user_claim: str, fact_checks: List[FactCheckMatch]) -> ClaimAnalysis:
    if not fact_checks:
        return ClaimAnalysis(
            extracted_claim=user_claim,
            claim_type=ClaimType.OTHER,
            is_checkable=True,
            entities=[],
            overall_verdict=VerdictLevel.UNVERIFIABLE,
            fact_checks=[],
            confidence=0.4
        )

    # Heuristic match determination
    verdict = VerdictLevel.UNVERIFIABLE
    confidence = 0.5

    # Check if any retrieved fact check indicates FALSE / HOAX / MISLEADING
    false_indicators = ["false", "misleading", "hoax", "fake", "debunked", "incorrect", "untrue", "phishing"]
    true_indicators = ["true", "correct", "authentic", "confirmed", "verified"]

    has_false = False
    has_true = False

    for fc in fact_checks:
        verdict_str = fc.verdict.lower()
        if any(ind in verdict_str for ind in false_indicators):
            has_false = True
        elif any(ind in verdict_str for ind in true_indicators):
            has_true = True

    if has_false:
        verdict = VerdictLevel.LIKELY_FALSE
        confidence = 0.85
    elif has_true:
        verdict = VerdictLevel.LIKELY_TRUE
        confidence = 0.85

    # Refine match via Gemini 2.5 Flash if available
    if settings.gemini_api_key:
        try:
            client = genai.Client(api_key=settings.gemini_api_key)
            checks_summary = "\n".join([f"- {fc.source_name}: {fc.verdict} ({fc.summary})" for fc in fact_checks])
            prompt = f"""Compare the user's claim with the retrieved fact-check articles.
User claim: "{user_claim}"

Fact Check Reports:
{checks_summary}

Determine the overall verdict for the user claim: LIKELY_FALSE, LIKELY_TRUE, or UNVERIFIABLE.
Also return a numerical confidence score between 0.0 and 1.0.

Respond in JSON format only:
{{"verdict": "LIKELY_FALSE", "confidence": 0.88}}"""

            response = await asyncio.to_thread(
                client.models.generate_content,
                model=settings.gemini_model,
                contents=prompt
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.replace("```json", "").replace("```", "").strip()
            res = json.loads(raw)

            verdict_val = res.get("verdict", "UNVERIFIABLE").upper()
            if verdict_val in VerdictLevel.__members__:
                verdict = VerdictLevel[verdict_val]
            confidence = float(res.get("confidence", confidence))
        except Exception as e:
            logger.warning(f"Gemini claim matching fallback: {e}")

    return ClaimAnalysis(
        extracted_claim=user_claim,
        claim_type=ClaimType.OTHER,
        is_checkable=True,
        entities=[],
        overall_verdict=verdict,
        fact_checks=fact_checks,
        confidence=confidence
    )
