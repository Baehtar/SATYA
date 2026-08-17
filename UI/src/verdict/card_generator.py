"""Verdict card generator: creates user-facing bilingual (English & Hindi) card outputs."""
import logging
import asyncio
import json
from google import genai
from src.config import settings
from src.models.schemas import EvidenceBundle, VerdictLevel, ConfidenceLevel, VerdictCard, FactCheckMatch

logger = logging.getLogger(__name__)


async def generate_verdict_card(
    evidence: EvidenceBundle,
    verdict: VerdictLevel,
    confidence: float,
    confidence_level: ConfidenceLevel
) -> VerdictCard:
    
    # Default fallbacks based on specific claim/image evidence
    english_explanation = "Based on our analysis, we could not definitively verify this claim. Always verify with official sources."
    hindi_explanation = "हमारे विश्लेषण के आधार पर, हम इस दावे को निश्चित रूप से सत्यापित नहीं कर सके। कृपया आधिकारिक स्रोतों से पुष्टि करें।"

    if evidence.claim_analysis and evidence.claim_analysis.fact_checks:
        fc = evidence.claim_analysis.fact_checks[0]
        if fc.summary:
            english_explanation = f"What we found: {fc.summary}"
            hindi_explanation = f"हमने क्या पाया: {fc.verdict} - {fc.summary}"

    elif evidence.image_analysis:
        flags = []
        if evidence.image_analysis.ai_detection and evidence.image_analysis.ai_detection.score > 0.8:
            flags.append("AI-generated visuals detected")
        if evidence.image_analysis.reverse_search and evidence.image_analysis.reverse_search.is_recycled:
            flags.append("Recycled image previously published online")
        if evidence.image_analysis.manipulation and evidence.image_analysis.manipulation.overall_score > 0.7:
            flags.append("Digital image manipulation/EXIF editing detected")

        if flags:
            english_explanation = "What we found: " + "; ".join(flags) + "."
            hindi_explanation = "हमने क्या पाया: इस तस्वीर में " + ", ".join(flags) + " की संभावना है।"

    if verdict == VerdictLevel.LIKELY_FALSE and not (evidence.claim_analysis and evidence.claim_analysis.fact_checks):
        english_explanation = "We found multiple indicators suggesting this content is misleading or unverified."
        hindi_explanation = "हमें ऐसे संकेत मिले हैं जो बताते हैं कि यह सामग्री भ्रामक या असत्यापित हो सकती है।"

    # Generate rich bilingual explanations via Gemini 2.5 Flash if API key configured
    if settings.gemini_api_key:
        try:
            client = genai.Client(api_key=settings.gemini_api_key)
            evidence_dict = evidence.model_dump(mode="json")
            prompt = f"""You are Satya, an AI fact-checking assistant. Write a concise, 2-sentence plain-language explanation (8th-grade reading level) for the fact-check verdict below.
Provide both an English version and a Hindi version.

Verdict: {verdict.value} (Confidence: {confidence_level.value}, Score: {confidence})
Evidence: {json.dumps(evidence_dict)}

Respond in JSON format ONLY:
{{
  "explanation_en": "Clear short explanation in English...",
  "explanation_hi": "हिंदी में स्पष्ट संक्षिप्त विवरण..."
}}"""

            response = await asyncio.to_thread(
                client.models.generate_content,
                model=settings.gemini_model,
                contents=prompt
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.replace("```json", "").replace("```", "").strip()
            res = json.loads(raw)
            if res.get("explanation_en"):
                english_explanation = res["explanation_en"]
            if res.get("explanation_hi"):
                hindi_explanation = res["explanation_hi"]
        except Exception as e:
            logger.warning(f"Gemini verdict card generation fallback: {e}")

    source_links: list[FactCheckMatch] = []
    if evidence.claim_analysis and evidence.claim_analysis.fact_checks:
        source_links = evidence.claim_analysis.fact_checks
    elif evidence.image_analysis and evidence.image_analysis.reverse_search:
        for match in evidence.image_analysis.reverse_search.matches:
            source_links.append(FactCheckMatch(
                source_name=match.title or "Visual Search",
                source_url=match.url,
                verdict="Visual Match",
                summary=match.snippet
            ))

    image_flags = []
    if evidence.image_analysis:
        if evidence.image_analysis.ai_detection and evidence.image_analysis.ai_detection.score > 0.8:
            image_flags.append("AI_GENERATED")
        if evidence.image_analysis.manipulation and evidence.image_analysis.manipulation.overall_score > 0.7:
            image_flags.append("MANIPULATED")
        if evidence.image_analysis.reverse_search and evidence.image_analysis.reverse_search.is_recycled:
            image_flags.append("RECYCLED")

    return VerdictCard(
        verdict=verdict,
        confidence=confidence,
        confidence_level=confidence_level,
        explanation_en=english_explanation,
        explanation_hi=hindi_explanation,
        sources=source_links,
        image_flags=image_flags,
        disclaimer="AI checks aren't perfect. Always verify with official sources."
    )
