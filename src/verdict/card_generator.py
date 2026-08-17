"""
src/verdict/card_generator.py — Verdict card generation using async google.genai.
Uses Gemini to write non-technical, clear explanations in English, Hindi, and Tamil.
"""
import json
import structlog
from google import genai
from google.genai import types
from src.config import settings
from src.models.schemas import VerdictCard, Verdict, FactCheckMatch
from src.verdict.confidence import CalibratedScore
from src.verdict.aggregator import AggregatedEvidence

log = structlog.get_logger(__name__)


def _get_client():
    if not settings.gemini_api_key:
        return None
    return genai.Client(api_key=settings.gemini_api_key)


def _clean_json_response(raw: str) -> dict:
    raw = raw.strip()
    start_idx = raw.find('{')
    end_idx = raw.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        raw_json = raw[start_idx:end_idx + 1]
        return json.loads(raw_json)
    return json.loads(raw)


CARD_GEN_PROMPT = """You are writing a fact-check result card for an Indian family WhatsApp group.
The audience is non-technical — think of explaining this to a grandparent.

VERDICT: {verdict}
CONFIDENCE: {confidence}%
EVIDENCE FOUND:
{signals}

FACT-CHECK SOURCES:
{sources}

Write a fact-check card in this EXACT JSON format:
{{
  "explanation_english": "2-3 simple sentences. No jargon. Start with what we found, then why it matters.",
  "explanation_hindi": "वही 2-3 वाक्य हिंदी में। सरल भाषा में। जैसे परिवार में बात करते हैं। Hinglish ठीक है।"
}}

Rules:
- English: Simple reading level. No jargon. Use plain words.
- Hindi: Natural Hinglish is fine. Avoid stiff formal Hindi. Write how people actually talk.
- For UNVERIFIABLE: say "we couldn't find any fact-check for this" not "it's false"
- For AI_GENERATED: say "this image was made by AI software, not a real photo/event"
- For MISLEADING_CONTEXT: say "the image is real but it's from [date/event], not [claimed event]"
- Maximum 60 words per language
"""


async def generate_card(
    evidence: AggregatedEvidence,
    score: CalibratedScore,
    request_id: str,
) -> VerdictCard:
    """
    Generates the final VerdictCard asynchronously using google.genai for explanations.
    """
    sources = evidence.best_sources or []
    source_urls = [s.source_url for s in sources if s.source_url]

    signals_text = "\n".join(f"- {s}" for s in evidence.signals) or "- No specific signals found"
    sources_text = "\n".join(
        f"- {s.source_name}: {s.snippet[:100]}" for s in sources[:3]
    ) or "- No fact-check sources found"

    verdict_label = {
        Verdict.LIKELY_TRUE: "LIKELY TRUE",
        Verdict.LIKELY_FALSE: "LIKELY FALSE",
        Verdict.UNVERIFIABLE: "CANNOT DETERMINE (Unverifiable)",
        Verdict.MISLEADING_CONTEXT: "MISLEADING CONTEXT (real but wrong context)",
        Verdict.AI_GENERATED: "AI GENERATED",
        Verdict.MANIPULATED: "DIGITALLY MANIPULATED",
    }.get(score.verdict, "CANNOT DETERMINE")

    prompt = CARD_GEN_PROMPT.format(
        verdict=verdict_label,
        confidence=int(score.confidence_score * 100),
        signals=signals_text,
        sources=sources_text,
    )

    try:
        client = _get_client()
        if not client:
            raise ValueError("No GEMINI_API_KEY configured")

        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3,
                max_output_tokens=400,
            ),
        )
        raw = response.text.strip()
        card_text = _clean_json_response(raw)

        explanation_en = card_text.get("explanation_english", _fallback_english(score.verdict))
        explanation_hi = card_text.get("explanation_hindi", _fallback_hindi(score.verdict))

    except Exception as e:
        log.error("card_generation_failed", error=str(e))
        explanation_en = _fallback_english(score.verdict)
        explanation_hi = _fallback_hindi(score.verdict)

    return VerdictCard(
        request_id=request_id,
        verdict=score.verdict,
        confidence_level=score.confidence_level,
        confidence_score=score.confidence_score,
        explanation_english=explanation_en,
        explanation_hindi=explanation_hi,
        sources=sources,
        source_urls=source_urls,
        signals_used=evidence.signals,
        blind_spot_warning=score.blind_spot_warning,
        is_adversarial_suspected=score.is_adversarial_suspected,
    )


def fallback_explanations(verdict: Verdict) -> tuple[str, str]:
    """Curated (english, hindi) text for a verdict — used whenever Gemini is
    unavailable. Shared with the web UI card adapter so both surfaces say the
    same thing."""
    return _fallback_english(verdict), _fallback_hindi(verdict)


def _fallback_english(verdict: Verdict) -> str:
    """Static fallback if Gemini fails."""
    return {
        Verdict.LIKELY_TRUE: "Our checks suggest this claim is likely accurate based on available fact-checks.",
        Verdict.LIKELY_FALSE: "This claim appears to be false based on fact-check sources.",
        Verdict.UNVERIFIABLE: "We could not find enough information to verify or deny this claim. This does not mean it's false.",
        Verdict.MISLEADING_CONTEXT: "The content appears real but is being shared with incorrect context or date.",
        Verdict.AI_GENERATED: "This image appears to have been generated by AI software, not from a real event.",
        Verdict.MANIPULATED: "This image shows signs of digital editing or manipulation.",
    }.get(verdict, "We could not determine the credibility of this forward.")


def _fallback_hindi(verdict: Verdict) -> str:
    """Static Hindi fallback if Gemini fails."""
    return {
        Verdict.LIKELY_TRUE: "हमारी जाँच के अनुसार यह दावा सही लगता है।",
        Verdict.LIKELY_FALSE: "यह दावा fact-check sources के अनुसार झूठा लगता है।",
        Verdict.UNVERIFIABLE: "हम इस दावे की सच्चाई नहीं जान सके। इसका मतलब यह नहीं कि यह झूठ है।",
        Verdict.MISLEADING_CONTEXT: "यह content असली है, लेकिन गलत संदर्भ में share किया जा रहा है।",
        Verdict.AI_GENERATED: "यह तस्वीर AI software से बनाई गई लगती है, किसी असली घटना की नहीं है।",
        Verdict.MANIPULATED: "इस तस्वीर में digital editing के निशान मिले हैं।",
    }.get(verdict, "हम इस forward की विश्वसनीयता नहीं जान सके।")
