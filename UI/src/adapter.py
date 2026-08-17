"""
UI/src/adapter.py — turns a `services.ml_service` result into the verdict-card
JSON the front-end renders.

The web UI and the Telegram bot run the *same* analysis code
(services/ml_service.py). Only the presentation differs: the bot gets a Telegram
HTML message from bot/response.py, the browser gets the JSON built here.

Verdict mapping (web slugs are lower-case; the backend's are UPPER_SNAKE):
  * A claim was checked           → the claim's verdict wins
    (likely_true | likely_false | unverifiable), because an AI-generated
    picture does not make an accompanying claim false, or vice versa.
  * No claim to check (bare image) → authenticity only:
    ai_generated when the detector is >= 0.70 confident, otherwise
    unverifiable — never "true", since nothing was actually verified.
The AI-image signal always travels separately in `image_flags`.
"""
import html
import json
import re
import structlog
from typing import Any, Dict, List, Tuple

from google import genai
from google.genai import types

from src.config import settings
from src.models.schemas import Verdict
from src.verdict.card_generator import fallback_explanations

log = structlog.get_logger(__name__)

# Same threshold the bot and src/verdict/evidence_aggregator.py use.
AI_IMAGE_THRESHOLD = 0.70

DISCLAIMER = (
    "AI checks are not 100% accurate. Always verify important information "
    "with official sources before forwarding."
)

_TAG_RE = re.compile(r"<[^>]+>")

EXPLAIN_PROMPT = """You are Satya, a fact-checking assistant for Indian WhatsApp forwards.
Rewrite the analysis below as a verdict explanation for a non-technical reader — think of explaining it to a grandparent.

VERDICT: {verdict}
WHAT WAS CHECKED: {claim}
ANALYSIS NOTES:
{evidence}

Rules:
- 2 short sentences per language, maximum 45 words each.
- Say what we found first, then what it means for the reader.
- Never claim more certainty than the notes support. For "unverifiable" say we could not find a fact-check — do NOT say it is false.
- Hindi: natural Hinglish is fine. Write how people actually speak, not stiff formal Hindi.
- No markdown, no HTML, no bullet points.

Respond with JSON only:
{{"explanation_en": "...", "explanation_hi": "..."}}"""


def _plain(text: str) -> str:
    """Backend explanations carry Telegram HTML — strip it for the browser."""
    if not text:
        return ""
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_json(raw: str) -> Dict[str, Any]:
    """Tolerates a model wrapping its JSON in prose or a ``` fence."""
    raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start:end + 1]
    return json.loads(raw)


def _confidence_level(score: float) -> str:
    """Mirrors src/verdict/evidence_aggregator.py's banding."""
    if score >= 0.80:
        return "HIGH"
    if score >= 0.60:
        return "MODERATE"
    return "LOW"


def _ai_score(result: Dict[str, Any]) -> float:
    """check_mixed() nests its image result, so look one level down too."""
    if result.get("image_ai_score") is not None:
        return float(result["image_ai_score"])
    nested = result.get("image") or {}
    return float(nested.get("image_ai_score") or 0.0)


def _sources(result: Dict[str, Any]) -> List[Dict[str, str]]:
    """ml_service emits {name, url, title, verdict}; the card wants source_*."""
    out: List[Dict[str, str]] = []
    for s in result.get("sources") or []:
        url = s.get("url") or ""
        if not url:
            continue
        out.append({
            "source_name": s.get("name") or "Source",
            "source_url": url,
            "verdict": s.get("verdict") or "",
            "snippet": s.get("title") or "",
        })
    return out


def _what_was_checked(result: Dict[str, Any], submitted_text: str) -> Tuple[str, str]:
    """Returns (label, text) describing exactly what the verdict is about."""
    kind = result.get("type")

    if kind == "voice" and result.get("transcript"):
        return "Transcript", result["transcript"]

    claim = result.get("extracted_claim") or ""
    if claim:
        if kind == "image":
            return "Claim read from the image", claim
        return "Claim checked", claim

    ocr_text = (result.get("ocr") or {}).get("cleaned_text") or ""
    if ocr_text:
        return "Text read from the image", ocr_text[:300]

    if submitted_text:
        return "Claim checked", submitted_text[:300]

    return "", ""


# The only slugs the card may carry — anything else becomes "unverifiable"
# rather than reaching the browser as an unrenderable verdict.
WEB_VERDICTS = {v.value.lower() for v in Verdict}


def _verdict_and_confidence(result: Dict[str, Any]) -> Tuple[str, float]:
    backend_verdict = str(result.get("verdict") or "UNVERIFIABLE").upper()
    if backend_verdict.lower() not in WEB_VERDICTS:
        log.warning("unknown_backend_verdict", verdict=backend_verdict)
        backend_verdict = "UNVERIFIABLE"
    confidence = float(result.get("confidence") or 0.0)
    ai_score = _ai_score(result)

    claim_was_checked = bool(result.get("extracted_claim")) or bool(result.get("sources"))

    if result.get("type") == "image" and not claim_was_checked:
        if ai_score >= AI_IMAGE_THRESHOLD:
            return "ai_generated", ai_score
        if backend_verdict == "UNVERIFIABLE":
            # The detector itself could not run (missing HF key or API error) —
            # report its own low confidence rather than implying an "authentic" call.
            return "unverifiable", confidence
        # Detector says the image looks authentic, but no claim was verified.
        return "unverifiable", max(0.5, 1.0 - ai_score)

    return backend_verdict.lower(), confidence


def _image_flags(result: Dict[str, Any]) -> List[str]:
    """Only signals this backend actually computes — no invented flags."""
    flags: List[str] = []
    if _ai_score(result) >= AI_IMAGE_THRESHOLD:
        flags.append("AI_GENERATED")
    return flags


def _fallback_verdict_enum(slug: str) -> Verdict:
    try:
        return Verdict(slug.upper())
    except ValueError:
        return Verdict.UNVERIFIABLE


async def _bilingual(verdict_slug: str, claim: str, evidence: str) -> Tuple[str, str]:
    """
    English + Hindi explanation. Uses Gemini when a key is configured, otherwise
    the curated bilingual fallback text shared with the Telegram card.
    """
    english, hindi = fallback_explanations(_fallback_verdict_enum(verdict_slug))
    if evidence:
        english = evidence

    if not settings.gemini_api_key:
        return english, hindi

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=EXPLAIN_PROMPT.format(
                verdict=verdict_slug,
                claim=claim[:500] or "(no explicit claim — media authenticity only)",
                evidence=evidence[:2000] or "(no evidence collected)",
            ),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3,
                max_output_tokens=500,
            ),
        )
        parsed = _parse_json(response.text or "")
        return (
            parsed.get("explanation_en") or english,
            parsed.get("explanation_hi") or hindi,
        )
    except Exception as e:
        log.warning("bilingual_explanation_fallback", error=str(e))
        return english, hindi


async def build_card(
    result: Dict[str, Any],
    submitted_text: str = "",
    latency_ms: int = 0,
) -> Dict[str, Any]:
    """Builds the JSON payload sent on the SSE `verdict` event."""
    verdict_slug, confidence = _verdict_and_confidence(result)
    claim_label, claim = _what_was_checked(result, submitted_text)
    evidence = _plain(result.get("explanation", ""))

    explanation_en, explanation_hi = await _bilingual(verdict_slug, claim, evidence)

    ai_score = _ai_score(result)
    disclaimer = DISCLAIMER
    if verdict_slug == "unverifiable":
        disclaimer = (
            "No fact-check was found for this claim. That does NOT mean it is false — "
            "it may be too recent, too local, or simply not checked yet. " + DISCLAIMER
        )

    card = {
        "verdict": verdict_slug,
        "confidence": round(confidence, 3),
        "confidence_level": result.get("confidence_level") or _confidence_level(confidence),
        "explanation_en": explanation_en,
        "explanation_hi": explanation_hi,
        "claim": claim,
        "claim_label": claim_label,
        "sources": _sources(result),
        "image_flags": _image_flags(result),
        "disclaimer": disclaimer,
        "meta": {
            "type": result.get("type", "text"),
            "image_ai_score": round(ai_score, 3),
            "language": result.get("language") or (result.get("ocr") or {}).get("language", ""),
            "latency_ms": latency_ms,
        },
    }

    log.info(
        "web_card_built",
        verdict=card["verdict"],
        confidence=card["confidence"],
        n_sources=len(card["sources"]),
        latency_ms=latency_ms,
    )
    return card
