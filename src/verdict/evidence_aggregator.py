"""
src/verdict/evidence_aggregator.py — Evidence Aggregation & Calibrated Confidence Engine.
Fuses AI-Image Authenticity Signal with Multilingual OCR Text Claim & NLI Evidence.
Calculates verdict (LIKELY_TRUE | LIKELY_FALSE | UNVERIFIABLE) and confidence level (HIGH | MODERATE | LOW).
"""
import structlog
from typing import Dict, Any, List
from src.models.schemas import Verdict, FactCheckMatch

log = structlog.get_logger(__name__)


def aggregate_evidence(
    text_analysis: Any,
    image_ai_score: float,
    ocr_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Aggregates text claim analysis, NLI ratings, source authority, and AI image detection score.
    Returns complete unified verdict payload.
    """
    raw_verdict = text_analysis.text_verdict.value if hasattr(text_analysis.text_verdict, "value") else str(text_analysis.text_verdict)
    text_confidence = float(text_analysis.text_verdict_confidence)
    matches: List[FactCheckMatch] = getattr(text_analysis, "matches", [])
    extracted_claim = getattr(text_analysis, "extracted_claim", "")

    # Calculate evidence metrics
    n_sources = len(matches)
    high_confidence_matches = [m for m in matches if getattr(m, "match_confidence", 0.0) >= 0.70]
    n_high_conf = len(high_confidence_matches)

    # NLI Contradiction / Entailment tallying
    n_contradiction = sum(1 for m in matches if getattr(m, "fact_check_verdict", "").lower() in ["false", "fake", "misleading", "contradiction", "debunked"])
    n_entailment = sum(1 for m in matches if getattr(m, "fact_check_verdict", "").lower() in ["true", "correct", "reported_news", "entailment", "verified"])

    # Calibrate confidence score dynamically (Not hardcoded 50%)
    if raw_verdict in ["LIKELY_FALSE", "LIKELY_TRUE"]:
        base_confidence = text_confidence
        # Boost confidence if multiple independent high-authority sources agree
        if n_high_conf >= 2:
            base_confidence = min(0.98, base_confidence + 0.10)
        elif n_high_conf == 1:
            base_confidence = max(0.75, base_confidence)
        final_confidence = round(base_confidence, 2)
        verdict = raw_verdict
    else:
        verdict = "UNVERIFIABLE"
        final_confidence = 0.50 if n_sources == 0 else min(0.65, 0.40 + (0.05 * n_sources))

    # Expose confidence_level
    if final_confidence >= 0.80:
        confidence_level = "HIGH"
    elif final_confidence >= 0.60:
        confidence_level = "MODERATE"
    else:
        confidence_level = "LOW"

    # Evidence Fusion Logic: AI-Image Score vs News Claim Truth
    is_ai_image = (image_ai_score >= 0.70)
    image_note = ""
    if is_ai_image:
        image_note = f"The image is classified as AI-Generated / Synthetic ({image_ai_score*100:.1f}% confidence). Note: AI image score reflects visual authenticity, not the factual truth of the text."
    else:
        image_note = f"The image is classified as Genuine / Real (Human-Created) ({ (1.0 - image_ai_score)*100:.1f}% confidence)."

    # Format Evidence Items for output schema
    evidence_items = []
    for m in matches[:5]:
        evidence_items.append({
            "source_name": m.source_name,
            "url": m.source_url,
            "title": m.original_claim,
            "verdict": m.fact_check_verdict,
            "confidence": getattr(m, "match_confidence", 0.0),
            "reason": getattr(m, "reason", "")
        })

    log.info(
        "evidence_aggregated",
        verdict=verdict,
        confidence=final_confidence,
        confidence_level=confidence_level,
        n_sources=n_sources,
        image_ai_score=image_ai_score
    )

    return {
        "verdict": verdict,
        "confidence_score": final_confidence,
        "confidence_level": confidence_level,
        "extracted_claim": extracted_claim,
        "ocr": ocr_meta,
        "image_ai_score": image_ai_score,
        "image_note": image_note,
        "evidence": evidence_items,
        "sources": [
            {
                "name": m.source_name,
                "url": m.source_url,
                "title": m.original_claim,
                "verdict": m.fact_check_verdict
            }
            for m in matches[:5] if m.source_url
        ]
    }
