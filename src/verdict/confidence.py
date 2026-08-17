"""
src/verdict/confidence.py — Calibrated confidence scoring.
Owned by: Person 3

This is the 30% judging-weight item.
Key design principle: UNVERIFIABLE is a first-class, preferred output.
Never jump to LIKELY_FALSE without strong corroborating evidence.

Decision matrix:
  1. Strong AI generation signal                    → AI_GENERATED (high confidence)
  2. Strong manipulation + recycled image           → MANIPULATED or MISLEADING_CONTEXT
  3. Recycled image alone                           → MISLEADING_CONTEXT (moderate)
  4. Verified fact-check says FALSE                 → LIKELY_FALSE (matches fc confidence)
  5. Verified fact-check says TRUE                  → LIKELY_TRUE (matches fc confidence)
  6. Weak signals only or conflicting signals       → UNVERIFIABLE (low confidence)
  7. No signals at all                              → UNVERIFIABLE (honest)
"""
import structlog
from dataclasses import dataclass
from src.models.schemas import Verdict, ConfidenceLevel

log = structlog.get_logger(__name__)


@dataclass
class CalibratedScore:
    verdict: Verdict
    confidence_score: float       # 0.0 – 1.0
    confidence_level: ConfidenceLevel
    blind_spot_warning: str | None = None
    is_adversarial_suspected: bool = False


def calibrate_confidence(evidence) -> CalibratedScore:
    """
    Maps aggregated evidence to a calibrated verdict and confidence score.
    Uses a decision matrix with explicit uncertainty propagation.
    """
    # ── AI generation (highest priority) ─────────────────────────────────────
    if evidence.is_ai_generated:
        score = evidence.bundle.image_analysis.ai_generation_score if evidence.bundle.image_analysis else 0.8
        return CalibratedScore(
            verdict=Verdict.AI_GENERATED,
            confidence_score=score,
            confidence_level=_level(score),
            blind_spot_warning="Our AI detector may miss images from the newest generation models.",
        )

    # ── Voice clone ───────────────────────────────────────────────────────────
    if evidence.is_voice_clone:
        score = evidence.bundle.audio_analysis.voice_clone_score if evidence.bundle.audio_analysis else 0.75
        return CalibratedScore(
            verdict=Verdict.AI_GENERATED,
            confidence_score=score,
            confidence_level=_level(score),
            blind_spot_warning="Voice clone detection is not 100% reliable. Verify with original sources.",
        )

    # ── Screenshot tampering ──────────────────────────────────────────────────
    if evidence.is_screenshot_tampered:
        return CalibratedScore(
            verdict=Verdict.MANIPULATED,
            confidence_score=0.80,
            confidence_level=ConfidenceLevel.HIGH,
        )

    # ── Manipulation + recycled (strong combination) ──────────────────────────
    if evidence.is_manipulated and evidence.is_recycled:
        score = max(
            evidence.bundle.image_analysis.manipulation_score if evidence.bundle.image_analysis else 0.7,
            evidence.recycled_confidence,
        )
        return CalibratedScore(
            verdict=Verdict.MANIPULATED,
            confidence_score=_blend(score, 0.05),
            confidence_level=_level(score),
        )

    # ── Recycled image alone (misleading context) ─────────────────────────────
    if evidence.is_recycled:
        score = evidence.recycled_confidence
        # Corroborate with fact-check if available
        if evidence.has_fact_check and evidence.fact_check_verdict == Verdict.LIKELY_FALSE:
            score = _blend(score, evidence.fact_check_confidence, w1=0.6, w2=0.4)
        return CalibratedScore(
            verdict=Verdict.MISLEADING_CONTEXT,
            confidence_score=score,
            confidence_level=_level(score),
        )

    # ── Manipulation alone ────────────────────────────────────────────────────
    if evidence.is_manipulated:
        score = evidence.bundle.image_analysis.manipulation_score if evidence.bundle.image_analysis else 0.7
        return CalibratedScore(
            verdict=Verdict.MANIPULATED,
            confidence_score=score,
            confidence_level=_level(score),
            blind_spot_warning="Image editing is common and not always deceptive. Context matters.",
        )

    # ── Fact-check exists ─────────────────────────────────────────────────────
    if evidence.has_fact_check and evidence.fact_check_confidence >= 0.60:
        verdict = evidence.fact_check_verdict
        score = evidence.fact_check_confidence

        # Check for adversarial pattern: image pipeline contradicts text verdict
        if (evidence.bundle.image_analysis and
                evidence.bundle.image_analysis.ai_generation_score >= 0.4 and
                verdict == Verdict.LIKELY_TRUE):
            # Something feels off — image looks AI-generated but text says true
            return CalibratedScore(
                verdict=Verdict.UNVERIFIABLE,
                confidence_score=0.5,
                confidence_level=ConfidenceLevel.LOW,
                is_adversarial_suspected=True,
                blind_spot_warning="Conflicting signals detected. Image looks partially AI-generated "
                                  "but the text claim matches a true fact-check. Verify manually.",
            )

        return CalibratedScore(
            verdict=verdict,
            confidence_score=score,
            confidence_level=_level(score),
        )

    # ── Weak fact-check (low confidence match) ────────────────────────────────
    if evidence.has_fact_check and evidence.fact_check_confidence < 0.60:
        return CalibratedScore(
            verdict=Verdict.UNVERIFIABLE,
            confidence_score=0.55,
            confidence_level=ConfidenceLevel.LOW,
            blind_spot_warning=(
                f"Found a related fact-check but it doesn't closely match this specific claim. "
                f"Check {evidence.best_sources[0].source_name if evidence.best_sources else 'sources'} manually."
            ),
        )

    # ── No evidence ───────────────────────────────────────────────────────────
    return CalibratedScore(
        verdict=Verdict.UNVERIFIABLE,
        confidence_score=0.7,       # high confidence that we CAN'T determine this
        confidence_level=ConfidenceLevel.MODERATE,
        blind_spot_warning=(
            "No fact-checks found for this specific claim. This does NOT mean it's false — "
            "it may be too recent, too local, or not yet checked by fact-checkers."
        ),
    )


def _level(score: float) -> ConfidenceLevel:
    if score >= 0.85:
        return ConfidenceLevel.HIGH
    elif score >= 0.60:
        return ConfidenceLevel.MODERATE
    return ConfidenceLevel.LOW


def _blend(s1: float, s2: float, w1: float = 0.7, w2: float = 0.3) -> float:
    return min(1.0, (s1 * w1) + (s2 * w2))
