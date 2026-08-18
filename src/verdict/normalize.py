"""
src/verdict/normalize.py — one definition of the display verdict slug.

`services/ml_service.py` speaks UPPER_SNAKE verdicts about *claims*. Every
front-end (the web card, the Telegram bot, the trend dashboard) needs the same
lower-case slug, and the same answer to the questions ml_service leaves open:
what does an image-only check mean, and when is a picture "AI-generated"?

Keeping that mapping here is what makes a bot check and a portal check land in
the trend dashboard as the same kind of row.
"""
from typing import Any, Dict, Tuple

import structlog

from src.models.schemas import Verdict

log = structlog.get_logger(__name__)

# Same thresholds src/verdict/aggregator.py uses: >= 0.70 is a call,
# 0.45–0.70 is too close to call, below that is clean.
AI_IMAGE_THRESHOLD = 0.70
AI_IMAGE_BORDERLINE = 0.45

MODE_FAKE_NEWS = "fake_news"
MODE_AI_IMAGE = "ai_image"

# Display-only slug: a decisive "no, this isn't AI" has no equivalent in the
# backend's Verdict enum (which only ever talks about claims).
AUTHENTIC_IMAGE = "authentic_image"

# The only slugs a front-end may carry — anything else becomes "unverifiable"
# rather than surfacing as an unrenderable verdict.
KNOWN_VERDICTS = {v.value.lower() for v in Verdict} | {AUTHENTIC_IMAGE}


def ai_score(result: Dict[str, Any]) -> float:
    """check_mixed() nests its image result, so look one level down too."""
    if result.get("image_ai_score") is not None:
        return float(result["image_ai_score"])
    nested = result.get("image") or {}
    return float(nested.get("image_ai_score") or 0.0)


def verdict_and_confidence(result: Dict[str, Any], mode: str = MODE_FAKE_NEWS) -> Tuple[str, float]:
    """Maps an ml_service result to (slug, confidence)."""
    backend_verdict = str(result.get("verdict") or "UNVERIFIABLE").upper()
    if backend_verdict.lower() not in KNOWN_VERDICTS:
        log.warning("unknown_backend_verdict", verdict=backend_verdict)
        backend_verdict = "UNVERIFIABLE"
    confidence = float(result.get("confidence") or 0.0)
    score = ai_score(result)

    # Dedicated AI-image check: answer that question and nothing else.
    if mode == MODE_AI_IMAGE:
        if backend_verdict == "UNVERIFIABLE":
            return "unverifiable", confidence      # detector unavailable
        if score >= AI_IMAGE_THRESHOLD:
            return "ai_generated", score
        if score >= AI_IMAGE_BORDERLINE:
            # Too close to call — don't hand back a clean bill of health.
            return "unverifiable", 1.0 - score
        return AUTHENTIC_IMAGE, max(confidence, 1.0 - score)

    claim_was_checked = bool(result.get("extracted_claim")) or bool(result.get("sources"))

    if result.get("type") == "image" and not claim_was_checked:
        if score >= AI_IMAGE_THRESHOLD:
            return "ai_generated", score
        if backend_verdict == "UNVERIFIABLE":
            # The detector itself could not run (missing HF key or API error) —
            # report its own low confidence rather than implying an "authentic" call.
            return "unverifiable", confidence
        # Detector says the image looks authentic, but no claim was verified.
        return "unverifiable", max(0.5, 1.0 - score)

    return backend_verdict.lower(), confidence
