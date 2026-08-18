"""
src/db/trend_log.py — one way in to the trend dashboard.

Every front-end that finishes a check writes a `ForwardCheck` row through here:
the Telegram bot (bot/handlers.py) and the web portal (UI/src/server.py). Both
go through the same normalisation, so /dashboard counts a bot check and a portal
check the same way.

Logging must never break a check the user already has an answer for — every
failure here is swallowed and logged.
"""
from typing import Any, Dict

import structlog

from src.db.database import log_check
from src.verdict.normalize import MODE_FAKE_NEWS, ai_score, verdict_and_confidence

log = structlog.get_logger(__name__)


def _claim(result: Dict[str, Any]) -> str:
    """The text the pipeline actually checked, whatever it was extracted from."""
    for key in ("extracted_claim", "transcript"):
        value = (result.get(key) or "").strip()
        if value:
            return value
    ocr_text = ((result.get("ocr") or {}).get("cleaned_text") or "").strip()
    return ocr_text


def _confidence_level(result: Dict[str, Any], confidence: float) -> str:
    level = result.get("confidence_level")
    if level:
        return str(level)
    if confidence >= 0.85:
        return "HIGH"
    if confidence >= 0.60:
        return "MODERATE"
    return "LOW"


async def log_result(
    result: Dict[str, Any],
    *,
    request_id: str,
    message_type: str,
    latency_ms: int = 0,
    user_id: int = 0,
    mode: str = MODE_FAKE_NEWS,
) -> None:
    """Records a finished `services.ml_service` result on the trend dashboard."""
    try:
        verdict, confidence = verdict_and_confidence(result, mode or MODE_FAKE_NEWS)
        await log_check({
            "request_id": request_id,
            "user_id": user_id,
            "message_type": message_type,
            "verdict": verdict,
            "confidence_score": round(float(confidence), 3),
            "confidence_level": _confidence_level(result, confidence),
            "ai_generation_score": round(ai_score(result), 3),
            "recycled_image": _is_recycled(result),
            "has_fact_check": bool(result.get("sources")),
            "claim_type": str(result.get("claim_type") or "other"),
            "extracted_claim": _claim(result)[:500],
            "latency_ms": latency_ms,
        })
    except Exception as e:
        log.warning("trend_log_failed", request_id=request_id, error=str(e))


def _is_recycled(result: Dict[str, Any]) -> bool:
    """True when the reverse-image engine placed the photo online before the claim
    (services/image/reverse_engine.py STATUS_RECYCLED)."""
    provenance = result.get("provenance") or (result.get("image") or {}).get("provenance") or {}
    return str(provenance.get("image_status") or "").upper() == "RECYCLED"
