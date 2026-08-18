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
    for key in ("extracted_claim", "transcript", "claim"):
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


def _infer_claim_type(result: Dict[str, Any], message_type: str, verdict: str) -> str:
    raw_type = str(result.get("claim_type") or "").strip().lower()
    if raw_type and raw_type not in ("other", "none", "unknown", ""):
        return raw_type

    claim_text = _claim(result).lower()
    msg_type = str(message_type or "").lower()

    if "video" in msg_type or result.get("type") == "video":
        return "video"
    if verdict == "ai_generated" or (msg_type == "image" and not claim_text.strip()):
        return "visual_media"

    # Topic classification heuristics
    if any(k in claim_text for k in ["minister", "assembly", "election", "politician", "party", "bjp", "congress", "dmk", "aap", "parliament", "modi", "rahul", "resolution", "delimitation", "vote", "chief minister"]):
        return "political"
    if any(k in claim_text for k in ["scheme", "yojana", "pension", "subsidy", "government", "govt", "pib", "aadhaar", "pan card", "traffic fine", "fines", "rto", "license", "official"]):
        return "government"
    if any(k in claim_text for k in ["fire", "flood", "accident", "killed", "deaths", "cyclone", "earthquake", "rain", "blast", "collapsed", "lodge", "hotel"]):
        return "disaster"
    if any(k in claim_text for k in ["bank", "rbi", "rupees", "rs", "lakh", "crore", "upi", "scam", "lottery", "investment", "crypto", "account"]):
        return "financial"
    if any(k in claim_text for k in ["hospital", "doctor", "virus", "covid", "disease", "medicine", "vaccine", "cancer", "cure", "health"]):
        return "health"
    if any(k in claim_text for k in ["police", "arrest", "custody", "kidnap", "murder", "thief", "theft", "gang", "crime", "detained"]):
        return "crime"
    if any(k in claim_text for k in ["cricket", "ipl", "world cup", "match", "tournament", "dhoni", "kohli", "rohit", "sport"]):
        return "sport"

    if msg_type == "image":
        return "visual_media"
    if msg_type == "voice":
        return "voice_note"

    return "other"


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
        claim_type = _infer_claim_type(result, message_type, verdict)
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
            "claim_type": claim_type,
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
