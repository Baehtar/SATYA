"""
bot/response.py — Grandparent-friendly, bilingual verdict card formatter for Telegram.
Renders clean, easy-to-read verdict cards in English + Hindi/Hinglish.
"""
from typing import Any, Dict, List


def verdict_emoji(verdict: str) -> str:
    v = (verdict or "").upper()
    if v in ("LIKELY_TRUE", "AUTHENTIC_IMAGE"):
        return "🟢"
    if v in ("LIKELY_FALSE", "AI_GENERATED", "MANIPULATED"):
        return "🔴" if v == "LIKELY_FALSE" else ("🤖" if v == "AI_GENERATED" else "✂️")
    if v == "MISLEADING_CONTEXT":
        return "🔁"
    return "🟡"


def verdict_title_bilingual(verdict: str) -> str:
    v = (verdict or "").upper()
    if v == "LIKELY_TRUE":
        return "LIKELY TRUE · सच होने की संभावना"
    if v == "LIKELY_FALSE":
        return "LIKELY FALSE · गलत / फ़र्ज़ी संदेश"
    if v == "AI_GENERATED":
        return "AI GENERATED · एआई द्वारा निर्मित तस्वीर"
    if v == "MANIPULATED":
        return "DIGITALLY EDITED · फोटो में छेड़छाड़"
    if v == "MISLEADING_CONTEXT":
        return "MISLEADING CONTEXT · पुरानी फोटो गलत संदर्भ में"
    if v == "AUTHENTIC_IMAGE":
        return "AUTHENTIC PHOTO · वास्तविक तस्वीर"
    return "UNVERIFIABLE · पुष्टि नहीं हो सकी"


def confidence_bar(confidence: float) -> str:
    total_blocks = 10
    filled = max(0, min(10, round(confidence * total_blocks)))
    empty = total_blocks - filled
    return "█" * filled + "░" * empty


def _confidence_badge(confidence: float) -> str:
    if confidence >= 0.80:
        return "High"
    if confidence >= 0.60:
        return "Moderate"
    return "Low"


def format_verdict(result: Dict[str, Any]) -> str:
    """
    Builds a clean, 2-line bilingual explanation card for Telegram users.
    Handles both raw ml_service results and adapter card payloads.
    """
    verdict = result.get("verdict", "UNVERIFIABLE")
    confidence = float(result.get("confidence", 0.0) or 0.0)

    # 2-line explanations in English and Hindi
    exp_en = result.get("explanation_en") or ""
    exp_hi = result.get("explanation_hi") or ""

    # Fallback to single explanation if bilingual keys not populated
    if not exp_en:
        raw_exp = result.get("explanation", "We analyzed this message against verified news and fact-check sources.")
        # Strip internal tags if present
        import re
        clean_exp = re.sub(r"<[^>]+>", "", raw_exp).strip()
        exp_en = clean_exp

    emoji = verdict_emoji(verdict)
    title = verdict_title_bilingual(verdict)
    percentage = round(confidence * 100, 1)
    bar = confidence_bar(confidence)
    badge = _confidence_badge(confidence)

    # Claim or transcript line
    claim_text = (
        result.get("claim")
        or result.get("extracted_claim")
        or result.get("transcript")
        or ""
    ).strip()

    claim_block = ""
    if claim_text:
        # Avoid filler text like "no visible text"
        is_filler = any(
            p in claim_text.lower()
            for p in ["no visible text", "no text provided", "no factual claim", "there is no visible"]
        )
        if not is_filler:
            label = "🎙️ <b>Spoken claim:</b>" if result.get("type") == "voice" else "📝 <b>Claim checked:</b>"
            claim_block = f"{label} <i>\"{claim_text[:250]}\"</i>\n\n"

    # Bilingual 2-line explanation section
    explanation_section = ""
    if exp_en and exp_hi:
        explanation_section = (
            "🔍 <b>What this means / इसका क्या मतलब है:</b>\n\n"
            f"🇬🇧 <b>English:</b>\n{exp_en}\n\n"
            f"🇮🇳 <b>सरल हिंदी:</b>\n{exp_hi}\n"
        )
    elif exp_en:
        explanation_section = (
            f"🔍 <b>What we found:</b>\n{exp_en}\n"
        )

    # Sources list
    sources = result.get("sources", [])
    sources_text = ""
    if sources:
        seen_urls = set()
        sources_list: List[str] = []
        for s in sources:
            url = s.get("url") or s.get("source_url") or ""
            name = s.get("name") or s.get("source_name") or "Verified Source"
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources_list.append(f"• <a href='{url}'>{name}</a>")
                if len(sources_list) >= 3:
                    break

        if sources_list:
            sources_text = "\n📎 <b>Fact-check sources:</b>\n" + "\n".join(sources_list) + "\n"

    # Disclaimer
    disclaimer = result.get("disclaimer") or (
        "⚠️ <i>Please verify important claims before forwarding to family & groups.</i>"
    )

    message = (
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"{emoji} <b>{title}</b>\n\n"
        f"<b>Certainty:</b> {bar} {percentage}% ({badge})\n\n"
        f"{claim_block}"
        f"{explanation_section}"
        f"{sources_text}\n"
        f"{disclaimer}\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )

    return message