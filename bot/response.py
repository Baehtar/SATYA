"""
bot/response.py — Comprehensive verdict card formatter for Telegram.
Includes:
- Big visual emoji & bilingual verdict header (supporting Text, Image, Audio, and Deepfake Video)
- Grandparent-friendly 2-line explanations in English + Hindi/Hinglish
- Reverse search & image/video provenance details
- Full technical breakdown in English (AI scores, Forensics, Keyframe analysis, Language, Latency)
- Verified fact-check source links
"""
import html
import re
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


def verdict_title_bilingual(verdict: str, media_type: str = "") -> str:
    v = (verdict or "").upper()
    is_video = media_type.lower() == "video"

    if v == "LIKELY_TRUE":
        return "LIKELY TRUE · सच होने की संभावना"
    if v == "LIKELY_FALSE":
        return "LIKELY FALSE · गलत / फ़र्ज़ी संदेश"
    if v == "AI_GENERATED":
        if is_video:
            return "AI DEEPFAKE VIDEO · एआई डीपफ़ेक वीडियो"
        return "AI GENERATED · एआई द्वारा निर्मित तस्वीर"
    if v == "MANIPULATED":
        if is_video:
            return "MANIPULATED VIDEO · वीडियो में छेड़छाड़"
        return "DIGITALLY EDITED · फोटो में छेड़छाड़"
    if v == "MISLEADING_CONTEXT":
        if is_video:
            return "MISLEADING CONTEXT · पुराना वीडियो गलत संदर्भ में"
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


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return text.strip()


def format_verdict(result: Dict[str, Any]) -> str:
    """
    Builds a complete, bilingual, grandparent-friendly card with technical
    breakdown and reverse search provenance for Telegram.
    Supports Text, Image, Audio, and Video (Deepfake).
    """
    verdict = result.get("verdict", "UNVERIFIABLE")
    confidence = float(result.get("confidence", 0.0) or 0.0)
    meta = result.get("meta") or {}
    media_type = str(result.get("type") or meta.get("type") or "text").lower()

    # 1. 2-line explanations in English and Hindi
    exp_en = _clean_text(result.get("explanation_en") or "")
    exp_hi = _clean_text(result.get("explanation_hi") or "")

    if not exp_en:
        raw_exp = result.get("explanation", "We analyzed this message against verified news and fact-check sources.")
        exp_en = _clean_text(raw_exp)

    emoji = verdict_emoji(verdict)
    title = verdict_title_bilingual(verdict, media_type=media_type)
    percentage = round(confidence * 100, 1)
    bar = confidence_bar(confidence)
    badge = _confidence_badge(confidence)

    # 2. Claim or transcript line
    claim_text = (
        result.get("claim")
        or result.get("extracted_claim")
        or result.get("transcript")
        or ""
    ).strip()

    claim_block = ""
    if claim_text:
        is_filler = any(
            p in claim_text.lower()
            for p in ["no visible text", "no text provided", "no factual claim", "there is no visible"]
        )
        if not is_filler:
            if media_type == "voice":
                label = "🎙️ <b>Spoken claim:</b>"
            elif media_type == "video":
                label = "🎬 <b>Video claim / speech:</b>"
            else:
                label = "📝 <b>Claim checked:</b>"
            claim_block = f"{label} <i>\"{claim_text[:250]}\"</i>\n\n"

    # 3. Bilingual 2-Line Simple Explanation (Grandparent Friendly)
    explanation_section = ""
    if exp_en and exp_hi:
        explanation_section = (
            "💡 <b>In Simple Words / सरल शब्दों में:</b>\n\n"
            f"🇬🇧 <b>English:</b>\n{exp_en}\n\n"
            f"🇮🇳 <b>सरल हिंदी:</b>\n{exp_hi}\n\n"
        )
    elif exp_en:
        explanation_section = (
            f"💡 <b>In Simple Words:</b>\n{exp_en}\n\n"
        )

    # 4. Reverse Search & Provenance Section
    provenance_section = ""
    prov = result.get("provenance")
    if prov and isinstance(prov, dict):
        prov_status = prov.get("status") or prov.get("image_status") or ""
        earliest_date = prov.get("earliest_located_date") or ""
        n_matches = prov.get("n_matches") or prov.get("n_matches_total") or 0
        matches = prov.get("matches") or prov.get("reverse_matches") or []

        prov_lines = ["🔍 <b>Reverse Provenance Search:</b>"]
        if prov_status:
            prov_status_clean = prov_status.replace("_", " ").title()
            prov_lines.append(f"• <b>Media Status:</b> {prov_status_clean}")
        if earliest_date:
            prov_lines.append(f"• <b>Earliest online appearance:</b> {earliest_date}")
        if n_matches > 0:
            prov_lines.append(f"• <b>Web Matches Found:</b> {n_matches} online pages indexed")

        if matches:
            seen = set()
            top_links = []
            for m in matches[:3]:
                url = m.get("url") or ""
                domain = m.get("domain") or (url.split("/")[2] if "//" in url else "web source")
                m_title = m.get("title") or m.get("page_title") or domain
                pub_date = m.get("published_date") or ""
                date_str = f" ({pub_date})" if pub_date else ""
                if url and url not in seen:
                    seen.add(url)
                    top_links.append(f"  └ <a href='{url}'>{domain}</a>{date_str}: {m_title[:60]}")

            if top_links:
                prov_lines.append("• <b>Prior web appearances:</b>\n" + "\n".join(top_links))

        provenance_section = "\n".join(prov_lines) + "\n\n"

    # 5. Technical Details Section (in English)
    ai_score = result.get("image_ai_score") or meta.get("image_ai_score") or result.get("video_deepfake_score")
    latency_ms = meta.get("latency_ms") or result.get("latency_ms") or 0
    language = meta.get("language") or result.get("language") or ""
    image_flags = result.get("image_flags") or []
    frames_analyzed = result.get("frames_analyzed") or meta.get("frames_analyzed")

    tech_lines = ["⚙️ <b>Technical Analysis Details:</b>"]
    if media_type == "video":
        tech_lines.append("• <b>Input Type:</b> Video (Multi-frame Temporal Forensics)")
        if frames_analyzed:
            tech_lines.append(f"• <b>Keyframes Analyzed:</b> {frames_analyzed} frames across timeline")
        if ai_score is not None:
            tech_lines.append(f"• <b>Visual Deepfake Probability:</b> {float(ai_score) * 100:.1f}% (Face-Swap / Diffusion Classifier)")
    elif media_type in ("image", "mixed"):
        if ai_score is not None:
            tech_lines.append(f"• <b>AI Generation Probability:</b> {float(ai_score) * 100:.1f}% (SDXL/ViT Classifier)")
        if image_flags:
            tech_lines.append(f"• <b>Forensics:</b> {', '.join(image_flags)}")
    elif media_type == "voice":
        tech_lines.append("• <b>Input Type:</b> Spoken Audio (Whisper ASR + Fact-Check)")

    if language:
        tech_lines.append(f"• <b>Detected Language:</b> {language}")
    tech_lines.append(f"• <b>Certainty Level:</b> {badge} ({percentage}%)")
    if latency_ms:
        tech_lines.append(f"• <b>Pipeline Latency:</b> {float(latency_ms)/1000:.2f}s")

    tech_section = "\n".join(tech_lines) + "\n\n"

    # 6. Fact-check sources list
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
                if len(sources_list) >= 4:
                    break

        if sources_list:
            sources_text = "📎 <b>Fact-Check Sources:</b>\n" + "\n".join(sources_list) + "\n\n"

    # 7. Disclaimer
    disclaimer = result.get("disclaimer") or (
        "⚠️ <i>Please verify important claims before forwarding to family & groups.</i>"
    )

    message = (
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"{emoji} <b>{title}</b>\n\n"
        f"<b>Certainty:</b> {bar} {percentage}% ({badge})\n\n"
        f"{claim_block}"
        f"{explanation_section}"
        f"{provenance_section}"
        f"{tech_section}"
        f"{sources_text}"
        f"{disclaimer}\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )

    return message