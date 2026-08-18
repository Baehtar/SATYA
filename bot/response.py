def verdict_emoji(verdict):

    if verdict == "LIKELY_TRUE":
        return "🟢"

    if verdict == "LIKELY_FALSE":
        return "🔴"

    return "🟡"


def verdict_title(verdict):

    if verdict == "LIKELY_TRUE":
        return "LIKELY TRUE"

    if verdict == "LIKELY_FALSE":
        return "LIKELY FALSE"

    return "UNVERIFIABLE"


def confidence_bar(confidence):

    total_blocks = 10
    filled = max(0, min(10, round(confidence * total_blocks)))
    empty = total_blocks - filled

    return (
        "█" * filled +
        "░" * empty
    )


def format_verdict(result):
    """
    Formats structured Telegram response cards per fake_news_workflow.md Section 18.
    Supports voice/audio transcripts & extracted claims.
    """
    verdict = result.get("verdict", "UNVERIFIABLE")
    confidence = result.get("confidence", 0.0)
    explanation = result.get("explanation", "No explanation available.")
    sources = result.get("sources", [])
    lang = result.get("language", "EN")
    transcript = result.get("transcript", "").strip()
    extracted_claim = result.get("extracted_claim", "").strip()

    emoji = verdict_emoji(verdict)
    title = verdict_title(verdict)

    percentage = round(confidence * 100, 1)
    bar = confidence_bar(confidence)

    # Multi-language heading for "What we found"
    if lang in ["HI_DEVANAGARI", "HI_ROMAN"]:
        section_heading = "🔍 <b>What we found / हमने क्या पाया</b>"
    elif lang in ["TA_TAMIL", "TA_ROMAN"]:
        section_heading = "🔍 <b>What we found / நாங்கள் கண்டறிந்தது</b>"
    else:
        section_heading = "🔍 <b>What we found</b>"

    # Transcript block
    transcript_block = ""
    if transcript:
        transcript_block = f"🎙️ <b>Transcript</b>\n<i>\"{transcript}\"</i>\n\n"

    # Extracted claim block
    claim_block = ""
    if extracted_claim and extracted_claim != transcript:
        is_filler = any(p in extracted_claim.lower() for p in ["no visible text", "no text provided", "no factual claim", "there is no visible"])
        if not is_filler:
            claim_block = f"📝 <b>Claim checked</b>\n<i>\"{extracted_claim}\"</i>\n\n"

    # Sources list formatting
    sources_text = ""
    if sources:
        sources_list = [
            f"• <a href='{s.get('url', '#')}'>{s.get('name', 'Source')}</a>"
            for s in sources if s.get('url')
        ]
        if sources_list:
            sources_text = "\n\n📎 <b>Sources</b>\n" + "\n".join(sources_list)

    message = (
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"{emoji} <b>{title}</b>\n\n"
        f"Confidence: {bar} {percentage}%\n\n"
        f"{transcript_block}"
        f"{claim_block}"
        f"{section_heading}\n"
        f"{explanation}"
        f"{sources_text}\n\n"
        "⚠️ <i>AI checks are not 100% accurate. "
        "Always verify important information with official sources.</i>\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )

    return message