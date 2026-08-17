def verdict_emoji(verdict):

    if verdict == "LIKELY_TRUE":
        return "🟢"

    if verdict == "LIKELY_FALSE":
        return "🔴"

    return "🟡"


def verdict_title(verdict):

    if verdict == "LIKELY_TRUE":
        return "GENUINE / REAL IMAGE"

    if verdict == "LIKELY_FALSE":
        return "AI-GENERATED IMAGE"

    return "UNVERIFIABLE"


def confidence_bar(confidence):

    total_blocks = 10
    filled = max(0, min(10, round(confidence * total_blocks)))
    empty = total_blocks - filled

    return (
        "█" * filled +
        "░" * empty
    )


def format_mixed_verdict(image_result, text_result):

    img_verdict = image_result.get("verdict", "UNVERIFIABLE")
    img_emoji = verdict_emoji(img_verdict)
    
    return (
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🔍 <b>MULTIMEDIA ANALYSIS RESULTS</b>\n\n"
        f"<b>🖼️ Image Analysis ({img_emoji}):</b>\n"
        f"{image_result.get('explanation', 'No result')}\n\n"
        "<b>📝 Caption Analysis:</b>\n"
        f"{text_result.get('explanation', 'No result')}\n\n"
        "⚠️ <i>AI checks are not 100% accurate. Always verify important information.</i>\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )


def format_verdict(result):

    if result.get("type") == "mixed":
        return format_mixed_verdict(
            result.get("image", {}),
            result.get("text", {})
        )

    verdict = result.get("verdict", "UNVERIFIABLE")
    confidence = result.get("confidence", 0.0)
    explanation = result.get("explanation", "No explanation available.")
    sources = result.get("sources", [])

    human_score = result.get("human_score")
    artificial_score = result.get("artificial_score")

    emoji = verdict_emoji(verdict)
    title = verdict_title(verdict)

    percentage = round(confidence * 100, 1)
    bar = confidence_bar(confidence)

    scores_section = ""
    if human_score is not None and artificial_score is not None:
        h_pct = round(human_score * 100, 1)
        a_pct = round(artificial_score * 100, 1)
        scores_section = (
            "📊 <b>Detailed Classification Breakdown:</b>\n"
            f"  • 👤 <b>Real / Genuine Photo:</b> <code>{h_pct}%</code>\n"
            f"  • 🤖 <b>AI-Generated (Synthetic):</b> <code>{a_pct}%</code>\n\n"
        )

    sources_text = ""
    if sources:
        sources_list = [f"• <a href='{s.get('url', '#')}'>{s.get('name', 'Source')}</a>" for s in sources]
        sources_text = "\n\n📌 <b>Detection Model:</b>\n" + "\n".join(sources_list)

    message = (
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"{emoji} <b>VERDICT: {title}</b>\n\n"
        f"🎯 <b>Primary Confidence:</b> {bar} <b>{percentage}%</b>\n\n"
        f"{scores_section}"
        "🔍 <b>Summary:</b>\n"
        f"{explanation}"
        f"{sources_text}\n\n"
        "⚠️ <i>AI detection models are probabilistic. "
        "Always verify critical media with trusted sources.</i>\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )

    return message