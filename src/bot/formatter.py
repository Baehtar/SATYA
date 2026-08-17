"""
src/bot/formatter.py — Formats VerdictCard into a rich Telegram HTML message.
Owned by: Person 3
"""
from telegram import InlineKeyboardButton
from src.models.schemas import VerdictCard, Verdict, ConfidenceLevel


VERDICT_EMOJI = {
    Verdict.LIKELY_TRUE: "🟢",
    Verdict.LIKELY_FALSE: "🔴",
    Verdict.UNVERIFIABLE: "🟡",
    Verdict.MISLEADING_CONTEXT: "🟠",
    Verdict.AI_GENERATED: "🤖",
    Verdict.MANIPULATED: "✂️",
}

VERDICT_LABEL = {
    Verdict.LIKELY_TRUE: "LIKELY TRUE / संभवतः सच",
    Verdict.LIKELY_FALSE: "LIKELY FALSE / संभवतः झूठ",
    Verdict.UNVERIFIABLE: "UNVERIFIABLE / अज्ञात",
    Verdict.MISLEADING_CONTEXT: "MISLEADING CONTEXT / भ्रामक संदर्भ",
    Verdict.AI_GENERATED: "AI GENERATED / AI द्वारा बनाया",
    Verdict.MANIPULATED: "MANIPULATED / छेड़छाड़ की गई",
}

CONFIDENCE_BAR = {
    ConfidenceLevel.HIGH: "██████████ HIGH",
    ConfidenceLevel.MODERATE: "███████░░░ MODERATE",
    ConfidenceLevel.LOW: "████░░░░░░ LOW",
}


def format_verdict_card(card: VerdictCard) -> tuple[str, list]:
    """Returns (html_text, keyboard_buttons)."""
    emoji = VERDICT_EMOJI.get(card.verdict, "❓")
    label = VERDICT_LABEL.get(card.verdict, card.verdict.value)
    bar = CONFIDENCE_BAR.get(card.confidence_level, "")

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"{emoji} <b>{label}</b>",
        f"<code>{bar}</code>  {int(card.confidence_score * 100)}%",
        "",
    ]

    # Signals used
    if card.signals_used:
        lines.append("🔬 <b>Signals detected:</b>")
        for sig in card.signals_used:
            lines.append(f"  • {sig}")
        lines.append("")

    # English explanation
    lines.append(f"🔍 <b>What we found:</b>")
    lines.append(card.explanation_english)
    lines.append("")

    # Hindi explanation
    lines.append(f"🔍 <b>हमने क्या पाया:</b>")
    lines.append(card.explanation_hindi)
    lines.append("")

    # Sources
    if card.sources:
        lines.append("📎 <b>Sources:</b>")
        for i, src in enumerate(card.sources[:3], 1):
            lines.append(f"  [{i}] <a href='{src.source_url}'>{src.source_name}</a> — {src.snippet[:80]}...")
        lines.append("")

    # Blind spot warning
    if card.blind_spot_warning:
        lines.append(f"⚠️ <i>{card.blind_spot_warning}</i>")
        lines.append("")

    # Latency
    lines.append(f"<i>⏱ Checked in {card.total_latency_ms // 1000}s  •  ID: {card.request_id}</i>")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    text = "\n".join(lines)

    # Inline keyboard
    keyboard = []
    if card.source_urls:
        keyboard.append([
            InlineKeyboardButton("📰 View Sources", url=card.source_urls[0])
        ])
    keyboard.append([
        InlineKeyboardButton("⚠️ Report Error", callback_data=f"report_error:{card.request_id}"),
        InlineKeyboardButton("🔁 Check Again", callback_data=f"recheck:{card.request_id}"),
    ])

    return text, keyboard
