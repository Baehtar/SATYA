"""
src/bot/telegram_bot.py — Main Telegram bot entry point.
Owned by: Person 3

Run with: python -m src.bot.telegram_bot
"""
import asyncio
import uuid
import logging
import structlog
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters,
)
from telegram.constants import ChatAction

from src.config import settings
from src.bot.router import route_message
from src.bot.formatter import format_verdict_card

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────
#  Command Handlers
# ─────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    welcome_text = (
        f"🛡️ <b>Welcome {user.first_name} to Satya — AI News & Media Fact-Checker</b>\n\n"
        "What analysis would you like to perform?\n\n"
        "1️⃣ <b>Fake News Detection</b>\n"
        "<i>Verify written news claims, viral messages, or news text extracted from images.</i>\n\n"
        "2️⃣ <b>Fake AI Image Detection</b>\n"
        "<i>Detect if an image is AI-generated (SDXL/Midjourney/DALL-E) or authentic.</i>\n\n"
        "👇 <b>Select an option below or send your content directly:</b>"
    )

    keyboard = [
        [
            InlineKeyboardButton(
                "📰 1. Fake News Detection",
                callback_data="mode_fake_news"
            )
        ],
        [
            InlineKeyboardButton(
                "🤖 2. Fake AI Image Detection",
                callback_data="mode_ai_image"
            )
        ]
    ]

    await update.message.reply_html(
        welcome_text,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "How to use Satya:\n\n"
        "1. Forward any suspicious message to this chat\n"
        "2. I'll analyse it using AI + fact-check databases\n"
        "3. You'll get a verdict in < 60 seconds\n\n"
        "Supported formats:\n"
        "• Images (photos, screenshots)\n"
        "• Text forwards\n"
        "• Voice notes\n\n"
        "Sources checked: PIB Fact Check · AltNews · BOOM\n\n"
        "⚠️ I can be wrong. Always verify important news."
    )


# ─────────────────────────────────────────────────────────────
#  Main Message Handler
# ─────────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles all incoming forwarded messages."""
    request_id = str(uuid.uuid4())[:8]

    # Show immediate feedback
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )
    thinking_msg = await update.message.reply_text(
        "🔍 जाँच कर रहा हूँ... / Checking...\n"
        "⏳ This usually takes 10–30 seconds."
    )

    try:
        # Route message to appropriate pipeline(s)
        verdict_card = await route_message(update, request_id)

        # Format and send verdict
        text, keyboard = format_verdict_card(verdict_card)
        await thinking_msg.edit_text(
            text=text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None,
            disable_web_page_preview=True,
        )

    except asyncio.TimeoutError:
        await thinking_msg.edit_text(
            "⏱️ Time limit reached (60s). This forward was too complex to analyse quickly.\n"
            "Please try again or check manually at factcheck.pib.gov.in"
        )
    except Exception as e:
        log.error("handle_message_error", request_id=request_id, error=str(e))
        await thinking_msg.edit_text(
            "❌ Something went wrong. Please try again.\n"
            f"Error ID: {request_id}"
        )


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles inline button presses (e.g. 'Report Error')."""
    query = update.callback_query
    await query.answer()

    if query.data.startswith("report_error:"):
        request_id = query.data.split(":")[1]
        await query.message.reply_text(
            f"✅ Reported! Request ID: {request_id}\n"
            "Thank you for helping improve Satya."
        )


# ─────────────────────────────────────────────────────────────
#  Bot Bootstrap
# ─────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=settings.log_level)
    log.info("satya_bot_starting")

    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .build()
    )

    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(handle_callback_query))

    # Catch all messages (photos, text, voice, documents)
    app.add_handler(MessageHandler(
        filters.ALL & ~filters.COMMAND,
        handle_message,
    ))

    log.info("satya_bot_polling")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
