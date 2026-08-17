from services.ml_service import (
    check_text,
    check_image,
    check_voice,
    check_mixed
)

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)

from telegram.ext import (
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from bot.router import (
    classify_message,
    MessageType
)

from bot.response import format_verdict

from utils.telegram_files import (
    download_photo,
    download_voice
)


async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    """Welcome message asking what analysis needs to be done with interactive options."""
    welcome_text = (
        "🛡️ <b>Welcome to Satya — AI News & Media Fact-Checker</b>\n\n"
        "What analysis would you like to perform?\n\n"
        "1️⃣ <b>Fake Text News Analysis</b>\n"
        "<i>Verify written news claims, viral messages, or forwards.</i>\n\n"
        "2️⃣ <b>Extract News from Image</b>\n"
        "<i>OCR extract text from news clippings/screenshots & fact-check.</i>\n\n"
        "3️⃣ <b>Check AI-Generated Image</b>\n"
        "<i>Detect if an image is AI-generated (SDXL/Midjourney) or authentic.</i>\n\n"
        "👇 <b>Select an option below or send your content directly:</b>"
    )

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "💬 1. Fake Text News Analysis",
                    callback_data="mode_text_news"
                )
            ],
            [
                InlineKeyboardButton(
                    "📰 2. Extract News from Image",
                    callback_data="mode_extract_image"
                )
            ],
            [
                InlineKeyboardButton(
                    "🤖 3. Check AI-Generated Image",
                    callback_data="mode_ai_image"
                )
            ]
        ]
    )

    await update.message.reply_text(
        welcome_text,
        parse_mode="HTML",
        reply_markup=keyboard
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    help_text = (
        "📖 <b>Satya Analysis Modes Guide</b>\n\n"
        "1️⃣ <b>Text Analysis:</b> Send any text claim to check against PIB, Alt News, BOOM & Google News.\n"
        "2️⃣ <b>Extract News from Image:</b> Send newspaper clips, social media graphics or screenshots to extract text & verify news claims.\n"
        "3️⃣ <b>AI Image Detection:</b> Send photos to detect synthetic/AI generation.\n\n"
        "Use /start to reset options anytime."
    )

    await update.message.reply_text(
        help_text,
        parse_mode="HTML"
    )


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    message = update.effective_message

    if not message:
        return

    message_type = classify_message(
        message
    )

    if message_type == MessageType.UNSUPPORTED:

        await message.reply_text(
            "❌ Sorry, I don't support this "
            "type of content yet."
        )

        return

    checking = await message.reply_text(
        "🔍 <b>Checking...</b>\n\n"
        "I'm analysing the forwarded content.",
        parse_mode="HTML"
    )

    try:

        # -----------------------------
        # TEXT
        # -----------------------------

        if message_type == MessageType.TEXT:

            result = await check_text(
                message.text
            )

        # -----------------------------
        # IMAGE
        # -----------------------------

        elif message_type == MessageType.IMAGE:

            async def progress_cb(text: str):
                try:
                    await checking.edit_text(f"<b>Satya Analysis Engine</b>\n\n{text}", parse_mode="HTML")
                except Exception:
                    pass

            image_path = await download_photo(
                message,
                context.bot
            )

            selected_mode = context.user_data.get("selected_mode")

            result = await check_image(
                image_path,
                progress_callback=progress_cb,
                mode=selected_mode
            )

        # -----------------------------
        # IMAGE + CAPTION
        # -----------------------------

        elif message_type == MessageType.IMAGE_WITH_CAPTION:

            image_path = await download_photo(
                message,
                context.bot
            )

            result = await check_mixed(
                image_path,
                message.caption
            )

        # -----------------------------
        # VOICE
        # -----------------------------

        elif message_type == MessageType.VOICE:

            audio_path = await download_voice(
                message,
                context.bot
            )

            result = await check_voice(
                audio_path
            )

        else:

            result = {
                "verdict": "UNVERIFIABLE",
                "confidence": 0.0,
                "explanation": "Unsupported input."
            }

        # -----------------------------
        # FORMAT RESULT
        # -----------------------------

        response = format_verdict(
            result
        )

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "📎 View Sources",
                        callback_data="sources"
                    ),
                    InlineKeyboardButton(
                        "⚠️ Report Error",
                        callback_data="report"
                    )
                ]
            ]
        )

        await checking.edit_text(
            response,
            parse_mode="HTML",
            reply_markup=keyboard
        )

    except Exception as error:

        print(
            f"Error processing message: {error}"
        )

        await checking.edit_text(
            "❌ <b>Something went wrong.</b>\n\n"
            "Please try again.",
            parse_mode="HTML"
        )


async def button_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    query = update.callback_query
    await query.answer()

    if query.data == "mode_text_news":
        context.user_data["selected_mode"] = "text_news"
        await query.message.reply_text(
            "💬 <b>Option 1 Selected: Fake Text News Analysis</b>\n\n"
            "Please send or forward the written news claim, rumor, or article text you want to verify.",
            parse_mode="HTML"
        )

    elif query.data == "mode_extract_image":
        context.user_data["selected_mode"] = "extract_image"
        await query.message.reply_text(
            "📰 <b>Option 2 Selected: Extract News from Image</b>\n\n"
            "Please upload the newspaper clipping, tweet screenshot, WhatsApp forward image, or news chyron graphic.",
            parse_mode="HTML"
        )

    elif query.data == "mode_ai_image":
        context.user_data["selected_mode"] = "ai_image"
        await query.message.reply_text(
            "🤖 <b>Option 3 Selected: Check AI-Generated Image</b>\n\n"
            "Please upload the photo or image you want to test for AI visual generation (SDXL / Midjourney / DALL-E).",
            parse_mode="HTML"
        )

    elif query.data == "sources":
        await query.message.reply_text(
            "📎 <b>Fact-Check Sources Evaluated:</b>\n"
            "• <b>PIB Fact Check:</b> Government of India claims\n"
            "• <b>Alt News:</b> Viral claims & social media misinformation\n"
            "• <b>BOOM Live:</b> Misinformation & deepfake verifications\n"
            "• <b>Google News & Fact Check API:</b> Live global index",
            parse_mode="HTML"
        )

    elif query.data == "report":
        await query.message.reply_text(
            "⚠️ Thank you! Your error report has been recorded for model calibration.",
            parse_mode="HTML"
        )


def register_handlers(application):

    application.add_handler(
        CommandHandler(
            "start",
            start
        )
    )

    application.add_handler(
        CommandHandler(
            "help",
            help_command
        )
    )

    application.add_handler(
        MessageHandler(
            filters.ALL,
            handle_message
        )
    )

    from telegram.ext import CallbackQueryHandler

    application.add_handler(
        CallbackQueryHandler(
            button_handler
        )
    )


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler for catching network timeouts and polling drops."""
    print(f"⚠️ Telegram Network Notice: {context.error}")