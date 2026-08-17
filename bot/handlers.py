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

    text = (
        "🛡️ <b>Welcome to Satya</b>\n\n"
        "I am an AI forward-checker.\n\n"
        "Send or forward me:\n"
        "• 📝 Suspicious text\n"
        "• 🖼️ Images\n"
        "• 🖼️ Images with captions\n"
        "• 🎙️ Voice notes\n\n"
        "I'll analyse the content and "
        "give you a credibility assessment."
    )

    await update.message.reply_text(
        text,
        parse_mode="HTML"
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = (
        "📖 <b>How to use Satya</b>\n\n"
        "Simply forward suspicious content "
        "to this bot.\n\n"
        "Supported:\n"
        "📝 Text\n"
        "🖼️ Images\n"
        "🖼️ Image + caption\n"
        "🎙️ Voice notes\n\n"
        "Satya will return a verdict with "
        "a confidence level and sources."
    )

    await update.message.reply_text(
        text,
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

            image_path = await download_photo(
                message,
                context.bot
            )

            result = await check_image(
                image_path
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

    if query.data == "sources":

        await query.message.reply_text(
            "📎 Sources will appear here "
            "once the fact-checking pipeline "
            "is connected."
        )

    elif query.data == "report":

        await query.message.reply_text(
            "⚠️ Thanks. Your report has been "
            "recorded for review."
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