import os
import time
import uuid

from services.ml_service import (
    check_text,
    check_image,
    check_voice,
    check_mixed,
    check_video
)

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)

from telegram.ext import (
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

from bot.router import (
    classify_message,
    MessageType
)

from bot.response import format_verdict
from UI.src.adapter import build_card

from utils.telegram_files import (
    download_photo,
    download_voice,
    download_video
)

# Bot checks feed the same trend dashboard the web portal does (/dashboard).
try:
    from src.db.trend_log import log_result
    TREND_LOGGING = True
except ImportError:      # DB stack not installed — the bot still answers.
    TREND_LOGGING = False


# The dashboard groups checks by input type; these are the labels it shows.
TREND_MESSAGE_TYPES = {
    MessageType.TEXT: "text",
    MessageType.IMAGE: "image",
    MessageType.IMAGE_WITH_CAPTION: "image_caption",
    MessageType.VOICE: "voice",
    MessageType.VIDEO: "video",
    MessageType.VIDEO_WITH_CAPTION: "video",
}


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler for catching network timeouts and polling drops."""
    print(f"⚠️ Telegram Network Notice: {context.error}")


async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    """Welcome message asking what analysis needs to be done with interactive options."""
    welcome_text = (
        "🛡️ <b>Welcome to Satya — AI News & Media Fact-Checker</b>\n\n"
        "What analysis would you like to perform?\n\n"
        "1️⃣ <b>Fake News Detection</b>\n"
        "<i>Verify written news claims, viral messages, or news text extracted from images.</i>\n\n"
        "2️⃣ <b>Fake AI Image Detection</b>\n"
        "<i>Detect if an image is AI-generated (SDXL/Midjourney/DALL-E), inspect reverse search provenance & EXIF forensics.</i>\n\n"
        "3️⃣ <b>Verify Audio / Voice Note</b>\n"
        "<i>Transcribe audio voice notes (Whisper AI) & fact-check extracted spoken news claims.</i>\n\n"
        "4️⃣ <b>Deepfake Video Detection</b>\n"
        "<i>Detect face-swap/diffusion synthetic manipulation across video frames, voice clones, and video provenance.</i>\n\n"
        "👇 <b>Select an option below or send your content directly:</b>"
    )

    keyboard = InlineKeyboardMarkup(
        [
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
            ],
            [
                InlineKeyboardButton(
                    "🎤 3. Verify Audio / Voice Note",
                    callback_data="mode_audio"
                )
            ],
            [
                InlineKeyboardButton(
                    "🎬 4. Deepfake Video Detection",
                    callback_data="mode_video"
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
        "1️⃣ <b>Fake News Detection:</b> Send written news claims, viral text, or news clipping/screenshot images to extract text & verify news claims against PIB, Alt News, BOOM & Google News.\n"
        "2️⃣ <b>Fake AI Image Detection:</b> Send photos or images to test for synthetic/AI visual generation, reverse image search provenance, earliest online appearance dates, and EXIF/ELA forensics.\n"
        "3️⃣ <b>Verify Audio / Voice Note:</b> Send audio files or Telegram voice notes to transcribe spoken speech (Whisper ASR) & verify news claims.\n"
        "4️⃣ <b>Deepfake Video Detection:</b> Send video clips (MP4/MOV) to extract keyframes, inspect facial deepfakes, check voice cloning, and verify spoken claims.\n\n"
        "Use /start to reset options anytime."
    )

    await update.message.reply_text(
        help_text,
        parse_mode="HTML"
    )


async def _log_to_dashboard(
    result: dict,
    request_id: str,
    message_type,
    latency_ms: int,
    user_id: int,
    mode: str = None
):
    """Records the check on the trend dashboard (/dashboard on the web portal)."""
    if not TREND_LOGGING:
        return

    await log_result(
        result,
        request_id=request_id,
        message_type=TREND_MESSAGE_TYPES.get(message_type, "text"),
        latency_ms=latency_ms,
        user_id=user_id,
        mode="ai_image" if mode == "ai_image" else ("video" if mode == "video" else "fake_news")
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
            "❌ Sorry, I don't support this type of content yet."
        )
        return

    checking = await message.reply_text(
        "🔍 <b>Checking...</b>\n\n"
        "I'm analysing the forwarded content.",
        parse_mode="HTML"
    )

    request_id = str(uuid.uuid4())
    started_at = time.monotonic()

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
            async def progress_cb(text: str, step: str = ""):
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
        # VOICE / AUDIO
        # -----------------------------
        elif message_type == MessageType.VOICE:
            async def progress_cb(text: str, step: str = ""):
                try:
                    await checking.edit_text(f"<b>Satya Analysis Engine</b>\n\n{text}", parse_mode="HTML")
                except Exception:
                    pass

            audio_path = None
            try:
                if progress_cb:
                    await progress_cb("🔍 Checking audio...")

                audio_path = await download_voice(
                    message,
                    context.bot
                )

                result = await check_voice(
                    audio_path,
                    progress_callback=progress_cb
                )
            finally:
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.unlink(audio_path)
                    except Exception:
                        pass

        # -----------------------------
        # VIDEO (DEEPFAKE & AUTHENTICITY ANALYSIS)
        # -----------------------------
        elif message_type in (MessageType.VIDEO, MessageType.VIDEO_WITH_CAPTION):
            async def progress_cb(text: str, step: str = ""):
                try:
                    await checking.edit_text(f"<b>Satya Video Engine</b>\n\n{text}", parse_mode="HTML")
                except Exception:
                    pass

            video_path = None
            try:
                video_path = await download_video(
                    message,
                    context.bot
                )
                caption = getattr(message, "caption", "") or ""
                result = await check_video(
                    video_path,
                    caption=caption,
                    progress_callback=progress_cb,
                    mode=context.user_data.get("selected_mode")
                )
            finally:
                if video_path and os.path.exists(video_path):
                    try:
                        os.unlink(video_path)
                    except Exception:
                        pass

        else:
            result = {
                "verdict": "UNVERIFIABLE",
                "confidence": 0.0,
                "explanation": "Unsupported input."
            }

        # -----------------------------
        # LOG TO THE TREND DASHBOARD
        # -----------------------------
        await _log_to_dashboard(
            result,
            request_id=request_id,
            message_type=message_type,
            latency_ms=int((time.monotonic() - started_at) * 1000),
            user_id=update.effective_user.id if update.effective_user else 0,
            mode=context.user_data.get("selected_mode")
        )

        # -----------------------------
        # BUILD GRANDPARENT-FRIENDLY BILINGUAL CARD
        # -----------------------------
        latency_ms = int((time.monotonic() - started_at) * 1000)
        try:
            sub_text = message.text or getattr(message, "caption", "") or result.get("extracted_claim", "") or result.get("transcript", "")
            card = await build_card(
                result,
                submitted_text=sub_text,
                latency_ms=latency_ms,
                mode=context.user_data.get("selected_mode", "fake_news"),
            )
        except Exception as card_err:
            print(f"Bilingual card generation fallback: {card_err}")
            card = result
            card["latency_ms"] = latency_ms

        response = format_verdict(
            card
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

    if query.data in ("mode_fake_news", "mode_text_news", "mode_extract_image"):
        context.user_data["selected_mode"] = "fake_news"
        await query.message.reply_text(
            "📰 <b>Option 1 Selected: Fake News Detection</b>\n\n"
            "Please send or forward the news claim text, viral message, or newspaper clipping/screenshot image you want to verify.",
            parse_mode="HTML"
        )

    elif query.data == "mode_ai_image":
        context.user_data["selected_mode"] = "ai_image"
        await query.message.reply_text(
            "🤖 <b>Option 2 Selected: Fake AI Image Detection</b>\n\n"
            "Please upload the photo or image you want to test for AI visual generation (SDXL / Midjourney / DALL-E).",
            parse_mode="HTML"
        )

    elif query.data in ("mode_audio", "mode_voice"):
        context.user_data["selected_mode"] = "audio"
        await query.message.reply_text(
            "🎤 <b>Option 3 Selected: Verify Audio / Voice Note</b>\n\n"
            "Please send or forward the Telegram voice note or audio recording (MP3, WAV, OGG, M4A) you want to transcribe & verify.",
            parse_mode="HTML"
        )

    elif query.data == "mode_video":
        context.user_data["selected_mode"] = "video"
        await query.message.reply_text(
            "🎬 <b>Option 4 Selected: Deepfake Video Detection</b>\n\n"
            "Please send or forward the video file (MP4, MOV, WebM, MKV) to inspect for AI face-swaps, synthetic speech, and fact-check claims.",
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

    application.add_handler(
        CallbackQueryHandler(
            button_handler
        )
    )
