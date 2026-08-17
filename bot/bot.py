import os

from dotenv import load_dotenv
from telegram.ext import Application

from bot.handlers import register_handlers


load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


def create_bot():

    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing from .env"
        )

    application = (
        Application.builder()
        .token(TOKEN)
        .build()
    )

    register_handlers(application)

    return application