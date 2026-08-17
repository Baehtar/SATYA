import os
from dotenv import load_dotenv
from telegram.ext import Application
from telegram.request import HTTPXRequest

from bot.handlers import register_handlers, global_error_handler


load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


def create_bot():

    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing from .env"
        )

    request = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )

    application = (
        Application.builder()
        .token(TOKEN)
        .request(request)
        .build()
    )

    register_handlers(application)
    application.add_error_handler(global_error_handler)

    return application