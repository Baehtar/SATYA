import os
from dotenv import load_dotenv
from telegram.ext import Application
from telegram.request import HTTPXRequest

from bot.handlers import register_handlers, global_error_handler


load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


async def _init_trend_dashboard(application) -> None:
    """Creates the tables the trend dashboard reads, so bot checks are recorded
    from the first message. A missing DB stack only costs trends, never replies."""
    try:
        from src.db.database import init_db
        await init_db()
        print("📊 Trend dashboard logging enabled.")
    except Exception as error:
        print(f"⚠️ Trend dashboard logging disabled: {error}")


def create_bot():

    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing from .env"
        )

    proxy_url = (
        os.getenv("TELEGRAM_PROXY_URL")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("HTTP_PROXY")
    )

    request_kwargs = {
        "connect_timeout": 30.0,
        "read_timeout": 30.0,
        "write_timeout": 30.0,
        "pool_timeout": 30.0,
    }

    if proxy_url:
        request_kwargs["proxy_url"] = proxy_url

    request = HTTPXRequest(**request_kwargs)

    application = (
        Application.builder()
        .token(TOKEN)
        .request(request)
        .post_init(_init_trend_dashboard)
        .build()
    )

    register_handlers(application)
    application.add_error_handler(global_error_handler)

    return application
