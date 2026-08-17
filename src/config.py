"""
src/config.py — Centralised settings (auto-loaded from .env)
All other modules import from here. Never hard-code keys or thresholds elsewhere.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Telegram ──────────────────────────────────────────────────────────────
    telegram_bot_token: str = Field(..., description="Telegram bot token from @BotFather")

    # ── LLM ───────────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    gemini_model: str = "gemini-2.0-flash"

    # ── Search ────────────────────────────────────────────────────────────────
    serpapi_key: str = Field(..., description="SerpAPI key for Google Lens")
    google_factcheck_api_key: str = Field(default="", description="Google Fact Check API key (free)")

    # ── App ───────────────────────────────────────────────────────────────────
    debug: bool = False
    log_level: str = "INFO"

    # ── Timeouts (seconds) ────────────────────────────────────────────────────
    image_pipeline_timeout: int = 30
    text_pipeline_timeout: int = 25
    audio_pipeline_timeout: int = 35
    total_timeout: int = 58

    # ── Confidence ────────────────────────────────────────────────────────────
    high_confidence_threshold: float = 0.85
    low_confidence_threshold: float = 0.60

    # ── Models ────────────────────────────────────────────────────────────────
    whisper_model_size: str = "medium"
    ai_image_detector_model: str = "umm-maybe/AI-image-detector"
    use_gpu: bool = True

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./satya.db"

    # ── Dashboard ─────────────────────────────────────────────────────────────
    dashboard_port: int = 8080


# Singleton — import this everywhere
settings = Settings()  # type: ignore[call-arg]
