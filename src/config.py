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
    telegram_bot_token: str = Field(default="", description="Telegram bot token from @BotFather")

    # ── LLM ───────────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = "gemini-3.5-flash-lite"

    # ── Hugging Face ──────────────────────────────────────────────────────────
    hf_api_key: str = Field(default="", description="Hugging Face API key")
    hf_image_model: str = Field(default="Organika/sdxl-detector", description="Hugging Face model")

    # ── Search ────────────────────────────────────────────────────────────────
    serpapi_key: str = Field(default="", description="SerpAPI key for Google Lens / web search")
    google_vision_api_key: str = Field(default="", description="Google Cloud Vision API key")
    google_factcheck_api_key: str = Field(default="", description="Google Fact Check API key (free)")

    # ── App ───────────────────────────────────────────────────────────────────
    debug: bool = False
    log_level: str = "INFO"

    # ── Timeouts (seconds) ────────────────────────────────────────────────────
    image_pipeline_timeout: int = 30
    text_pipeline_timeout: int = 45
    audio_pipeline_timeout: int = 35
    total_timeout: int = 58

    # ── Confidence ────────────────────────────────────────────────────────────
    high_confidence_threshold: float = 0.85
    low_confidence_threshold: float = 0.60

    # ── Models ────────────────────────────────────────────────────────────────
    whisper_model_size: str = "medium"
    ai_image_detector_model: str = "Organika/sdxl-detector"
    use_gpu: bool = True

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./satya.db"

    # ── Dashboard ─────────────────────────────────────────────────────────────
    dashboard_port: int = 8080


# Singleton — import this everywhere
settings = Settings()  # type: ignore[call-arg]
