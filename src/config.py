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

    # ── Speech-to-text ────────────────────────────────────────────────────────
    stt_engine: str = Field(default="auto", description="whisper | gemini | auto (Whisper if a key is set, else Gemini)")
    whisper_backend: str = Field(default="api", description="api (OpenAI-compatible endpoint) | local (openai-whisper package)")
    whisper_api_key: str = Field(default="", description="Whisper API key; overrides openai_api_key / groq_api_key")
    openai_api_key: str = Field(default="", description="OpenAI API key — used for Whisper when whisper_api_key is blank")
    groq_api_key: str = Field(default="", description="Groq API key — Whisper fallback, OpenAI-compatible")
    whisper_api_base: str = Field(default="", description="Override endpoint base URL; blank → derived from the key in use")
    whisper_api_model: str = Field(default="", description="Override model; blank → whisper-1 (OpenAI) / whisper-large-v3 (Groq)")
    whisper_language: str = Field(default="", description="ISO-639-1 code to force a language; blank → auto-detect")
    whisper_timeout: int = Field(default=60, description="Whisper API request timeout (seconds)")

    # ── Search ────────────────────────────────────────────────────────────────
    serpapi_key: str = Field(default="", description="SerpAPI key for Google Lens / web search")
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
    whisper_model_size: str = "medium"  # local backend only (tiny/base/small/medium/large)
    ai_image_detector_model: str = "Organika/sdxl-detector"
    use_gpu: bool = True

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./satya.db"

    # ── Dashboard ─────────────────────────────────────────────────────────────
    dashboard_port: int = 8080


# Singleton — import this everywhere
settings = Settings()  # type: ignore[call-arg]
