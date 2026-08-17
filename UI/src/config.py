"""Centralised configuration for Satya, loaded from .env file."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings. All values can be overridden via environment variables or .env file."""

    # ── API Keys ─────────────────────────────────────────────
    gemini_api_key: str = ""
    serpapi_key: str = ""

    # ── Timeout Budgets (seconds) ────────────────────────────
    total_timeout: int = 60
    image_pipeline_timeout: int = 30
    text_pipeline_timeout: int = 25
    verdict_timeout: int = 15

    # ── Confidence Thresholds ────────────────────────────────
    high_confidence_threshold: float = 0.85
    moderate_confidence_threshold: float = 0.60

    # ── Model Settings ───────────────────────────────────────
    ai_detector_model: str = "umm-maybe/AI-image-detector"
    ai_detector_fallback: str = "Organika/sdxl-detector"
    gemini_model: str = "gemini-2.5-flash"

    # ── Paths ────────────────────────────────────────────────
    upload_dir: Path = Path("uploads")
    db_url: str = "sqlite:///satya.db"

    # ── Server ───────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()

# Ensure upload directory exists
settings.upload_dir.mkdir(parents=True, exist_ok=True)
