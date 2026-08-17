"""
src/api/client.py — HTTP client for the Satya backend API.

This is what the Telegram bot imports. Zero pipeline logic here —
just clean async HTTP calls to the backend.

Usage in the bot:
    from src.api.client import SatyaClient

    client = SatyaClient(base_url="http://localhost:8000")

    # Text check
    result = await client.check_text("BREAKING: Government gives 15 lakh to everyone!")

    # Image check
    with open("suspicious.jpg", "rb") as f:
        result = await client.check_image(f.read(), caption="Yesterday's flood in Odisha")

    # Voice note
    with open("voice.ogg", "rb") as f:
        result = await client.check_audio(f.read())

    # Access result
    print(result.verdict)              # "likely_false"
    print(result.explanation_english)  # plain English explanation
    print(result.explanation_hindi)    # Hindi explanation
    print(result.confidence_score)     # 0.0 – 1.0
"""
import httpx
import structlog
from dataclasses import dataclass, field
from typing import Optional, BinaryIO

log = structlog.get_logger(__name__)

# Default: bot and API run on the same machine
DEFAULT_API_URL = "http://localhost:8000"


@dataclass
class VerdictResult:
    """Parsed response from the Satya API — what the bot receives."""
    request_id: str
    verdict: str                        # "likely_true" | "likely_false" | "unverifiable" | "ai_generated" | "manipulated" | "misleading_context"
    confidence_level: str               # "high" | "moderate" | "low"
    confidence_score: float             # 0.0 – 1.0
    explanation_english: str
    explanation_hindi: str
    signals_used: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    blind_spot_warning: Optional[str] = None
    is_adversarial_suspected: bool = False
    total_latency_ms: int = 0
    pipeline_breakdown: dict = field(default_factory=dict)

    # Convenience properties for the formatter
    @property
    def is_flagged(self) -> bool:
        return self.verdict in ("likely_false", "ai_generated", "manipulated", "misleading_context")

    @property
    def verdict_emoji(self) -> str:
        return {
            "likely_true": "🟢",
            "likely_false": "🔴",
            "unverifiable": "🟡",
            "ai_generated": "🤖",
            "manipulated": "✂️",
            "misleading_context": "🟠",
        }.get(self.verdict, "❓")

    @property
    def confidence_bar(self) -> str:
        filled = int(self.confidence_score * 10)
        return "█" * filled + "░" * (10 - filled)

    @classmethod
    def from_dict(cls, data: dict) -> "VerdictResult":
        return cls(
            request_id=data.get("request_id", ""),
            verdict=data.get("verdict", "unverifiable"),
            confidence_level=data.get("confidence_level", "low"),
            confidence_score=data.get("confidence_score", 0.0),
            explanation_english=data.get("explanation_english", ""),
            explanation_hindi=data.get("explanation_hindi", ""),
            signals_used=data.get("signals_used", []),
            sources=data.get("sources", []),
            source_urls=data.get("source_urls", []),
            blind_spot_warning=data.get("blind_spot_warning"),
            is_adversarial_suspected=data.get("is_adversarial_suspected", False),
            total_latency_ms=data.get("total_latency_ms", 0),
            pipeline_breakdown=data.get("pipeline_breakdown", {}),
        )


class SatyaClient:
    """
    Async HTTP client for the Satya backend.

    Usage:
        client = SatyaClient()
        result = await client.check_text("some claim")
        # result is a VerdictResult dataclass
    """

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = 65.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def health(self) -> bool:
        """Returns True if the backend is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/health")
                return r.status_code == 200
        except Exception:
            return False

    async def check_text(
        self,
        text: str,
        claimed_date: Optional[str] = None,
        user_id: int = 0,
    ) -> VerdictResult:
        """
        Fact-check a text forward.

        Args:
            text: Raw WhatsApp forward text (Hindi/English/Hinglish)
            claimed_date: Date mentioned in the forward (YYYY-MM-DD), optional
            user_id: Telegram user ID for logging, optional
        """
        log.info("client_check_text", text_len=len(text))
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/check/text",
                json={"text": text, "claimed_date": claimed_date, "user_id": user_id},
            )
            r.raise_for_status()
            return VerdictResult.from_dict(r.json())

    async def check_image(
        self,
        image_bytes: bytes,
        filename: str = "image.jpg",
        caption: Optional[str] = None,
        claimed_date: Optional[str] = None,
        user_id: int = 0,
    ) -> VerdictResult:
        """
        Fact-check an image (and optional caption).

        Args:
            image_bytes: Raw bytes of the image file
            filename: Filename with extension (used to determine MIME type)
            caption: Caption text if present
            claimed_date: Date context, optional
            user_id: Telegram user ID for logging, optional
        """
        log.info("client_check_image", bytes_len=len(image_bytes), has_caption=bool(caption))
        ext = filename.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/jpeg")

        files = {"file": (filename, image_bytes, mime)}
        data: dict = {"user_id": str(user_id)}
        if caption:
            data["caption"] = caption
        if claimed_date:
            data["claimed_date"] = claimed_date

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/check/image", files=files, data=data)
            r.raise_for_status()
            return VerdictResult.from_dict(r.json())

    async def check_audio(
        self,
        audio_bytes: bytes,
        filename: str = "voice.ogg",
        user_id: int = 0,
    ) -> VerdictResult:
        """
        Fact-check a voice note.

        Args:
            audio_bytes: Raw bytes of the audio file (OGG/MP3/WAV)
            filename: Filename with extension
            user_id: Telegram user ID for logging, optional
        """
        log.info("client_check_audio", bytes_len=len(audio_bytes))
        ext = filename.rsplit(".", 1)[-1].lower()
        mime = {"ogg": "audio/ogg", "mp3": "audio/mpeg", "wav": "audio/wav",
                "m4a": "audio/mp4"}.get(ext, "audio/ogg")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/check/audio",
                files={"file": (filename, audio_bytes, mime)},
                data={"user_id": str(user_id)},
            )
            r.raise_for_status()
            return VerdictResult.from_dict(r.json())

    async def check_screenshot(
        self,
        image_bytes: bytes,
        filename: str = "screenshot.jpg",
        user_id: int = 0,
    ) -> VerdictResult:
        """Fact-check a news channel screenshot."""
        log.info("client_check_screenshot", bytes_len=len(image_bytes))
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/check/screenshot",
                files={"file": (filename, image_bytes, "image/jpeg")},
                data={"user_id": str(user_id)},
            )
            r.raise_for_status()
            return VerdictResult.from_dict(r.json())

    async def check_auto(
        self,
        text: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        claimed_date: Optional[str] = None,
        user_id: int = 0,
    ) -> VerdictResult:
        """
        Unified endpoint — lets the backend auto-detect the type.
        Useful when the bot just passes everything and doesn't want to classify.
        """
        log.info("client_check_auto", has_text=bool(text), has_file=bool(file_bytes))
        data: dict = {"user_id": str(user_id)}
        if text:
            data["text"] = text
        if claimed_date:
            data["claimed_date"] = claimed_date

        files = {}
        if file_bytes and filename:
            ext = filename.rsplit(".", 1)[-1].lower()
            mime_map = {
                "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "webp": "image/webp", "ogg": "audio/ogg", "mp3": "audio/mpeg",
                "wav": "audio/wav", "m4a": "audio/mp4",
            }
            files["file"] = (filename, file_bytes, mime_map.get(ext, "application/octet-stream"))

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/check/",
                files=files or None,
                data=data,
            )
            r.raise_for_status()
            return VerdictResult.from_dict(r.json())
