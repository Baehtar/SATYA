"""
src/models/schemas.py — SHARED DATA CONTRACTS.

Every pipeline, the verdict engine, the Telegram bot, the HTTP API and the web UI
all speak through the models below. Nothing else in the codebase may redefine them.

Enum value conventions (relied on by bot/response.py, the web adapter and tests):
  * Verdict / ConfidenceLevel / LanguageCode → UPPER_SNAKE values ("LIKELY_FALSE")
  * MessageType / ClaimType                  → lower_snake values ("image_with_caption")

Contract flow:
    CheckRequest
        → ImageAnalysis | ClaimAnalysis | AudioAnalysis | ScreenshotAnalysis
            → EvidenceBundle
                → VerdictCard
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────────────────────────────────────

class MessageType(str, Enum):
    """What kind of content the user sent."""
    TEXT = "text"
    IMAGE = "image"
    IMAGE_WITH_CAPTION = "image_with_caption"
    VOICE = "voice"
    SCREENSHOT = "screenshot"
    UNKNOWN = "unknown"


class Verdict(str, Enum):
    """Final credibility call. UNVERIFIABLE is a first-class, preferred answer."""
    LIKELY_TRUE = "LIKELY_TRUE"
    LIKELY_FALSE = "LIKELY_FALSE"
    UNVERIFIABLE = "UNVERIFIABLE"
    MISLEADING_CONTEXT = "MISLEADING_CONTEXT"   # real content, wrong context/date
    AI_GENERATED = "AI_GENERATED"
    MANIPULATED = "MANIPULATED"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"           # >= 0.85
    MODERATE = "MODERATE"   # >= 0.60
    LOW = "LOW"


class LanguageCode(str, Enum):
    """Roman script is never assumed to be English."""
    EN = "EN"
    HI_DEVANAGARI = "HI_DEVANAGARI"
    HI_ROMAN = "HI_ROMAN"           # Hinglish
    TA_TAMIL = "TA_TAMIL"
    TA_ROMAN = "TA_ROMAN"           # Tanglish
    MIXED = "MIXED"


class ClaimType(str, Enum):
    """Topic bucket — decides which fact-check sources are worth querying."""
    POLITICAL = "political"
    GOVERNMENT = "government"
    HEALTH = "health"
    DISASTER = "disaster"
    COMMUNAL = "communal"
    FINANCIAL = "financial"
    SCIENCE = "science"
    CRIME = "crime"
    SPORT = "sport"
    OTHER = "other"


# ─────────────────────────────────────────────────────────────────────────────
#  Incoming request
# ─────────────────────────────────────────────────────────────────────────────

class CheckRequest(BaseModel):
    """One credibility check, whatever the surface it arrived from."""
    request_id: str
    message_type: MessageType = MessageType.UNKNOWN
    user_id: int = 0
    chat_id: int = 0

    text_content: Optional[str] = None      # message text, caption, or transcription
    image_path: Optional[str] = None
    audio_path: Optional[str] = None

    claimed_date: Optional[str] = None      # date the forward claims the event happened
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─────────────────────────────────────────────────────────────────────────────
#  Gemini structured-output schemas (passed as response_schema)
# ─────────────────────────────────────────────────────────────────────────────

class ClaimExtractionSchema(BaseModel):
    """Structured output of src/pipelines/text/claim_extractor.py."""
    claim: str = ""                 # core factual proposition, in English
    claim_hindi: str = ""
    claim_tamil: str = ""
    claim_type: str = "other"       # one of ClaimType's values
    is_checkable: bool = True       # false only for opinion / greeting / prediction
    checkability_reason: str = ""
    keywords: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)


class MatchItemSchema(BaseModel):
    """One evidence article scored against the claim."""
    index: int = 0                  # index into the FactCheckMatch list sent to the LLM
    match_confidence: float = 0.0   # 0.0–1.0; > 0.70 means same event/claim
    verdict_extracted: str = ""     # the source's own rating, e.g. "FALSE"
    verdict_explanation: str = ""


class ClaimMatchSchema(BaseModel):
    """Structured output of claim_matcher.py / nli_verifier.py."""
    matches: list[MatchItemSchema] = Field(default_factory=list)
    overall_verdict: str = "unverifiable"
    overall_confidence: float = 0.0
    reasoning: str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  Evidence — text pipeline
# ─────────────────────────────────────────────────────────────────────────────

class FactCheckMatch(BaseModel):
    """A single piece of retrieved evidence (fact-check article or news report)."""
    source_name: str = ""           # "PIB Fact Check", "Alt News", "NDTV", …
    source_url: str = ""
    original_claim: str = ""        # headline / claim as published by that source
    fact_check_verdict: str = ""    # that source's rating, verbatim
    fact_check_date: str = ""
    snippet: str = ""
    match_confidence: float = 0.0   # semantic match against the user's claim
    reason: str = ""                # why it matches (filled by the matcher)


class ClaimAnalysis(BaseModel):
    """Output of src/pipelines/text/pipeline.py."""
    raw_text: str = ""
    extracted_claim: str = ""
    claim_type: ClaimType = ClaimType.OTHER
    language: LanguageCode = LanguageCode.EN
    is_checkable: bool = True

    entities_people: list[str] = Field(default_factory=list)
    entities_places: list[str] = Field(default_factory=list)
    entities_dates: list[str] = Field(default_factory=list)
    entities_organisations: list[str] = Field(default_factory=list)

    matches: list[FactCheckMatch] = Field(default_factory=list)
    best_match: Optional[FactCheckMatch] = None

    text_verdict: Verdict = Verdict.UNVERIFIABLE
    text_verdict_confidence: float = 0.0
    no_match_reason: Optional[str] = None

    error: Optional[str] = None
    pipeline_latency_ms: int = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Evidence — image pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ReverseImageResult(BaseModel):
    """One earlier appearance of the image found online."""
    url: str = ""
    title: str = ""
    snippet: str = ""
    date_published: Optional[str] = None
    source_domain: str = ""
    match_type: str = ""                    # EXACT / FULL / PARTIAL / *_VISUAL_SIMILARITY
    date_confidence: float = 0.0            # how much to trust date_published


class ImageAnalysis(BaseModel):
    """Output of src/pipelines/image/pipeline.py."""
    ai_generation_score: float = 0.0        # 0 = camera photo, 1 = synthetic
    ai_generation_model_used: str = ""

    manipulation_score: float = 0.0         # ELA + EXIF + noise + copy-move, weighted
    ela_heatmap_path: Optional[str] = None
    exif_anomalies: list[str] = Field(default_factory=list)
    noise_inconsistency: float = 0.0
    copy_move_score: float = 0.0            # a region duplicated within the frame
    resampling_score: float = 0.0           # resize/rotate artefacts

    # Identity of the original file, computed before any analysis touches it.
    image_hash: str = ""                    # SHA-256
    phash: str = ""                         # perceptual hash

    reverse_search_results: list[ReverseImageResult] = Field(default_factory=list)
    # "Earliest LOCATED appearance" — the oldest copy our providers indexed, which
    # is not a claim about when the photograph was actually first published.
    earliest_appearance_date: Optional[str] = None
    image_status: str = ""                  # RECYCLED / CONTEMPORANEOUS / NO_MATCHES_LOCATED / …
    provenance_searched: bool = False       # False = no provider ran; NOT "nothing found"
    recycled_image: bool = False            # old image passed off as recent
    recycled_confidence: float = 0.0

    error: Optional[str] = None
    pipeline_latency_ms: int = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Evidence — audio & screenshot pipelines
# ─────────────────────────────────────────────────────────────────────────────

class AudioAnalysis(BaseModel):
    """Output of src/pipelines/audio/voice_analyzer.py."""
    transcription: str = ""
    transcription_language: str = ""
    transcription_confidence: float = 0.0
    voice_clone_score: float = 0.0          # 1 = synthetic voice
    spectral_anomalies: list[str] = Field(default_factory=list)

    error: Optional[str] = None
    pipeline_latency_ms: int = 0


class ScreenshotAnalysis(BaseModel):
    """Output of src/pipelines/screenshot/chyron_detector.py."""
    detected_channel: Optional[str] = None
    channel_confidence: float = 0.0
    extracted_chyron_text: str = ""
    chyron_verified: bool = False
    tampering_detected: bool = False
    tampering_details: str = ""

    error: Optional[str] = None
    pipeline_latency_ms: int = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregation & final output
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceBundle(BaseModel):
    """Everything the pipelines found, before any verdict is drawn."""
    request_id: str
    message_type: MessageType = MessageType.UNKNOWN

    image_analysis: Optional[ImageAnalysis] = None
    claim_analysis: Optional[ClaimAnalysis] = None
    audio_analysis: Optional[AudioAnalysis] = None
    screenshot_analysis: Optional[ScreenshotAnalysis] = None

    total_latency_ms: int = 0


class VerdictCard(BaseModel):
    """The user-facing answer, rendered by bot/formatter.py and the web UI."""
    request_id: str = ""
    verdict: Verdict = Verdict.UNVERIFIABLE
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    confidence_score: float = 0.0

    explanation_english: str = ""
    explanation_hindi: str = ""

    sources: list[FactCheckMatch] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    signals_used: list[str] = Field(default_factory=list)

    blind_spot_warning: Optional[str] = None    # honest limitation caveat
    is_adversarial_suspected: bool = False
    total_latency_ms: int = 0
