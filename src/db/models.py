"""
src/db/models.py — Database models for logging and trend tracking.
Owned by: Person 3
"""
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class ForwardCheck(SQLModel, table=True):
    """Logs every fact-check request for trend analysis and auditing."""
    id: Optional[int] = Field(default=None, primary_key=True)
    request_id: str = Field(index=True)
    user_id: int
    message_type: str
    verdict: str
    confidence_score: float
    confidence_level: str
    ai_generation_score: float = 0.0
    manipulation_score: float = 0.0
    recycled_image: bool = False
    has_fact_check: bool = False
    claim_type: str = "other"
    extracted_claim: str = ""
    latency_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FactCheckCache(SQLModel, table=True):
    """Caches fact-check search results to avoid re-scraping."""
    id: Optional[int] = Field(default=None, primary_key=True)
    claim_hash: str = Field(index=True, unique=True)   # SHA256 of normalised claim
    source_name: str
    source_url: str
    verdict: str
    snippet: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
