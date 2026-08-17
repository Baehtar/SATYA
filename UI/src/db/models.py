from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field

class ForwardCheck(SQLModel, table=True):
    id: str = Field(primary_key=True)
    message_type: str
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class CachedFactCheck(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    claim_hash: str = Field(index=True)
    source_name: str
    source_url: str
    verdict: str
    cached_at: datetime = Field(default_factory=datetime.utcnow)
