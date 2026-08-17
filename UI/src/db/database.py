import logging
from typing import Optional
from datetime import datetime
from sqlmodel import create_engine, Session, select
from src.config import settings
from src.db.models import SQLModel, ForwardCheck, CachedFactCheck

logger = logging.getLogger(__name__)

engine = create_engine(settings.db_url, echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

def log_check(check_id: str, message_type: str, verdict: Optional[str] = None, confidence: Optional[float] = None, latency_ms: Optional[int] = None):
    with Session(engine) as session:
        check = ForwardCheck(
            id=check_id,
            message_type=message_type,
            verdict=verdict,
            confidence=confidence,
            latency_ms=latency_ms,
            created_at=datetime.utcnow()
        )
        session.add(check)
        session.commit()

def get_cached_fact_check(claim_hash: str) -> Optional[CachedFactCheck]:
    with Session(engine) as session:
        statement = select(CachedFactCheck).where(CachedFactCheck.claim_hash == claim_hash)
        result = session.exec(statement).first()
        return result

def cache_fact_check(claim_hash: str, source_name: str, source_url: str, verdict: str):
    with Session(engine) as session:
        fc = CachedFactCheck(
            claim_hash=claim_hash,
            source_name=source_name,
            source_url=source_url,
            verdict=verdict,
            cached_at=datetime.utcnow()
        )
        session.add(fc)
        session.commit()
