"""
src/db/database.py — Async database session management.
Owned by: Person 3
"""
from sqlmodel import SQLModel, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from src.config import settings

engine = create_async_engine(settings.database_url, echo=settings.debug)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def log_check(check_data: dict) -> None:
    """Convenience function to log a completed check."""
    from src.db.models import ForwardCheck
    async with AsyncSessionLocal() as session:
        record = ForwardCheck(**check_data)
        session.add(record)
        await session.commit()
