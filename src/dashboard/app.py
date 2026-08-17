"""
src/dashboard/app.py — Trend dashboard FastAPI app.
Owned by: Person 4

Serves trend data and a simple dashboard frontend.
Run with: uvicorn src.dashboard.app:app --port 8080
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy import select, func, text
from src.db.database import AsyncSessionLocal
from src.db.models import ForwardCheck
import json

app = FastAPI(title="Satya Trend Dashboard", version="1.0")
app.mount("/static", StaticFiles(directory="src/dashboard/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("src/dashboard/static/index.html") as f:
        return f.read()


@app.get("/api/stats")
async def get_stats():
    """Summary stats for the dashboard header."""
    async with AsyncSessionLocal() as session:
        total = await session.scalar(select(func.count(ForwardCheck.id)))
        false_count = await session.scalar(
            select(func.count(ForwardCheck.id)).where(ForwardCheck.verdict == "likely_false")
        )
        unverifiable = await session.scalar(
            select(func.count(ForwardCheck.id)).where(ForwardCheck.verdict == "unverifiable")
        )
        avg_latency = await session.scalar(select(func.avg(ForwardCheck.latency_ms)))

    return {
        "total_checks": total or 0,
        "likely_false": false_count or 0,
        "unverifiable": unverifiable or 0,
        "avg_latency_ms": round(avg_latency or 0),
    }


@app.get("/api/recent")
async def get_recent(limit: int = 20):
    """Most recent checks for the live feed."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ForwardCheck)
            .order_by(ForwardCheck.timestamp.desc())
            .limit(limit)
        )
        checks = result.scalars().all()

    return [
        {
            "request_id": c.request_id,
            "message_type": c.message_type,
            "verdict": c.verdict,
            "confidence": c.confidence_score,
            "claim": c.extracted_claim[:100],
            "latency_ms": c.latency_ms,
            "timestamp": c.timestamp.isoformat(),
        }
        for c in checks
    ]


@app.get("/api/trends")
async def get_trends():
    """Verdict distribution and claim type breakdown for charts."""
    async with AsyncSessionLocal() as session:
        verdict_dist = await session.execute(
            select(ForwardCheck.verdict, func.count(ForwardCheck.id))
            .group_by(ForwardCheck.verdict)
        )
        claim_dist = await session.execute(
            select(ForwardCheck.claim_type, func.count(ForwardCheck.id))
            .group_by(ForwardCheck.claim_type)
        )

    return {
        "verdict_distribution": dict(verdict_dist.all()),
        "claim_type_distribution": dict(claim_dist.all()),
    }
