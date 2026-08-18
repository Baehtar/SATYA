"""
src/dashboard/app.py — Trend dashboard.
Owned by: Person 4

The dashboard is an APIRouter plus one HTML page, so it can be mounted into the
Satya web portal (UI/src/server.py) instead of running as a second service.
`app` is kept for running it on its own:

    uvicorn src.dashboard.app:app --port 8080
"""
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse
from sqlalchemy import func, select

from src.db.database import AsyncSessionLocal, init_db
from src.db.models import ForwardCheck

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# Mounted by the portal under /api/dashboard — the page fetches those paths.
router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


async def dashboard_page() -> FileResponse:
    """The dashboard single-page app."""
    return FileResponse(INDEX_HTML)



@router.get("/stats")
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


@router.get("/recent")
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


@router.get("/trends")
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


# Standalone mode: same router, same page, on its own port.
app = FastAPI(title="Satya Trend Dashboard", version="1.0")


@app.on_event("startup")
async def on_startup():
    await init_db()


app.include_router(router)
app.add_api_route("/", dashboard_page, methods=["GET"], include_in_schema=False)

