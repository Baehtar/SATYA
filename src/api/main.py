"""
src/api/main.py — FastAPI backend for the Satya fact-check pipeline.

This is what the Telegram bot calls. Exposes the full pipeline as HTTP endpoints.

Run with:
    uvicorn src.api.main:app --reload --port 8000

Endpoints:
    POST /check/text        — fact-check a text forward
    POST /check/image       — fact-check an image
    POST /check/audio       — fact-check a voice note
    POST /check             — unified endpoint (auto-detects type from multipart)
    GET  /health            — liveness check
    GET  /verdict/{id}      — retrieve a past verdict by request_id
"""
import uuid
import time
import structlog
import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.check import router as check_router
from src.db.database import init_db
from src.config import settings

log = structlog.get_logger(__name__)
logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic."""
    log.info("satya_api_starting")
    await init_db()
    log.info("database_initialised")
    yield
    log.info("satya_api_shutdown")


app = FastAPI(
    title="Satya Fact-Check API",
    description=(
        "Backend for the Satya AI Forward-Checker. "
        "Send suspicious images, text, or audio — receive a calibrated credibility verdict."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(check_router, prefix="/check", tags=["Fact-Check"])


@app.get("/health", tags=["Meta"])
async def health():
    return {"status": "ok", "service": "satya-api", "version": "1.0.0"}
