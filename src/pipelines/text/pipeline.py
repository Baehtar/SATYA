"""
src/pipelines/text/pipeline.py — Multilingual Text Claim Verification Pipeline.
Orchestrates claim extraction → source search (PIB, Alt News, BOOM) → semantic matching.
"""
import asyncio
import time
import structlog
from src.models.schemas import CheckRequest, ClaimAnalysis, Verdict, LanguageCode
from src.config import settings
from src.pipelines.text.claim_extractor import extract_claim
from src.pipelines.text.fact_check_search import search_fact_checks
from src.pipelines.text.claim_matcher import match_and_summarise

log = structlog.get_logger(__name__)


async def run_text_pipeline(request: CheckRequest, progress_callback=None) -> ClaimAnalysis:
    """
    Main entry point for the multilingual text claim pipeline.

    progress_callback(message: str, step: str) — optional async hook so callers
    (web UI, bot) can stream stage-by-stage progress to the user.
    """
    start = time.monotonic()
    text = request.text_content or ""
    log.info("text_pipeline_start", request_id=request.request_id, text_len=len(text))

    if not text.strip():
        return ClaimAnalysis(
            raw_text=text,
            extracted_claim="",
            error="No text content provided",
            pipeline_latency_ms=0,
        )

    try:
        async with asyncio.timeout(settings.text_pipeline_timeout):
            # Step 1: Language Detection & Claim Extraction
            if progress_callback:
                await progress_callback("📝 Extracting the core claim…", "text_analysis")
            claim_info = await extract_claim(text)

            # Uncheckable opinion / satire handling
            if not claim_info.get("is_checkable", True):
                return ClaimAnalysis(
                    raw_text=text,
                    extracted_claim=claim_info.get("claim", text[:200]),
                    language=claim_info.get("language", LanguageCode.EN),
                    is_checkable=False,
                    text_verdict=Verdict.UNVERIFIABLE,
                    text_verdict_confidence=0.8,
                    no_match_reason="This text appears to be a personal opinion, sarcastic post, or uncheckable assertion.",
                    pipeline_latency_ms=int((time.monotonic() - start) * 1000),
                )

            # Step 2: Search Fact-Check Sources (PIB, Alt News, BOOM)
            if progress_callback:
                await progress_callback(
                    "🔎 Searching PIB, Alt News, BOOM & Google News…", "fact_check"
                )
            matches = await search_fact_checks(
                claim=claim_info.get("claim", ""),
                keywords=claim_info.get("keywords", []),
                claim_type=claim_info.get("claim_type", "other"),
                search_queries=claim_info.get("search_queries", []),
            )

            # Step 3: Semantic Match & Summarise Evidence
            if progress_callback:
                await progress_callback(
                    f"⚖️ Weighing {len(matches)} source(s) against the claim…",
                    "generating_verdict",
                )
            analysis = await match_and_summarise(
                raw_text=text,
                claim_info=claim_info,
                matches=matches,
            )

    except asyncio.TimeoutError:
        log.warning("text_pipeline_timeout", request_id=request.request_id)
        return ClaimAnalysis(
            raw_text=text,
            extracted_claim=text[:200],
            text_verdict=Verdict.UNVERIFIABLE,
            text_verdict_confidence=0.5,
            error="Text verification pipeline timed out",
            pipeline_latency_ms=int((time.monotonic() - start) * 1000),
        )
    except Exception as e:
        log.error("text_pipeline_error", request_id=request.request_id, error=str(e))
        return ClaimAnalysis(
            raw_text=text,
            extracted_claim=text[:200],
            text_verdict=Verdict.UNVERIFIABLE,
            text_verdict_confidence=0.5,
            error=str(e),
            pipeline_latency_ms=int((time.monotonic() - start) * 1000),
        )

    analysis.pipeline_latency_ms = int((time.monotonic() - start) * 1000)
    log.info(
        "text_pipeline_complete",
        request_id=request.request_id,
        claim=analysis.extracted_claim[:80],
        verdict=analysis.text_verdict.value,
        n_matches=len(analysis.matches),
        latency_ms=analysis.pipeline_latency_ms,
    )
    return analysis
