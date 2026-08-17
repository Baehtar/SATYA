"""
src/pipelines/text/pipeline.py — Text claim pipeline orchestrator.
Owned by: Person 2

Sequential: extract claim → search fact-checks → match & summarise.
"""
import asyncio
import time
import structlog
from src.models.schemas import CheckRequest, ClaimAnalysis, Verdict
from src.config import settings

log = structlog.get_logger(__name__)


async def run_text_pipeline(request: CheckRequest) -> ClaimAnalysis:
    """Main entry point for the text pipeline."""
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

    from src.pipelines.text.claim_extractor import extract_claim
    from src.pipelines.text.fact_check_search import search_fact_checks
    from src.pipelines.text.claim_matcher import match_and_summarise

    try:
        async with asyncio.timeout(settings.text_pipeline_timeout):
            # Step 1: Extract core claim
            claim_info = await extract_claim(text)

            if not claim_info.get("is_checkable", True):
                return ClaimAnalysis(
                    raw_text=text,
                    extracted_claim=claim_info.get("claim", text[:200]),
                    is_checkable=False,
                    text_verdict=Verdict.UNVERIFIABLE,
                    text_verdict_confidence=0.9,
                    no_match_reason="This appears to be an opinion, not a checkable factual claim.",
                    pipeline_latency_ms=int((time.monotonic() - start) * 1000),
                )

            # Step 2: Search fact-check sources
            matches = await search_fact_checks(
                claim=claim_info.get("claim", ""),
                entities=claim_info.get("entities", {}),
            )

            # Step 3: Match and summarise
            analysis = await match_and_summarise(
                raw_text=text,
                claim_info=claim_info,
                matches=matches,
            )

    except asyncio.TimeoutError:
        log.warning("text_pipeline_timeout", request_id=request.request_id)
        return ClaimAnalysis(
            raw_text=text,
            extracted_claim="",
            error="Text pipeline timed out",
            pipeline_latency_ms=int((time.monotonic() - start) * 1000),
        )
    except Exception as e:
        log.error("text_pipeline_error", request_id=request.request_id, error=str(e))
        return ClaimAnalysis(
            raw_text=text,
            extracted_claim="",
            error=str(e),
            pipeline_latency_ms=int((time.monotonic() - start) * 1000),
        )

    analysis.pipeline_latency_ms = int((time.monotonic() - start) * 1000)
    log.info(
        "text_pipeline_done",
        request_id=request.request_id,
        claim=analysis.extracted_claim[:80],
        n_matches=len(analysis.matches),
        verdict=analysis.text_verdict,
        latency_ms=analysis.pipeline_latency_ms,
    )
    return analysis
