"""
src/pipelines/image/pipeline.py — Image pipeline orchestrator.
Owned by: Person 1

Runs AI detection, manipulation detection, and reverse image search in parallel.
"""
import asyncio
import time
import structlog
from src.models.schemas import CheckRequest, ImageAnalysis
from src.config import settings

log = structlog.get_logger(__name__)


async def run_image_pipeline(request: CheckRequest) -> ImageAnalysis:
    """
    Main entry point for the image pipeline.
    Runs all three sub-pipelines in parallel and merges results.
    """
    start = time.monotonic()
    log.info("image_pipeline_start", request_id=request.request_id)

    from src.pipelines.image.ai_detector import detect_ai_generation
    from src.pipelines.image.manipulation_detector import detect_manipulation
    from src.pipelines.image.reverse_search import run_reverse_search

    try:
        async with asyncio.timeout(settings.image_pipeline_timeout):
            ai_task = asyncio.create_task(detect_ai_generation(request.image_path))
            manip_task = asyncio.create_task(detect_manipulation(request.image_path))
            reverse_task = asyncio.create_task(
                run_reverse_search(request.image_path, request.claimed_date)
            )

            ai_result, manip_result, reverse_result = await asyncio.gather(
                ai_task, manip_task, reverse_task, return_exceptions=True
            )

    except asyncio.TimeoutError:
        log.warning("image_pipeline_timeout", request_id=request.request_id)
        return ImageAnalysis(
            ai_generation_score=0.0,
            manipulation_score=0.0,
            error="Image pipeline timed out",
            pipeline_latency_ms=int((time.monotonic() - start) * 1000),
        )

    # Merge results, gracefully handling partial failures
    analysis = ImageAnalysis(
        ai_generation_score=ai_result.get("score", 0.0) if not isinstance(ai_result, Exception) else 0.0,
        ai_generation_model_used=ai_result.get("model", "") if not isinstance(ai_result, Exception) else "",
        manipulation_score=manip_result.get("score", 0.0) if not isinstance(manip_result, Exception) else 0.0,
        ela_heatmap_path=manip_result.get("heatmap_path") if not isinstance(manip_result, Exception) else None,
        exif_anomalies=manip_result.get("exif_anomalies", []) if not isinstance(manip_result, Exception) else [],
        noise_inconsistency=manip_result.get("noise_inconsistency", 0.0) if not isinstance(manip_result, Exception) else 0.0,
        reverse_search_results=reverse_result.get("results", []) if not isinstance(reverse_result, Exception) else [],
        earliest_appearance_date=reverse_result.get("earliest_date") if not isinstance(reverse_result, Exception) else None,
        recycled_image=reverse_result.get("recycled", False) if not isinstance(reverse_result, Exception) else False,
        recycled_confidence=reverse_result.get("recycled_confidence", 0.0) if not isinstance(reverse_result, Exception) else 0.0,
        pipeline_latency_ms=int((time.monotonic() - start) * 1000),
        error="; ".join(str(e) for e in [ai_result, manip_result, reverse_result] if isinstance(e, Exception)) or None,
    )

    log.info(
        "image_pipeline_done",
        request_id=request.request_id,
        ai_score=analysis.ai_generation_score,
        manip_score=analysis.manipulation_score,
        recycled=analysis.recycled_image,
        latency_ms=analysis.pipeline_latency_ms,
    )
    return analysis
