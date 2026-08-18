"""
src/pipelines/image/reverse_search.py — Reverse image search + date comparison.
Owned by: Person 1

Thin adapter. The implementation lives in services/image/ (the Reverse Image
Engine), which is shared with services/ml_service.py so the Telegram bot, the
web UI and this pipeline all reach the same conclusions from the same code.

What the engine adds over the original SerpAPI-only version this file used to
contain:
  * Google Vision Web Detection as the primary provider — it takes the local
    file directly, so no public URL is needed and nothing is published
  * matches deduplicated across providers and ranked by match strength, so a
    merely similar photo can no longer drive a recycled verdict
  * publication dates read from the pages themselves (JSON-LD → meta → <time> →
    text → URL) instead of trusting whatever the search result carried
"""
import structlog
from typing import Optional

from src.models.schemas import ReverseImageResult

log = structlog.get_logger(__name__)


async def run_reverse_search(image_path: str, claimed_date: Optional[str] = None) -> dict:
    """
    Performs reverse image search and date comparison.
    Returns:
      {
        "results": List[ReverseImageResult],
        "earliest_date": str | None,
        "recycled": bool,
        "recycled_confidence": float,
        "provenance": dict,          # the full engine result
      }
    """
    empty = {
        "results": [], "earliest_date": None, "recycled": False,
        "recycled_confidence": 0.0, "provenance": {},
    }
    if not image_path:
        return empty

    from services.image import reverse_image_check
    from services.image.reverse_engine import STATUS_RECYCLED

    provenance = await reverse_image_check(image_path, claim_date=claimed_date)

    results = [
        ReverseImageResult(
            url=m.get("url", ""),
            title=m.get("page_title", ""),
            snippet=f"{m.get('match_type', '')} via {', '.join(m.get('sources', []))}",
            date_published=m.get("published_date"),
            source_domain=m.get("domain", ""),
        )
        for m in provenance.get("reverse_matches", [])
    ]

    recycled = provenance.get("image_status") == STATUS_RECYCLED

    log.info(
        "reverse_search_done",
        n_results=len(results),
        earliest_date=provenance.get("earliest_located_date"),
        status=provenance.get("image_status"),
    )

    return {
        "results": results,
        "earliest_date": provenance.get("earliest_located_date"),
        "recycled": recycled,
        "recycled_confidence": float(provenance.get("status_confidence") or 0.0),
        "provenance": provenance,
    }
