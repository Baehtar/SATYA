"""
src/pipelines/image/reverse_search.py — Reverse image search + date comparison.
Owned by: Person 1

Uses SerpAPI Google Lens to find earlier appearances of an image.
Compares earliest found date against the claim date to detect recycled images.
"""
import asyncio
import re
import httpx
import structlog
from datetime import datetime, timezone
from typing import Optional
from src.config import settings
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
      }
    """
    if not image_path:
        return {"results": [], "earliest_date": None, "recycled": False, "recycled_confidence": 0.0}

    results = await _serpapi_google_lens(image_path)
    earliest_date = _find_earliest_date(results)
    recycled, recycled_conf = _check_recycled(earliest_date, claimed_date)

    log.info(
        "reverse_search_done",
        n_results=len(results),
        earliest_date=earliest_date,
        recycled=recycled,
    )

    return {
        "results": results,
        "earliest_date": earliest_date,
        "recycled": recycled,
        "recycled_confidence": recycled_conf,
    }


async def _serpapi_google_lens(image_path: str) -> list[ReverseImageResult]:
    """Calls SerpAPI Google Lens endpoint with the image file."""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            with open(image_path, "rb") as f:
                response = await client.post(
                    "https://serpapi.com/search",
                    data={
                        "engine": "google_lens",
                        "api_key": settings.serpapi_key,
                    },
                    files={"image_file": ("image.jpg", f, "image/jpeg")},
                )
            response.raise_for_status()
            data = response.json()

        results = []
        # Parse visual matches
        for item in data.get("visual_matches", [])[:10]:
            results.append(ReverseImageResult(
                url=item.get("link", ""),
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                date_published=_extract_date_from_item(item),
                source_domain=item.get("source", ""),
            ))

        # Also parse pages_with_matching_images
        for item in data.get("pages_with_matching_images", [])[:5]:
            results.append(ReverseImageResult(
                url=item.get("link", ""),
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                date_published=_extract_date_from_item(item),
                source_domain=item.get("source", ""),
            ))

        return results

    except Exception as e:
        log.error("serpapi_reverse_search_failed", error=str(e))
        return []


def _extract_date_from_item(item: dict) -> Optional[str]:
    """Tries to extract a publication date from a search result item."""
    # Direct date field
    if "date" in item:
        return item["date"]

    # Try to parse from snippet or title using regex
    text = f"{item.get('title', '')} {item.get('snippet', '')}"
    date_patterns = [
        r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b',
        r'\b(\d{4}-\d{2}-\d{2})\b',
        r'\b(\d{1,2}/\d{1,2}/\d{4})\b',
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def _find_earliest_date(results: list[ReverseImageResult]) -> Optional[str]:
    """Finds the earliest publication date from all search results."""
    parsed_dates = []
    for r in results:
        if not r.date_published:
            continue
        dt = _parse_date(r.date_published)
        if dt:
            parsed_dates.append((dt, r.date_published))

    if not parsed_dates:
        return None

    parsed_dates.sort(key=lambda x: x[0])
    return parsed_dates[0][1]


def _parse_date(date_str: str) -> Optional[datetime]:
    """Try multiple date formats."""
    formats = [
        "%Y-%m-%d",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _check_recycled(
    earliest_date: Optional[str],
    claimed_date: Optional[str],
    min_days_gap: int = 30,
) -> tuple[bool, float]:
    """
    Compares the earliest known appearance of the image against the claimed event date.
    Returns (recycled: bool, confidence: float).
    """
    if not earliest_date:
        return False, 0.0

    earliest_dt = _parse_date(earliest_date)
    if not earliest_dt:
        return False, 0.0

    # If a specific claimed date is given, compare against it
    if claimed_date:
        claim_dt = _parse_date(claimed_date)
        if claim_dt:
            delta_days = (claim_dt - earliest_dt).days
            if delta_days > min_days_gap:
                confidence = min(1.0, delta_days / 365.0)  # caps at 1 year
                return True, confidence

    # If no claimed date, check if the image is older than 6 months
    # (old images shared as "recent" is the main pattern)
    now = datetime.now(timezone.utc)
    age_days = (now - earliest_dt).days
    if age_days > 180:
        confidence = min(1.0, age_days / 730.0)  # caps at 2 years
        return True, confidence * 0.6  # lower confidence without explicit claim date

    return False, 0.0
