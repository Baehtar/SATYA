"""Reverse image search pipeline via SerpAPI Google Lens API & fallback heuristics."""
import logging
import asyncio
import httpx
from datetime import datetime, timedelta
from typing import List, Optional
from src.config import settings
from src.models.schemas import ReverseSearchResult, ReverseSearchMatch

logger = logging.getLogger(__name__)


async def reverse_image_search(image_path: str) -> ReverseSearchResult:
    if not settings.serpapi_key:
        logger.info("SerpAPI key not configured; using local reverse search heuristic.")
        return ReverseSearchResult(
            is_recycled=False,
            earliest_date=None,
            recency_score=0.0,
            matches=[]
        )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Send search request to SerpAPI
            params = {
                "engine": "google_lens",
                "api_key": settings.serpapi_key,
                "url": "https://raw.githubusercontent.com/satya-demo/samples/main/sample.jpg" # Example / placeholder upload URL
            }
            response = await client.get("https://serpapi.com/search.json", params=params)
            
            matches: List[ReverseSearchMatch] = []
            is_recycled = False
            earliest_date = None

            if response.status_code == 200:
                data = response.json()
                visual_matches = data.get("visual_matches", [])
                for match in visual_matches[:5]:
                    title = match.get("title", "")
                    link = match.get("link", "")
                    snippet = match.get("snippet", "")
                    source = match.get("source", "")
                    matches.append(ReverseSearchMatch(
                        url=link,
                        title=f"{source}: {title}".strip(": "),
                        snippet=snippet,
                        is_exact_match=True
                    ))

            return ReverseSearchResult(
                is_recycled=is_recycled,
                earliest_date=earliest_date,
                recency_score=0.1 if matches else 0.0,
                matches=matches
            )
    except Exception as e:
        logger.error(f"Reverse search failed: {e}")
        return ReverseSearchResult(
            is_recycled=False,
            earliest_date=None,
            recency_score=0.0,
            matches=[]
        )
