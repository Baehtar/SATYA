"""
src/pipelines/text/fact_check_search.py — Parallel fact-check source search with caching.
Searches PIB Fact Check, Alt News, BOOM, and Google Fact Check API concurrently.
"""
import asyncio
import hashlib
import httpx
import structlog
from typing import List, Optional
from src.config import settings
from src.models.schemas import FactCheckMatch
from src.pipelines.text.adapters.pib import search_pib
from src.pipelines.text.adapters.altnews import search_altnews
from src.pipelines.text.adapters.boom import search_boom
from src.pipelines.text.adapters.web_news import search_web_news

log = structlog.get_logger(__name__)

# In-memory cache for fast repeat lookups: claim_hash -> List[FactCheckMatch]
_SEARCH_CACHE = {}


def _compute_claim_hash(claim: str) -> str:
    """Computes SHA-256 fingerprint of normalized claim."""
    clean_claim = claim.strip().lower()
    return hashlib.sha256(clean_claim.encode("utf-8")).hexdigest()


async def search_fact_checks(
    claim: str,
    keywords: List[str] = None,
    claim_type: str = "other",
    search_queries: List[str] = None,
) -> List[FactCheckMatch]:
    """
    Searches all fact-check sources and live web news in parallel.
    Uses caching and resilient fallback handling.
    """
    claim_hash = _compute_claim_hash(claim)
    if claim_hash in _SEARCH_CACHE:
        log.info("fact_check_cache_hit", claim=claim[:60])
        return _SEARCH_CACHE[claim_hash]

    log.info("searching_fact_checks_parallel", claim=claim[:80], claim_type=claim_type)

    queries_to_run = search_queries or [claim]
    primary_query = queries_to_run[0]

    tasks = [
        _search_google_factcheck_api(primary_query),
        search_altnews(primary_query, keywords),
        search_boom(primary_query, keywords),
        search_web_news(primary_query),
    ]

    # PIB focus: query PIB if claim concerns government, political, or financial matters
    if claim_type in ["government", "political", "financial", "disaster", "other"]:
        tasks.append(search_pib(primary_query, keywords))

    # Gather all results resiliently (individual adapter errors don't crash the request)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_matches: List[FactCheckMatch] = []
    for r in results:
        if isinstance(r, list):
            all_matches.extend(r)
        elif isinstance(r, Exception):
            log.warning("fact_check_adapter_failed", error=str(r))

    # Deduplicate by URL
    seen_urls = set()
    unique_matches: List[FactCheckMatch] = []
    for m in all_matches:
        if m.source_url and m.source_url not in seen_urls:
            seen_urls.add(m.source_url)
            unique_matches.append(m)

    # Store in cache
    _SEARCH_CACHE[claim_hash] = unique_matches
    log.info("fact_check_search_complete", n_matches=len(unique_matches))

    return unique_matches


async def _search_google_factcheck_api(claim: str) -> List[FactCheckMatch]:
    """Queries Google Fact Check Tools API."""
    if not settings.google_factcheck_api_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params={
                    "query": claim[:200],
                    "key": settings.google_factcheck_api_key,
                    "languageCode": "en",
                    "pageSize": 5,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                matches = []
                for item in data.get("claims", []):
                    reviews = item.get("claimReview", [])
                    if reviews:
                        review = reviews[0]
                        publisher = review.get("publisher", {}).get("name", "Fact Checker")
                        matches.append(
                            FactCheckMatch(
                                source_name=publisher,
                                source_url=review.get("url", ""),
                                original_claim=item.get("text", ""),
                                fact_check_verdict=review.get("textualRating", ""),
                                fact_check_date=review.get("reviewDate", ""),
                                snippet=review.get("title", "")[:250],
                                match_confidence=0.75,
                            )
                        )
                return matches
    except Exception as e:
        log.warning("google_factcheck_api_error", error=str(e))

    return []
