"""
src/pipelines/text/fact_check_search.py — Parallel fact-check source search.
Owned by: Person 2

Searches PIB Fact Check, AltNews, BOOM, and Google Fact Check Tools API
in parallel. Returns a list of FactCheckMatch objects.
"""
import asyncio
import httpx
import structlog
from bs4 import BeautifulSoup
from typing import Optional
from src.config import settings
from src.models.schemas import FactCheckMatch

log = structlog.get_logger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SatyaFactChecker/1.0; +https://satya-bot.in)",
}


async def search_fact_checks(claim: str, entities: dict) -> list[FactCheckMatch]:
    """
    Searches all fact-check sources in parallel.
    Returns deduplicated list of FactCheckMatch objects, sorted by match confidence.
    """
    log.info("searching_fact_checks", claim=claim[:80])

    tasks = [
        _search_google_factcheck_api(claim),
        _search_pib(claim),
        _search_via_google_site(claim, "altnews.in", "AltNews"),
        _search_via_google_site(claim, "boomlive.in", "BOOM Live"),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_matches: list[FactCheckMatch] = []
    for r in results:
        if isinstance(r, list):
            all_matches.extend(r)
        elif isinstance(r, Exception):
            log.warning("fact_check_source_failed", error=str(r))

    # Deduplicate by URL
    seen_urls = set()
    unique = []
    for m in all_matches:
        if m.source_url not in seen_urls:
            seen_urls.add(m.source_url)
            unique.append(m)

    log.info("fact_check_search_done", n_matches=len(unique))
    return unique


async def _search_google_factcheck_api(claim: str) -> list[FactCheckMatch]:
    """Google Fact Check Tools API — covers ClaimReview from multiple publishers."""
    if not settings.google_factcheck_api_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=HEADERS) as client:
            r = await client.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params={
                    "query": claim[:200],
                    "key": settings.google_factcheck_api_key,
                    "languageCode": "en",
                    "pageSize": 5,
                },
            )
            r.raise_for_status()
            data = r.json()

        matches = []
        for item in data.get("claims", []):
            review = item.get("claimReview", [{}])[0]
            matches.append(FactCheckMatch(
                source_name=review.get("publisher", {}).get("name", "Unknown"),
                source_url=review.get("url", ""),
                original_claim=item.get("text", ""),
                fact_check_verdict=review.get("textualRating", ""),
                fact_check_date=review.get("reviewDate", ""),
                snippet=review.get("title", "")[:200],
                match_confidence=0.7,  # will be refined by claim_matcher
            ))
        return matches

    except Exception as e:
        log.warning("google_factcheck_api_failed", error=str(e))
        return []


async def _search_pib(claim: str) -> list[FactCheckMatch]:
    """Search PIB Fact Check website."""
    try:
        query = claim[:150].replace(" ", "+")
        async with httpx.AsyncClient(timeout=10.0, headers=HEADERS, follow_redirects=True) as client:
            r = await client.get(
                f"https://factcheck.pib.gov.in/Home/Search?q={query}",
            )
            r.raise_for_status()

        soup = BeautifulSoup(r.text, "lxml")
        matches = []

        for card in soup.select(".fact-check-card, .search-result, article")[:5]:
            title_el = card.select_one("h2, h3, .title, a")
            link_el = card.select_one("a[href]")
            snippet_el = card.select_one("p, .description, .summary")

            if not title_el:
                continue

            matches.append(FactCheckMatch(
                source_name="PIB Fact Check",
                source_url=_make_absolute(link_el.get("href", ""), "https://factcheck.pib.gov.in") if link_el else "",
                original_claim=title_el.get_text(strip=True),
                fact_check_verdict="",
                snippet=snippet_el.get_text(strip=True)[:200] if snippet_el else "",
                match_confidence=0.6,
            ))

        return matches

    except Exception as e:
        log.warning("pib_search_failed", error=str(e))
        return []


async def _search_via_google_site(claim: str, site: str, source_name: str) -> list[FactCheckMatch]:
    """
    Uses SerpAPI Google Search to find articles on a specific fact-check site.
    More reliable than scraping the site directly.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://serpapi.com/search",
                params={
                    "engine": "google",
                    "q": f"site:{site} {claim[:150]}",
                    "api_key": settings.serpapi_key,
                    "num": 5,
                    "gl": "in",
                    "hl": "en",
                },
            )
            r.raise_for_status()
            data = r.json()

        matches = []
        for result in data.get("organic_results", []):
            matches.append(FactCheckMatch(
                source_name=source_name,
                source_url=result.get("link", ""),
                original_claim=result.get("title", ""),
                fact_check_verdict="",  # will be extracted by claim_matcher
                fact_check_date=result.get("date", ""),
                snippet=result.get("snippet", "")[:200],
                match_confidence=0.65,
            ))

        return matches

    except Exception as e:
        log.warning("site_search_failed", site=site, error=str(e))
        return []


def _make_absolute(url: str, base: str) -> str:
    if url.startswith("http"):
        return url
    return base.rstrip("/") + "/" + url.lstrip("/")
