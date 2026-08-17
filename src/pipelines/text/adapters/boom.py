"""
src/pipelines/text/adapters/boom.py — BOOM Live Fact Check adapter.
Searches boomlive.in (English & Hindi) for viral claim verifications.
"""
import httpx
import structlog
from bs4 import BeautifulSoup
from typing import List
from src.config import settings
from src.models.schemas import FactCheckMatch

log = structlog.get_logger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


async def search_boom(claim: str, keywords: List[str] = None) -> List[FactCheckMatch]:
    """
    Queries BOOM Live fact-check archives.
    Covering social media claims, viral messages, deepfakes, and misinformation.
    """
    search_term = " ".join(keywords) if keywords else claim[:150]
    log.info("searching_boom_live", term=search_term[:80])

    matches: List[FactCheckMatch] = []

    # 1. SerpAPI site:boomlive.in
    if settings.serpapi_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://serpapi.com/search",
                    params={
                        "engine": "google",
                        "q": f"site:boomlive.in {search_term}",
                        "api_key": settings.serpapi_key,
                        "num": 5,
                        "gl": "in",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("organic_results", []):
                        matches.append(
                            FactCheckMatch(
                                source_name="BOOM Live",
                                source_url=item.get("link", "https://boomlive.in"),
                                original_claim=item.get("title", ""),
                                fact_check_verdict="",
                                snippet=item.get("snippet", "")[:250],
                                match_confidence=0.75,
                            )
                        )
        except Exception as e:
            log.warning("boom_serpapi_failed", error=str(e))

    # 2. Direct web search fallback
    if not matches:
        try:
            query_encoded = search_term.replace(" ", "+")
            async with httpx.AsyncClient(timeout=10.0, headers=HEADERS, follow_redirects=True) as client:
                resp = await client.get(f"https://www.boomlive.in/search?q={query_encoded}")
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "lxml")
                    for card in soup.select(".search-card, .story-card, article, .news-card")[:5]:
                        title_el = card.select_one("h2, h3, .title, a")
                        snippet_el = card.select_one("p, .summary, .deck")

                        if title_el:
                            href = title_el.get("href", "") if title_el.name == "a" else (title_el.find("a")["href"] if title_el.find("a") else "")
                            full_url = href if href.startswith("http") else f"https://www.boomlive.in{href}"
                            matches.append(
                                FactCheckMatch(
                                    source_name="BOOM Live",
                                    source_url=full_url,
                                    original_claim=title_el.get_text(strip=True),
                                    fact_check_verdict="",
                                    snippet=snippet_el.get_text(strip=True)[:250] if snippet_el else "",
                                    match_confidence=0.70,
                                )
                            )
        except Exception as e:
            log.warning("boom_scrape_failed", error=str(e))

    return matches
