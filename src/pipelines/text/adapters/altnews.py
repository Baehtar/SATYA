"""
src/pipelines/text/adapters/altnews.py — Alt News Fact Check adapter.
Searches altnews.in and Alt News Hindi archives.
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


async def search_altnews(claim: str, keywords: List[str] = None) -> List[FactCheckMatch]:
    """
    Queries Alt News archive.
    Covers broader viral claims, video/image fabrications, and social media misinformation.
    """
    search_term = " ".join(keywords) if keywords else claim[:150]
    log.info("searching_altnews", term=search_term[:80])

    matches: List[FactCheckMatch] = []

    # 1. SerpAPI search site:altnews.in
    if settings.serpapi_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://serpapi.com/search",
                    params={
                        "engine": "google",
                        "q": f"site:altnews.in {search_term}",
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
                                source_name="Alt News",
                                source_url=item.get("link", "https://altnews.in"),
                                original_claim=item.get("title", ""),
                                fact_check_verdict="",
                                snippet=item.get("snippet", "")[:250],
                                match_confidence=0.75,
                            )
                        )
        except Exception as e:
            log.warning("altnews_serpapi_failed", error=str(e))

    # 2. Direct site search fallback
    if not matches:
        try:
            query_encoded = search_term.replace(" ", "+")
            async with httpx.AsyncClient(timeout=10.0, headers=HEADERS, follow_redirects=True) as client:
                resp = await client.get(f"https://www.altnews.in/?s={query_encoded}")
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "lxml")
                    for article in soup.select("article, .post-entry, .entry-title")[:5]:
                        title_el = article.select_one("h2, h3, .entry-title a, a")
                        snippet_el = article.select_one(".entry-summary, p, .post-excerpt")
                        if title_el:
                            href = title_el.get("href", "") if title_el.name == "a" else (title_el.find("a")["href"] if title_el.find("a") else "")
                            matches.append(
                                FactCheckMatch(
                                    source_name="Alt News",
                                    source_url=href or "https://altnews.in",
                                    original_claim=title_el.get_text(strip=True),
                                    fact_check_verdict="",
                                    snippet=snippet_el.get_text(strip=True)[:250] if snippet_el else "",
                                    match_confidence=0.70,
                                )
                            )
        except Exception as e:
            log.warning("altnews_scrape_failed", error=str(e))

    return matches
