"""
src/pipelines/text/adapters/pib.py — PIB Fact Check adapter.
Performs search on PIB Fact Check (factcheck.pib.gov.in) for Government of India claims.
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


async def search_pib(claim: str, keywords: List[str] = None) -> List[FactCheckMatch]:
    """
    Queries PIB Fact Check.
    PIB focuses on Union Government, ministries, and public sector claims.
    """
    search_term = " ".join(keywords) if keywords else claim[:150]
    log.info("searching_pib_fact_check", term=search_term[:80])

    matches: List[FactCheckMatch] = []

    # 1. Primary PIB search via SerpAPI if key available
    if settings.serpapi_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://serpapi.com/search",
                    params={
                        "engine": "google",
                        "q": f"site:factcheck.pib.gov.in {search_term}",
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
                                source_name="PIB Fact Check",
                                source_url=item.get("link", "https://factcheck.pib.gov.in"),
                                original_claim=item.get("title", ""),
                                fact_check_verdict="PIB_VERIFIED",
                                snippet=item.get("snippet", "")[:250],
                                match_confidence=0.75,
                            )
                        )
        except Exception as e:
            log.warning("pib_serpapi_failed", error=str(e))

    # 2. Direct scraping fallback on factcheck.pib.gov.in
    if not matches:
        try:
            query_encoded = search_term.replace(" ", "+")
            async with httpx.AsyncClient(timeout=10.0, headers=HEADERS, follow_redirects=True) as client:
                resp = await client.get(f"https://factcheck.pib.gov.in/Home/Search?q={query_encoded}")
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "lxml")
                    for card in soup.select(".fact-check-card, .search-result, article, .item")[:5]:
                        title_el = card.select_one("h2, h3, .title, a")
                        link_el = card.select_one("a[href]")
                        snippet_el = card.select_one("p, .description, .summary")

                        if title_el:
                            href = link_el.get("href", "") if link_el else ""
                            full_url = href if href.startswith("http") else f"https://factcheck.pib.gov.in{href}"
                            matches.append(
                                FactCheckMatch(
                                    source_name="PIB Fact Check",
                                    source_url=full_url,
                                    original_claim=title_el.get_text(strip=True),
                                    fact_check_verdict="PIB_VERIFIED",
                                    snippet=snippet_el.get_text(strip=True)[:250] if snippet_el else "",
                                    match_confidence=0.70,
                                )
                            )
        except Exception as e:
            log.warning("pib_direct_scrape_failed", error=str(e))

    return matches
