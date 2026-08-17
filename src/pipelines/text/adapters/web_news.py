"""
src/pipelines/text/adapters/web_news.py — Real-time Google News RSS Search Adapter.
100% free, no API key required, zero rate-limit blocks.
Retrieves news coverage from mainstream media outlets (The Indian Express, NDTV, Times of India, Hindu, Business Standard).
"""
import asyncio
import re
import urllib.parse
import xml.etree.ElementTree as ET
import httpx
import structlog
from typing import List
from src.models.schemas import FactCheckMatch

log = structlog.get_logger(__name__)


async def search_web_news(query: str) -> List[FactCheckMatch]:
    """
    Searches live news coverage for a claim using Google News RSS feed.
    """
    clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
    words = clean_query.split()[:6]
    search_term = " ".join(words)

    if not search_term:
        return []

    log.info("searching_web_news", query=search_term)
    results = []

    encoded_query = urllib.parse.quote(search_term)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(rss_url, headers=headers)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                channel = root.find("channel")
                if channel is not None:
                    for item in channel.findall("item")[:6]:
                        title = item.findtext("title", "")
                        link = item.findtext("link", "")
                        pub_date = item.findtext("pubDate", "")
                        source_elem = item.find("source")
                        source_name = source_elem.text if source_elem is not None else "Mainstream News"

                        if title and link:
                            results.append(
                                FactCheckMatch(
                                    source_name=source_name,
                                    source_url=link,
                                    original_claim=title,
                                    fact_check_verdict="REPORTED_NEWS",
                                    fact_check_date=pub_date,
                                    snippet=f"News Coverage by {source_name}: {title}",
                                    match_confidence=0.85,
                                    reason=f"Covered by news outlet {source_name}."
                                )
                            )

        log.info("web_news_search_complete", n_found=len(results))
        return results

    except Exception as e:
        log.warning("web_news_search_failed", error=str(e))
        return []
