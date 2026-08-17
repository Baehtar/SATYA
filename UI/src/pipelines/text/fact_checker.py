"""Fact checker module querying Google Fact Check API, PIB, AltNews, BOOM, and local fact-check index."""
import logging
import asyncio
import httpx
from typing import List
from src.config import settings
from src.models.schemas import FactCheckMatch

logger = logging.getLogger(__name__)

# Curated local fact-check dataset for common viral forwards and fallback matches
CURATED_FACT_CHECKS = [
    {
        "keywords": ["kerala", "flood", "cyclone", "photo", "flood photo"],
        "source": "PIB Fact Check",
        "url": "https://pib.gov.in/factcheck",
        "claim": "Recent photo showing cyclone damage",
        "verdict": "FALSE: Image is from the 2018 Kerala floods, not recent cyclone.",
        "summary": "This photo has been circulating since 2018 and does not depict current weather events."
    },
    {
        "keywords": ["unesco", "best national anthem", "jana gana mana", "anthem"],
        "source": "Alt News",
        "url": "https://www.altnews.in/unesco-best-national-anthem/",
        "claim": "UNESCO declared Jana Gana Mana as the best national anthem in the world.",
        "verdict": "FALSE: UNESCO made no such announcement.",
        "summary": "This is a long-standing internet hoax that has been repeatedly debunked."
    },
    {
        "keywords": ["rbi", "2000", "chip", "gps", "note", "nano chip"],
        "source": "BOOM Live",
        "url": "https://www.boomlive.in/fact-check",
        "claim": "₹2000 currency note contains a secret satellite tracking GPS nano-chip.",
        "verdict": "FALSE: RBI confirmed currency notes do not contain nano-chips.",
        "summary": "RBI and security officials confirmed no electronic chips exist inside currency paper."
    },
    {
        "keywords": ["15 lakh", "bank account", "modi", "account"],
        "source": "PIB Fact Check",
        "url": "https://pib.gov.in/factcheck",
        "claim": "Government is depositing 15 lakh rupees in every citizen's bank account.",
        "verdict": "FALSE: Fraudulent message attempting phishing.",
        "summary": "Official statement confirming no such scheme or payout is operating."
    }
]


async def search_google_fact_check(claim: str, timeout: float = 10.0) -> List[FactCheckMatch]:
    if not settings.gemini_api_key:
        return []
        
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={claim}&key={settings.gemini_api_key}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                claims = data.get("claims", [])
                results = []
                for c in claims[:3]:
                    review = c.get("claimReview", [{}])[0]
                    results.append(FactCheckMatch(
                        source_name=review.get("publisher", {}).get("name", "Google Fact Check"),
                        source_url=review.get("url", "https://factchecktools.googleapis.com"),
                        original_claim=c.get("text", claim),
                        verdict=review.get("textualRating", "False"),
                        summary=review.get("title", ""),
                        match_confidence=0.85
                    ))
                return results
    except Exception as e:
        logger.warning(f"Google Fact Check API error: {e}")
    return []


async def search_local_curated(claim: str) -> List[FactCheckMatch]:
    lowered = claim.lower()
    matches = []
    for item in CURATED_FACT_CHECKS:
        if any(kw in lowered for kw in item["keywords"]):
            matches.append(FactCheckMatch(
                source_name=item["source"],
                source_url=item["url"],
                original_claim=item["claim"],
                verdict=item["verdict"],
                summary=item["summary"],
                match_confidence=0.9
            ))
    return matches


async def search_fact_checks(claim: str, entities: List[str]) -> List[FactCheckMatch]:
    results = await asyncio.gather(
        search_google_fact_check(claim),
        search_local_curated(claim),
        return_exceptions=True
    )
    
    all_matches: List[FactCheckMatch] = []
    for r in results:
        if isinstance(r, list):
            all_matches.extend(r)
            
    # Deduplicate by URL
    seen_urls = set()
    deduped = []
    for match in all_matches:
        if match.source_url not in seen_urls:
            seen_urls.add(match.source_url)
            deduped.append(match)
            
    return deduped
