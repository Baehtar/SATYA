"""
services/image/match_ranker.py — Match confidence ranking and URL canonical deduplication.
Ranks reverse image search results into weighted match types (EXACT_MATCH, FULL_MATCH, PARTIAL_MATCH, etc.).
"""
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from typing import List, Dict, Any, Optional
import structlog

log = structlog.get_logger(__name__)

MATCH_TYPE_WEIGHTS = {
    "EXACT_MATCH": 1.00,
    "FULL_MATCH": 0.90,
    "PARTIAL_MATCH": 0.75,
    "HIGH_VISUAL_SIMILARITY": 0.40,
    "LOW_VISUAL_SIMILARITY": 0.15,
}


def normalize_canonical_url(url: str) -> str:
    """Normalizes URL by stripping trailing slashes, tracking query params, and fragments."""
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url.strip())
        # Filter out common tracking query params
        filtered_query = [
            (k, v) for k, v in parse_qsl(parsed.query)
            if not k.lower().startswith("utm_") and k.lower() not in ["fbclid", "gclid", "ref"]
        ]
        canonical_query = urlencode(filtered_query)
        canonical_path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            canonical_path,
            parsed.params,
            canonical_query,
            ""  # Strip fragment
        ))
        return normalized
    except Exception:
        return url.strip()


def rank_and_deduplicate_matches(
    google_vision_res: Dict[str, Any],
    serpapi_lens_res: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Deduplicates and ranks matches from Google Vision Web Detection and SerpAPI Google Lens.
    """
    matches_map: Dict[str, Dict[str, Any]] = {}

    # Helper to add match to map
    def add_match(url: str, match_type: str, provider: str, title: str = "", date: Optional[str] = None):
        if not url or not url.startswith("http"):
            return
        canonical = normalize_canonical_url(url)
        weight = MATCH_TYPE_WEIGHTS.get(match_type, 0.15)

        if canonical not in matches_map or weight > matches_map[canonical]["similarity_weight"]:
            matches_map[canonical] = {
                "url": url,
                "canonical_url": canonical,
                "match_type": match_type,
                "similarity_weight": weight,
                "source_provider": provider,
                "page_title": title,
                "published_date": date,
                "date_confidence": 0.0
            }

    # 1. Parse Google Vision Results
    if google_vision_res:
        for page in google_vision_res.get("pages_with_matching_images", []):
            u = page.get("url", "")
            title = page.get("page_title", "")
            if page.get("full_matching_images"):
                add_match(u, "EXACT_MATCH", "google_vision", title)
            elif page.get("partial_matching_images"):
                add_match(u, "PARTIAL_MATCH", "google_vision", title)
            else:
                add_match(u, "FULL_MATCH", "google_vision", title)

        for img in google_vision_res.get("full_matching_images", []):
            add_match(img.get("url", ""), "EXACT_MATCH", "google_vision")

        for img in google_vision_res.get("partial_matching_images", []):
            add_match(img.get("url", ""), "PARTIAL_MATCH", "google_vision")

        for img in google_vision_res.get("visually_similar_images", []):
            add_match(img.get("url", ""), "HIGH_VISUAL_SIMILARITY", "google_vision")

    # 2. Parse SerpAPI Google Lens Results
    if serpapi_lens_res:
        for m in serpapi_lens_res.get("exact_matches", []):
            add_match(m.get("url", ""), "EXACT_MATCH", "serpapi_google_lens", m.get("title", ""), m.get("date"))

        for m in serpapi_lens_res.get("pages_with_matching_images", []):
            add_match(m.get("url", ""), "FULL_MATCH", "serpapi_google_lens", m.get("title", ""), m.get("date"))

        for m in serpapi_lens_res.get("visual_matches", []):
            add_match(m.get("url", ""), "HIGH_VISUAL_SIMILARITY", "serpapi_google_lens", m.get("title", ""), m.get("date"))

    # Convert map to list sorted by similarity_weight descending
    ranked_list = list(matches_map.values())
    ranked_list.sort(key=lambda x: x["similarity_weight"], reverse=True)

    log.info("matches_ranked_and_deduplicated", total_unique=len(ranked_list))
    return ranked_list
