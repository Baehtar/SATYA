"""
services/image/serpapi_lens.py — SerpAPI Google Lens reverse image search client.
Sends local image files or public image URLs to SerpAPI Google Lens.
"""
import os
import httpx
import structlog
from typing import Dict, Any, List, Optional
from src.config import settings

log = structlog.get_logger(__name__)

SERPAPI_ENDPOINT = "https://serpapi.com/search"


async def search_serpapi_google_lens(image_path: str, image_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs reverse image search via SerpAPI Google Lens endpoint.
    
    Returns:
      {
        "provider": "serpapi_google_lens",
        "visual_matches": List[Dict[str, str]],
        "exact_matches": List[Dict[str, str]],
        "pages_with_matching_images": List[Dict[str, str]],
        "error": Optional[str]
      }
    """
    api_key = os.getenv("SERPAPI_KEY") or os.getenv("SERP_API_KEY") or getattr(settings, "serpapi_key", "")

    result: Dict[str, Any] = {
        "provider": "serpapi_google_lens",
        "visual_matches": [],
        "exact_matches": [],
        "pages_with_matching_images": [],
        "error": None
    }

    if not api_key:
        log.warning("serpapi_key_missing")
        result["error"] = "SerpAPI key missing"
        return result

    if not image_url and (not image_path or not os.path.exists(image_path)):
        result["error"] = f"Image path or URL required: {image_path}"
        return result

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            if image_url:
                params = {
                    "engine": "google_lens",
                    "url": image_url,
                    "api_key": api_key
                }
                response = await client.get(SERPAPI_ENDPOINT, params=params)
            else:
                params = {
                    "engine": "google_lens",
                    "api_key": api_key
                }
                with open(image_path, "rb") as f:
                    files = {"image_file": (os.path.basename(image_path), f, "image/jpeg")}
                    response = await client.post(SERPAPI_ENDPOINT, params=params, files=files)

        if response.status_code != 200:
            log.warning("serpapi_lens_http_error", status=response.status_code, body=response.text[:200])
            result["error"] = f"HTTP {response.status_code}: {response.text[:100]}"
            return result

        data = response.json()

        # 1. Parse visual matches
        for item in data.get("visual_matches", []):
            link = item.get("link", "") or item.get("source_url", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            source = item.get("source", "")
            if link:
                match_data = {
                    "url": link,
                    "title": title,
                    "snippet": snippet,
                    "source": source,
                    "date": item.get("date", "")
                }
                result["visual_matches"].append(match_data)
                if item.get("is_exact_match") or "exact" in str(item.get("match_type", "")).lower():
                    result["exact_matches"].append(match_data)

        # 2. Parse exact matches list if present
        for item in data.get("exact_matches", []):
            link = item.get("link", "")
            if link:
                result["exact_matches"].append({
                    "url": link,
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("source", ""),
                    "date": item.get("date", "")
                })

        # 3. Parse pages_with_matching_images
        for item in data.get("pages_with_matching_images", []):
            link = item.get("link", "")
            if link:
                result["pages_with_matching_images"].append({
                    "url": link,
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("source", ""),
                    "date": item.get("date", "")
                })

        log.info(
            "serpapi_google_lens_done",
            visual=len(result["visual_matches"]),
            exact=len(result["exact_matches"]),
            pages=len(result["pages_with_matching_images"])
        )

    except Exception as e:
        log.error("serpapi_google_lens_failed", error=str(e))
        result["error"] = str(e)

    return result
