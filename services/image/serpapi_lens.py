"""
services/image/serpapi_lens.py — Reverse search via SerpAPI's Google Lens engine.

The second, corroborating provider. Lens indexes social and regional sites that
Vision's Web Detection often misses, so a page found by both is meaningfully
stronger evidence than one found by either alone.

⚠️ PRIVACY CONSTRAINT — read before enabling.

SerpAPI's google_lens engine searches an image *by URL*: it fetches the picture
from a location it can reach on the public internet. A Telegram photo sitting in
`temp/` is not reachable, so using this provider means the user's private image
has to be published somewhere first.

This module will not do that silently. It runs only when the caller supplies a
URL, which requires `PUBLIC_IMAGE_BASE_URL` to be set — an explicit statement by
the operator that images served from there are already public. With no URL
configured the provider reports `available=False` with the reason, and Google
Vision (which takes raw bytes and publishes nothing) carries the search alone.

`SERPAPI_LENS_ALLOW_UPLOAD=true` additionally permits a direct file upload for
deployments whose SerpAPI plan accepts one. It is off by default for the same
reason: it sends the user's image to a third party.
"""
import os
import structlog
from typing import Any, Dict, List, Optional

import httpx

from src.config import is_real_key, settings
from services.image.match_ranker import (
    EXACT_MATCH, FULL_MATCH, HIGH_VISUAL_SIMILARITY, LOW_VISUAL_SIMILARITY,
    normalise_match,
)

log = structlog.get_logger(__name__)

SERPAPI_ENDPOINT = "https://serpapi.com/search"

PROVIDER = "serpapi_lens"

# Lens returns a long tail of loosely-similar images; only the head is useful.
MAX_VISUAL_MATCHES = 12
HIGH_SIMILARITY_CUTOFF = 5


def is_configured() -> bool:
    return is_real_key(settings.serpapi_key)


def public_url_for(image_path: str) -> Optional[str]:
    """
    Maps a local file to its public URL, if the operator has declared one.
    Returns None when no public base URL is configured — the normal case, and
    the reason this provider usually sits out.
    """
    base = (settings.public_image_base_url or "").strip().rstrip("/")
    if not base or not image_path:
        return None
    return f"{base}/{os.path.basename(image_path)}"


def _match_type_for_visual(index: int) -> str:
    """
    Lens `visual_matches` mixes genuine reposts with merely similar pictures and
    doesn't distinguish them, so rank order is all we have. Treated as
    similarity, never as a full match — over-claiming here would let a
    lookalike photo drive a "recycled" verdict.
    """
    return HIGH_VISUAL_SIMILARITY if index < HIGH_SIMILARITY_CUTOFF else LOW_VISUAL_SIMILARITY


def parse_lens_response(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turns a raw SerpAPI Lens payload into normalised matches."""
    matches: List[Dict[str, Any]] = []

    # search_type=exact_matches — the provider asserts this is the same image.
    for item in (data.get("exact_matches") or []):
        link = item.get("link") or ""
        if link:
            matches.append(normalise_match(
                url=link,
                match_type=EXACT_MATCH,
                source=PROVIDER,
                page_title=item.get("title") or "",
                image_url=item.get("thumbnail") or "",
                provider_date=item.get("date"),
            ))

    # Legacy/alternate field from the reverse-image engine.
    for item in (data.get("pages_with_matching_images") or []):
        link = item.get("link") or ""
        if link:
            matches.append(normalise_match(
                url=link,
                match_type=FULL_MATCH,
                source=PROVIDER,
                page_title=item.get("title") or "",
                provider_date=item.get("date"),
            ))

    for index, item in enumerate((data.get("visual_matches") or [])[:MAX_VISUAL_MATCHES]):
        link = item.get("link") or ""
        if link:
            matches.append(normalise_match(
                url=link,
                match_type=_match_type_for_visual(index),
                source=PROVIDER,
                page_title=item.get("title") or "",
                image_url=item.get("thumbnail") or item.get("image") or "",
                provider_date=item.get("date"),
            ))

    return matches


async def _call(params: Dict[str, Any], timeout: float = 20.0) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(SERPAPI_ENDPOINT, params=params)
    response.raise_for_status()
    return response.json()


SERPAPI_IMAGE_ENDPOINT = "https://serpapi.com/image"


async def _upload_image(image_path: str, timeout: float = 20.0) -> str | None:
    """
    Uploads an image to SerpAPI and returns the image_id.
    SerpAPI's file upload is a two-step process:
      1. POST the file to /image → get an image_id
      2. GET /search with engine=google_lens&image_id=...
    """
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                SERPAPI_IMAGE_ENDPOINT,
                data={"api_key": settings.serpapi_key},
                files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            )
        response.raise_for_status()
        data = response.json()
        image_id = data.get("image_id")
        if not image_id:
            log.warning("serpapi_upload_no_id", response=data)
            return None
        log.info("serpapi_image_uploaded", image_id=image_id)
        return image_id
    except Exception as e:
        log.warning("serpapi_upload_failed", error=str(e))
        return None


async def search(
    image_path: str,
    image_url: Optional[str] = None,
    timeout: float | None = None,
) -> Dict[str, Any]:
    """
    Runs Google Lens for the image.
    Returns {matches, available, error}. `available=False` means the provider
    never ran — usually because no public image URL exists, which is not a
    finding about the image.
    """
    unavailable: Dict[str, Any] = {"matches": [], "available": False}
    timeout = timeout or settings.reverse_search_timeout

    if not is_configured():
        return {**unavailable, "error": "SERPAPI_KEY is not configured."}

    url = image_url or public_url_for(image_path)

    if not url and not settings.serpapi_lens_allow_upload:
        return {
            **unavailable,
            "error": (
                "Google Lens needs a publicly reachable image URL. Set "
                "PUBLIC_IMAGE_BASE_URL (or SERPAPI_LENS_ALLOW_UPLOAD=true) to enable it. "
                "Both publish the user's image to a third party — Google Vision does not."
            ),
        }

    try:
        if url:
            data = await _call(
                {
                    "engine": "google_lens",
                    "url": url,
                    "api_key": settings.serpapi_key,
                    "hl": "en",
                    "country": "in",
                },
                timeout=timeout,
            )
        else:
            # Two-step upload: POST image to /image, then search by image_id.
            log.info("serpapi_lens_direct_upload", path=image_path)
            image_id = await _upload_image(image_path, timeout=timeout)
            if not image_id:
                return {**unavailable, "error": "Failed to upload image to SerpAPI."}
            data = await _call(
                {
                    "engine": "google_lens",
                    "image_id": image_id,
                    "api_key": settings.serpapi_key,
                    "hl": "en",
                    "country": "in",
                },
                timeout=timeout,
            )

        if data.get("error"):
            log.warning("serpapi_lens_error", error=data["error"])
            return {**unavailable, "error": str(data["error"])}

        matches = parse_lens_response(data)
        log.info("serpapi_lens_done", n_matches=len(matches))
        return {"matches": matches, "available": True, "error": None}

    except Exception as e:
        log.warning("serpapi_lens_failed", error=str(e))
        return {**unavailable, "error": str(e)}

