"""
services/image/google_vision.py — Reverse search via Google Cloud Vision
Web Detection.

This is the primary provider because it takes the image bytes directly: a
Telegram photo goes straight from `temp/<id>.jpg` to the API as base64, with no
need to publish it to a public URL first. That matters for privacy as much as
for plumbing — the user's image never gets hosted anywhere.

Called over plain REST with an API key rather than through the
`google-cloud-vision` SDK. httpx is already a dependency and the SDK would pull
in grpc plus the whole google-api-core stack for one endpoint.

Web Detection returns five fields, which map onto our match vocabulary:
    fullMatchingImages       → FULL_MATCH   (the same image, possibly re-encoded)
    partialMatchingImages    → PARTIAL_MATCH (a crop of it, or it inside a collage)
    pagesWithMatchingImages  → the pages carrying either of the above
    visuallySimilarImages    → HIGH/LOW_VISUAL_SIMILARITY (a different photo!)
    webEntities              → topic labels, used for context, never as matches
"""
import base64
import structlog
from typing import Any, Dict, List

import httpx

from src.config import settings
from services.image.match_ranker import (
    FULL_MATCH, HIGH_VISUAL_SIMILARITY, LOW_VISUAL_SIMILARITY,
    PARTIAL_MATCH, normalise_match,
)

log = structlog.get_logger(__name__)

VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

PROVIDER = "google_vision"

# Vision hands back a lot of weak "visually similar" noise; keep the head of it.
MAX_VISUALLY_SIMILAR = 8
MAX_RESULTS_PER_FIELD = 20


def is_configured() -> bool:
    return bool(settings.google_vision_api_key)


def _pages(web: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Pages carrying the image. A page is classified by what it actually holds:
    a full match beats a partial one, and a page listing neither is still a
    match — Vision only puts it here when it found the image on it.
    """
    out = []
    for page in (web.get("pagesWithMatchingImages") or [])[:MAX_RESULTS_PER_FIELD]:
        url = page.get("url") or ""
        if not url:
            continue
        if page.get("fullMatchingImages"):
            match_type = FULL_MATCH
        elif page.get("partialMatchingImages"):
            match_type = PARTIAL_MATCH
        else:
            match_type = FULL_MATCH
        out.append(normalise_match(
            url=url,
            match_type=match_type,
            source=PROVIDER,
            page_title=page.get("pageTitle") or "",
            image_url=(page.get("fullMatchingImages") or [{}])[0].get("url", ""),
        ))
    return out


def _bare_images(web: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Direct image hits with no containing page. Their URL is the image file, so
    they rarely yield a publication date, but they still prove the image is
    indexed somewhere and can carry a usable /2019/03/ path.
    """
    out = []
    for field, match_type in (
        ("fullMatchingImages", FULL_MATCH),
        ("partialMatchingImages", PARTIAL_MATCH),
    ):
        for item in (web.get(field) or [])[:MAX_RESULTS_PER_FIELD]:
            url = item.get("url") or ""
            if url:
                out.append(normalise_match(
                    url=url, match_type=match_type, source=PROVIDER, image_url=url,
                ))
    return out


def _visually_similar(web: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    NOT the same image — a different photograph that looks alike. Weighted low
    on purpose: a similar-looking flood photo from 2018 says nothing about the
    image in hand.
    """
    out = []
    items = (web.get("visuallySimilarImages") or [])[:MAX_VISUALLY_SIMILAR]
    for index, item in enumerate(items):
        url = item.get("url") or ""
        if not url:
            continue
        # Vision returns these roughly best-first and gives no score.
        match_type = HIGH_VISUAL_SIMILARITY if index < 3 else LOW_VISUAL_SIMILARITY
        out.append(normalise_match(
            url=url, match_type=match_type, source=PROVIDER, image_url=url,
        ))
    return out


def parse_web_detection(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Turns a raw Vision response into {matches, entities, best_guess_labels}."""
    responses = payload.get("responses") or [{}]
    first = responses[0] if responses else {}

    if first.get("error"):
        message = first["error"].get("message", "Vision returned an error")
        return {"matches": [], "entities": [], "best_guess_labels": [], "error": message}

    web = first.get("webDetection") or {}

    entities = [
        {"description": e.get("description", ""), "score": float(e.get("score") or 0.0)}
        for e in (web.get("webEntities") or [])
        if e.get("description")
    ]
    labels = [
        label.get("label", "") for label in (web.get("bestGuessLabels") or [])
        if label.get("label")
    ]

    matches = _pages(web) + _bare_images(web) + _visually_similar(web)
    return {
        "matches": matches,
        "entities": entities[:10],
        "best_guess_labels": labels,
        "error": None,
    }


async def search(image_path: str, timeout: float | None = None) -> Dict[str, Any]:
    """
    Runs Web Detection on a local file.
    Returns {matches, entities, best_guess_labels, available, error}.
    `available=False` means the provider could not run at all — which is very
    different from "ran and found nothing", and the caller must not conflate them.
    """
    unavailable = {
        "matches": [], "entities": [], "best_guess_labels": [], "available": False,
    }

    if not is_configured():
        return {**unavailable, "error": "GOOGLE_VISION_API_KEY is not configured."}

    try:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        return {**unavailable, "error": f"Could not read image: {e}"}

    body = {
        "requests": [{
            "image": {"content": encoded},
            "features": [{"type": "WEB_DETECTION", "maxResults": MAX_RESULTS_PER_FIELD}],
            "imageContext": {"webDetectionParams": {"includeGeoResults": False}},
        }]
    }

    try:
        async with httpx.AsyncClient(timeout=timeout or settings.reverse_search_timeout) as client:
            response = await client.post(
                VISION_ENDPOINT,
                params={"key": settings.google_vision_api_key},
                json=body,
            )

        if response.status_code != 200:
            detail = response.text[:200]
            log.warning("vision_http_error", status=response.status_code, detail=detail)
            return {**unavailable, "error": f"Vision API returned {response.status_code}: {detail}"}

        parsed = parse_web_detection(response.json())
        if parsed.get("error"):
            return {**unavailable, "error": parsed["error"]}

        log.info(
            "vision_web_detection_done",
            n_matches=len(parsed["matches"]),
            labels=parsed["best_guess_labels"][:2],
        )
        return {**parsed, "available": True}

    except Exception as e:
        log.warning("vision_search_failed", error=str(e))
        return {**unavailable, "error": str(e)}
