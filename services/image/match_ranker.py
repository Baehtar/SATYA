"""
services/image/match_ranker.py — Stage 5: normalise, deduplicate and rank matches.

Google Vision and SerpAPI Lens frequently return the same page. Counting it
twice would inflate the evidence, so matches are keyed by canonical URL and
merged: the strongest match type wins, and the providers that found it are
recorded, because a page found *independently by both* is stronger evidence
than a page only one provider indexed.

The weights below are ENGINEERING WEIGHTS, not empirical probabilities. They
encode one ordering — an exact byte-level match is worth far more than "looks
a bit similar" — and are meant to be tuned on the judging set, not cited as
measured accuracy.
"""
import re
import structlog
from typing import Any, Dict, List
from urllib.parse import urlparse, urlunparse

log = structlog.get_logger(__name__)

# ── Match types, strongest first ────────────────────────────────────────────
EXACT_MATCH = "EXACT_MATCH"                      # provider says byte/near-byte identical
FULL_MATCH = "FULL_MATCH"                        # whole image present, maybe re-encoded
PARTIAL_MATCH = "PARTIAL_MATCH"                  # crop of, or containing, the image
HIGH_VISUAL_SIMILARITY = "HIGH_VISUAL_SIMILARITY"
LOW_VISUAL_SIMILARITY = "LOW_VISUAL_SIMILARITY"

MATCH_WEIGHTS: Dict[str, float] = {
    EXACT_MATCH: 1.00,
    FULL_MATCH: 0.90,
    PARTIAL_MATCH: 0.75,
    HIGH_VISUAL_SIMILARITY: 0.40,
    LOW_VISUAL_SIMILARITY: 0.15,
}

MATCH_ORDER = [
    EXACT_MATCH, FULL_MATCH, PARTIAL_MATCH,
    HIGH_VISUAL_SIMILARITY, LOW_VISUAL_SIMILARITY,
]

# Only these carry enough weight to support a "this image existed earlier" claim.
STRONG_MATCH_TYPES = {EXACT_MATCH, FULL_MATCH, PARTIAL_MATCH}

# Tracking junk that changes per-visit and would defeat deduplication.
_TRACKING_PARAMS = re.compile(
    r"^(utm_|fbclid|gclid|igshid|mc_[ce]id|ref|ref_src|s|_ga|yclid|msclkid)", re.I
)

# Aggregators and social mirrors: fine as corroboration, but their timestamps
# describe when *the post* appeared, not when the photograph was taken.
LOW_AUTHORITY_DOMAINS = {
    "pinterest.com", "facebook.com", "instagram.com", "x.com", "twitter.com",
    "reddit.com", "tumblr.com", "quora.com", "9gag.com", "imgur.com",
    "shutterstock.com", "alamy.com", "dreamstime.com", "istockphoto.com",
}


def canonical_url(url: str) -> str:
    """
    Collapses the variants of one page into a single key: scheme normalised to
    https, lower-cased host, no `www.`, no tracking query params, no fragment,
    no trailing slash.

    The scheme is normalised because http:// and https:// URLs for the same
    article are the same article — Vision and Lens routinely disagree on which
    one to return, and counting both would double the apparent evidence.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]

        query = "&".join(
            part for part in (parsed.query or "").split("&")
            if part and not _TRACKING_PARAMS.match(part.split("=", 1)[0])
        )
        path = (parsed.path or "").rstrip("/") or "/"
        return urlunparse(("https", host, path, "", query, ""))
    except Exception:
        return url.strip()


def domain_of(url: str) -> str:
    try:
        host = (urlparse(url).netloc or "").lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def is_low_authority(url: str) -> bool:
    domain = domain_of(url)
    return any(domain == d or domain.endswith("." + d) for d in LOW_AUTHORITY_DOMAINS)


def _strength(match_type: str) -> int:
    """Lower index = stronger."""
    try:
        return MATCH_ORDER.index(match_type)
    except ValueError:
        return len(MATCH_ORDER)


def normalise_match(
    url: str,
    match_type: str,
    source: str,
    page_title: str = "",
    similarity: float | None = None,
    image_url: str = "",
    provider_date: str | None = None,
) -> Dict[str, Any]:
    """One match in the shape the rest of the engine expects."""
    if match_type not in MATCH_WEIGHTS:
        match_type = LOW_VISUAL_SIMILARITY
    return {
        "url": url,
        "canonical_url": canonical_url(url),
        "match_type": match_type,
        "similarity": round(
            MATCH_WEIGHTS[match_type] if similarity is None else float(similarity), 3
        ),
        "source": source,
        "sources": [source],
        "page_title": page_title or "",
        "image_url": image_url or "",
        "domain": domain_of(url),
        "low_authority": is_low_authority(url),
        # Some providers attach a date to the result itself. It's a weak hint —
        # the date extractor prefers what the page actually says.
        "provider_date": provider_date,
        "published_date": None,
        "date_confidence": 0.0,
        "date_method": None,
    }


def rank_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates by canonical URL and sorts strongest-first.

    Merge rules when two providers return the same page:
      * keep the strongest match type either one reported
      * union the provider list — corroboration is recorded, not double-counted
      * keep the highest similarity
    """
    merged: Dict[str, Dict[str, Any]] = {}

    for match in matches:
        key = match.get("canonical_url") or match.get("url") or ""
        if not key:
            continue

        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(match)
            continue

        if _strength(match["match_type"]) < _strength(existing["match_type"]):
            existing["match_type"] = match["match_type"]
        existing["similarity"] = max(existing["similarity"], match["similarity"])
        for provider in match.get("sources", []):
            if provider not in existing["sources"]:
                existing["sources"].append(provider)
        if not existing.get("page_title") and match.get("page_title"):
            existing["page_title"] = match["page_title"]
        if not existing.get("image_url") and match.get("image_url"):
            existing["image_url"] = match["image_url"]

    ranked = sorted(
        merged.values(),
        key=lambda m: (
            _strength(m["match_type"]),          # strongest match type first
            -len(m.get("sources", [])),          # then corroborated by both
            m.get("low_authority", False),       # then real publishers
            -m.get("similarity", 0.0),
        ),
    )

    log.info("matches_ranked", n_raw=len(matches), n_unique=len(ranked))
    return ranked


def has_strong_match(matches: List[Dict[str, Any]]) -> bool:
    return any(m.get("match_type") in STRONG_MATCH_TYPES for m in matches)
