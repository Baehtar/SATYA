"""
services/image/date_extractor.py — Stage 6: when did this page publish?

Reverse-search providers return URLs, not trustworthy dates, so every candidate
page is fetched and read directly. Sources are tried in descending order of how
hard they are to get wrong, and each carries its own confidence:

  1. JSON-LD  datePublished           0.95  machine-written, schema.org typed
  2. <meta>   article:published_time  0.90  the Open Graph / CMS timestamp
  3. <time datetime="...">            0.75  semantic, but often "updated"
  4. Visible text near "Published:"   0.55  human-written, locale-dependent
  5. /2019/03/14/ in the URL          0.45  a path convention, not a statement

Every candidate found is kept, not just the winner, so a page whose JSON-LD
says 2024 while its URL says 2019 is *reported as conflicting* rather than
silently resolved. That disagreement is itself useful evidence.

Two guardrails matter more than the parsing:
  * Nothing in the future and nothing before 1990 is accepted — bad parses
    reliably produce absurd years, and one of those becoming the "earliest
    appearance" would fabricate a recycled-image verdict.
  * dateModified is never used as a publication date. A 2019 article re-touched
    last week would otherwise look brand new.
"""
import asyncio
import json
import re
import structlog
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
from dateutil import parser as date_parser

from src.config import settings

log = structlog.get_logger(__name__)

# ── Confidence per extraction method ────────────────────────────────────────
METHOD_JSONLD = "jsonld"
METHOD_META = "meta"
METHOD_TIME_TAG = "time_tag"
METHOD_VISIBLE_TEXT = "visible_text"
METHOD_URL = "url_pattern"
METHOD_PROVIDER = "search_provider"

METHOD_CONFIDENCE = {
    METHOD_JSONLD: 0.95,
    METHOD_META: 0.90,
    METHOD_TIME_TAG: 0.75,
    METHOD_VISIBLE_TEXT: 0.55,
    METHOD_URL: 0.45,
    METHOD_PROVIDER: 0.40,
}

# Anything outside this window is a parsing failure, not a publication date.
EARLIEST_PLAUSIBLE = datetime(1990, 1, 1, tzinfo=timezone.utc)
FUTURE_TOLERANCE = timedelta(days=2)  # allows for timezone skew on fresh posts

# Meta tags that carry a publication date, best first.
META_DATE_ATTRS = [
    "article:published_time", "og:published_time", "article:published",
    "datePublished", "publish-date", "publication_date", "pubdate",
    "parsely-pub-date", "sailthru.date", "DC.date.issued", "dc.date.issued",
    "date", "cXenseParse:recs:publishtime", "timestamp",
]

# JSON-LD keys, in order. dateModified is deliberately absent.
JSONLD_DATE_KEYS = ["datePublished", "dateCreated", "uploadDate"]

# A leading 4-digit component means the string is year-first (ISO-8601 and
# friends), so day/month must not be swapped by dayfirst parsing.
_YEAR_FIRST = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")

_URL_DATE_PATTERNS = [
    re.compile(r"/(20\d{2})[/\-](\d{1,2})[/\-](\d{1,2})(?:/|$|[\-_.])"),
    re.compile(r"/(20\d{2})[/\-](\d{1,2})(?:/|$)"),
    re.compile(r"[\-_](20\d{2})(\d{2})(\d{2})[\-_.]"),
]

_MONTHS = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)

_TEXT_DATE_PATTERNS = [
    re.compile(rf"\b(\d{{1,2}}\s+(?:{_MONTHS})\.?,?\s+\d{{4}})\b", re.I),
    re.compile(rf"\b((?:{_MONTHS})\.?\s+\d{{1,2}},?\s+\d{{4}})\b", re.I),
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
    re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b"),
]

# Words that mark a nearby date as the publication date rather than, say, the
# date of the event being reported.
_PUBLICATION_CUES = re.compile(
    r"(published|posted|updated|last\s+modified|dated|प्रकाशित|வெளியிடப்பட்ட)", re.I
)


def parse_date(value: Any, dayfirst: bool = True) -> Optional[datetime]:
    """
    Tolerant parse to an aware UTC datetime, or None.

    `dayfirst=True` because 03/04/2024 is 3 April in Indian sources — but it
    must NOT be applied to year-first strings. Every machine-readable date we
    read (JSON-LD, article:published_time, <time datetime>) is ISO-8601, and
    dayfirst would silently transpose 2019-05-02 into 5 February 2019.

    Rejects implausible results rather than passing them on.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        # A 4-digit leading component is unambiguously a year.
        if _YEAR_FIRST.match(text):
            dayfirst = False
        try:
            parsed = date_parser.parse(text, dayfirst=dayfirst, fuzzy=False)
        except (ValueError, OverflowError, TypeError):
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)

    if parsed < EARLIEST_PLAUSIBLE:
        return None
    if parsed > datetime.now(timezone.utc) + FUTURE_TOLERANCE:
        return None
    return parsed


def _candidate(value: Any, method: str, dayfirst: bool = True) -> Optional[Dict[str, Any]]:
    parsed = parse_date(value, dayfirst=dayfirst)
    if not parsed:
        return None
    return {
        "date": parsed.date().isoformat(),
        "datetime": parsed.isoformat(),
        "confidence": METHOD_CONFIDENCE.get(method, 0.3),
        "method": method,
        "raw": str(value)[:100],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Priority 1 — JSON-LD
# ─────────────────────────────────────────────────────────────────────────────

def _walk_jsonld(node: Any, found: List[Dict[str, Any]]) -> None:
    """JSON-LD nests arbitrarily (@graph, arrays, nested objects) — walk it all."""
    if isinstance(node, dict):
        for key in JSONLD_DATE_KEYS:
            if key in node:
                candidate = _candidate(node[key], METHOD_JSONLD)
                if candidate:
                    candidate["raw"] = f"{key}={candidate['raw']}"
                    found.append(candidate)
        for value in node.values():
            _walk_jsonld(value, found)
    elif isinstance(node, list):
        for item in node:
            _walk_jsonld(item, found)


def from_jsonld(soup) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text() or ""
        if not raw.strip():
            continue
        try:
            _walk_jsonld(json.loads(raw), found)
        except (json.JSONDecodeError, ValueError):
            continue
    return found


# ─────────────────────────────────────────────────────────────────────────────
#  Priorities 2–4 — meta tags, <time>, visible text
# ─────────────────────────────────────────────────────────────────────────────

def from_meta(soup) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    for attr in META_DATE_ATTRS:
        for key in ("property", "name", "itemprop"):
            tag = soup.find("meta", attrs={key: re.compile(rf"^{re.escape(attr)}$", re.I)})
            if tag and tag.get("content"):
                candidate = _candidate(tag["content"], METHOD_META)
                if candidate:
                    candidate["raw"] = f"{attr}={candidate['raw']}"
                    found.append(candidate)
                break
    return found


def from_time_tags(soup) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    for tag in soup.find_all("time")[:10]:
        value = tag.get("datetime") or tag.get_text(strip=True)
        candidate = _candidate(value, METHOD_TIME_TAG)
        if candidate:
            found.append(candidate)
    return found


def from_visible_text(soup) -> List[Dict[str, Any]]:
    """
    Last resort. Only looks at text near a publication cue, because an article
    body is full of dates that belong to the *events* being described.
    """
    found: List[Dict[str, Any]] = []
    text = soup.get_text(" ", strip=True)[:6000]

    for cue in _PUBLICATION_CUES.finditer(text):
        window = text[cue.start(): cue.start() + 120]
        for pattern in _TEXT_DATE_PATTERNS:
            match = pattern.search(window)
            if match:
                candidate = _candidate(match.group(1), METHOD_VISIBLE_TEXT)
                if candidate:
                    found.append(candidate)
                    break
        if found:
            break
    return found


def from_url(url: str) -> List[Dict[str, Any]]:
    """Dates baked into the path, e.g. /2018/08/16/kerala-floods."""
    for pattern in _URL_DATE_PATTERNS:
        match = pattern.search(url or "")
        if not match:
            continue
        groups = match.groups()
        try:
            year = int(groups[0])
            month = int(groups[1]) if len(groups) > 1 else 1
            day = int(groups[2]) if len(groups) > 2 else 1
            if not (1 <= month <= 12 and 1 <= day <= 31):
                continue
            candidate = _candidate(datetime(year, month, day, tzinfo=timezone.utc), METHOD_URL)
            if candidate:
                candidate["raw"] = match.group(0)
                return [candidate]
        except (ValueError, TypeError):
            continue
    return []


# ─────────────────────────────────────────────────────────────────────────────
#  Combining
# ─────────────────────────────────────────────────────────────────────────────

def extract_dates_from_html(html: str, url: str = "") -> Dict[str, Any]:
    """
    Runs every extractor and picks a winner.
    Returns {date, datetime, confidence, method, candidates, conflicting}.

    `conflicting` is True when two candidates that are each reasonably trusted
    disagree by more than a month — reported, not silently resolved.
    """
    empty = {
        "date": None, "datetime": None, "confidence": 0.0,
        "method": None, "candidates": [], "conflicting": False,
    }
    if not html:
        return empty

    try:
        from bs4 import BeautifulSoup
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        log.warning("html_parse_failed", error=str(e))
        return empty

    candidates: List[Dict[str, Any]] = []
    candidates.extend(from_jsonld(soup))
    candidates.extend(from_meta(soup))
    candidates.extend(from_time_tags(soup))
    candidates.extend(from_visible_text(soup))
    candidates.extend(from_url(url))

    if not candidates:
        return empty

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    best = candidates[0]

    # Disagreement among trusted candidates is worth surfacing.
    conflicting = False
    trusted = [c for c in candidates if c["confidence"] >= 0.70]
    if len(trusted) > 1:
        dates = [datetime.fromisoformat(c["datetime"]) for c in trusted]
        if (max(dates) - min(dates)) > timedelta(days=31):
            conflicting = True
            # Prefer the earliest trusted date: a page updated later is still a
            # page that existed earlier, and provenance is about first appearance.
            best = min(trusted, key=lambda c: c["datetime"])

    return {
        "date": best["date"],
        "datetime": best["datetime"],
        "confidence": round(best["confidence"] * (0.8 if conflicting else 1.0), 3),
        "method": best["method"],
        "candidates": candidates[:6],
        "conflicting": conflicting,
    }


async def fetch_page_date(
    url: str, client: httpx.AsyncClient, provider_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetches one page and extracts its publication date.
    Falls back to the search provider's own date when the page can't be read —
    paywalls, 403s and JS-only sites are routine.
    """
    result: Dict[str, Any] = {
        "date": None, "datetime": None, "confidence": 0.0,
        "method": None, "candidates": [], "conflicting": False,
        "title": "", "fetch_error": None,
    }

    try:
        response = await client.get(url, follow_redirects=True)
        if response.status_code != 200:
            result["fetch_error"] = f"HTTP {response.status_code}"
        else:
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type.lower():
                result["fetch_error"] = f"Not HTML ({content_type[:40]})"
            else:
                html = response.text
                result.update(extract_dates_from_html(html, url))
                match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
                if match:
                    result["title"] = re.sub(r"\s+", " ", match.group(1)).strip()[:200]
    except Exception as e:
        result["fetch_error"] = str(e)[:120]

    if not result["date"]:
        # The URL path alone still beats nothing, and works when the fetch failed.
        url_candidates = from_url(url)
        if url_candidates:
            result.update({k: url_candidates[0][k] for k in ("date", "datetime", "confidence", "method")})
            result["candidates"] = url_candidates
        elif provider_date:
            candidate = _candidate(provider_date, METHOD_PROVIDER)
            if candidate:
                result.update({k: candidate[k] for k in ("date", "datetime", "confidence", "method")})
                result["candidates"] = [candidate]

    return result


async def enrich_matches_with_dates(
    matches: List[Dict[str, Any]], limit: int | None = None, timeout: float | None = None
) -> List[Dict[str, Any]]:
    """
    Fetches the top matches concurrently and attaches published_date /
    date_confidence / date_method / page_title to each.
    Only the top `limit` are fetched — the tail is long and mostly duplicates.
    """
    limit = limit or settings.date_extraction_max_pages
    timeout = timeout or settings.page_fetch_timeout
    to_fetch = [m for m in matches if m.get("url")][:limit]
    if not to_fetch:
        return matches

    headers = {
        # Some publishers 403 an unidentified client outright.
        "User-Agent": (
            "Mozilla/5.0 (compatible; SatyaFactCheck/1.0; +https://github.com/Chetanchaudhary08)"
        ),
        "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8,ta;q=0.7",
    }

    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        results = await asyncio.gather(
            *(fetch_page_date(m["url"], client, m.get("provider_date")) for m in to_fetch),
            return_exceptions=True,
        )

    for match, result in zip(to_fetch, results):
        if isinstance(result, Exception):
            match["date_error"] = str(result)[:120]
            continue
        match["published_date"] = result["date"]
        match["date_confidence"] = result["confidence"]
        match["date_method"] = result["method"]
        match["date_conflicting"] = result["conflicting"]
        match["date_candidates"] = result["candidates"]
        if result.get("title") and not match.get("page_title"):
            match["page_title"] = result["title"]
        if result.get("fetch_error"):
            match["date_error"] = result["fetch_error"]

    log.info(
        "dates_extracted",
        n_fetched=len(to_fetch),
        n_dated=sum(1 for m in to_fetch if m.get("published_date")),
    )
    return matches


# ─────────────────────────────────────────────────────────────────────────────
#  Claim-side dates
# ─────────────────────────────────────────────────────────────────────────────

_RELATIVE_TERMS = {
    "today": 0, "tonight": 0, "this morning": 0, "just now": 0, "right now": 0,
    "aaj": 0, "abhi": 0, "इंदु": 0, "आज": 0, "இன்று": 0,
    "yesterday": 1, "last night": 1, "kal": 1, "कल": 1, "நேற்று": 1,
    "this week": 3, "is hafte": 3, "last week": 7, "pichle hafte": 7,
}


def extract_claim_date(text: str, now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Finds the date the claim asserts the image is from.

    "Yesterday's flood in Bihar" is the whole reason reverse search works: it
    pins the claim to a date the image's provenance can contradict. Explicit
    dates win over relative words; if neither is present the caller compares
    against "now" instead, with lower confidence.

    Returns {date, confidence, method, raw}.
    """
    result: Dict[str, Any] = {"date": None, "confidence": 0.0, "method": None, "raw": None}
    if not text:
        return result

    now = now or datetime.now(timezone.utc)
    lowered = text.lower()

    # Explicit date first — more specific than any relative word in the text.
    for pattern in _TEXT_DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            parsed = parse_date(match.group(1))
            if parsed:
                return {
                    "date": parsed.date().isoformat(),
                    "confidence": 0.85,
                    "method": "explicit_date",
                    "raw": match.group(1),
                }

    for term, days_ago in _RELATIVE_TERMS.items():
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered):
            resolved = now - timedelta(days=days_ago)
            return {
                "date": resolved.date().isoformat(),
                # "kal" means both yesterday and tomorrow in Hindi, and "today"
                # depends on when the message was sent, not when we read it.
                "confidence": 0.60 if term in ("kal", "कल") else 0.70,
                "method": "relative_term",
                "raw": term,
            }

    return result
