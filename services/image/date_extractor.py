"""
services/image/date_extractor.py — Multi-tier web page publication date extraction engine.
Parses JSON-LD structured data, HTML OpenGraph/meta tags, <time> elements, visible text, and URL paths.
"""
import json
import re
import httpx
import structlog
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

log = structlog.get_logger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 SatyaFactChecker/1.0"


def parse_date_string(date_str: str) -> Optional[datetime]:
    """Tries multiple date formats to return a timezone-aware UTC datetime."""
    if not date_str or not isinstance(date_str, str):
        return None

    cleaned = date_str.strip()

    # ISO 8601 parsing (e.g., 2024-02-16T14:30:00Z)
    if "T" in cleaned:
        try:
            # Strip sub-second precision or timezone strings for simpler parsing
            iso_clean = re.sub(r'(\.\d+)?([+-]\d{2}:\d{2}|Z)?$', '', cleaned)
            return datetime.strptime(iso_clean, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    # YYYY-MM-DD
    match_iso = re.search(r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b', cleaned)
    if match_iso:
        y, m, d = int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3))
        if 1990 <= y <= 2030 and 1 <= m <= 12 and 1 <= d <= 31:
            return datetime(y, m, d, tzinfo=timezone.utc)

    # Standard textual formats (e.g. 16 August 2018, Aug 16, 2018)
    formats = [
        "%d %B %Y", "%d %b %Y",
        "%B %d, %Y", "%b %d, %Y",
        "%d/%m/%Y", "%m/%d/%Y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return None


def extract_date_from_json_ld(soup: BeautifulSoup) -> Optional[Tuple[str, float]]:
    """Priority 1: Extract date from JSON-LD scripts."""
    scripts = soup.find_all("script", type="application/ld+json")
    for s in scripts:
        if not s.string:
            continue
        try:
            data = json.loads(s.string)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                # Check for NewsArticle, Article, BlogPosting, WebPage
                date_val = (
                    item.get("datePublished")
                    or item.get("dateCreated")
                    or item.get("dateModified")
                )
                if date_val and isinstance(date_val, str):
                    dt = parse_date_string(date_val)
                    if dt:
                        return dt.strftime("%Y-%m-%d"), 0.95
        except Exception:
            continue
    return None


def extract_date_from_meta_tags(soup: BeautifulSoup) -> Optional[Tuple[str, float]]:
    """Priority 2: Extract date from HTML meta tags."""
    meta_names = [
        ("property", "article:published_time"),
        ("property", "article:modified_time"),
        ("property", "og:updated_time"),
        ("name", "date"),
        ("name", "publish-date"),
        ("name", "pubdate"),
        ("name", "DC.date.issued"),
        ("name", "parsely-pub-date"),
        ("name", "sailthru.date")
    ]
    for attr_name, attr_val in meta_names:
        tag = soup.find("meta", {attr_name: attr_val})
        if tag and tag.get("content"):
            dt = parse_date_string(str(tag["content"]))
            if dt:
                return dt.strftime("%Y-%m-%d"), 0.90
    return None


def extract_date_from_time_tags(soup: BeautifulSoup) -> Optional[Tuple[str, float]]:
    """Priority 3: Extract date from <time> tags."""
    time_tags = soup.find_all("time")
    for t in time_tags:
        val = t.get("datetime") or t.text
        if val:
            dt = parse_date_string(str(val))
            if dt:
                return dt.strftime("%Y-%m-%d"), 0.80
    return None


def extract_date_from_visible_text(text: str) -> Optional[Tuple[str, float]]:
    """Priority 4: Regex extract date from visible page text."""
    date_patterns = [
        r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b',
        r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b',
        r'\b(\d{4}-\d{2}-\d{2})\b',
        r'\b(\d{1,2}/\d{1,2}/\d{4})\b'
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            dt = parse_date_string(match.group(1))
            if dt:
                return dt.strftime("%Y-%m-%d"), 0.65
    return None


def extract_date_from_url(url: str) -> Optional[Tuple[str, float]]:
    """Priority 5: Extract date from URL path patterns (e.g. /2024/02/16/)."""
    match = re.search(r'/(20\d{2})/(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/', url)
    if match:
        date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        return date_str, 0.50

    match_dash = re.search(r'/(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])/', url)
    if match_dash:
        date_str = f"{match_dash.group(1)}-{match_dash.group(2)}-{match_dash.group(3)}"
        return date_str, 0.50

    return None


async def extract_page_date(url: str, client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
    """
    Fetches web page HTML and extracts publication date using multi-tier fallback.
    
    Returns:
      {
        "url": url,
        "published_date": str | None, (e.g. "2018-08-16")
        "date_confidence": float, (0.0 to 1.0)
        "extraction_method": str, (json_ld, meta_tag, time_tag, visible_text, url_pattern)
        "page_title": str
      }
    """
    res = {
        "url": url,
        "published_date": None,
        "date_confidence": 0.0,
        "extraction_method": "none",
        "page_title": ""
    }

    if not url or not url.startswith("http"):
        return res

    # First check URL pattern (fast check)
    url_date = extract_date_from_url(url)

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers={"User-Agent": USER_AGENT})
        close_client = True

    try:
        response = await client.get(url)
        if response.status_code == 200:
            html = response.text
            soup = BeautifulSoup(html, "lxml")

            res["page_title"] = soup.title.string.strip() if soup.title and soup.title.string else ""

            # Priority 1: JSON-LD
            p1 = extract_date_from_json_ld(soup)
            if p1:
                res["published_date"], res["date_confidence"] = p1
                res["extraction_method"] = "json_ld"
                return res

            # Priority 2: Meta tags
            p2 = extract_date_from_meta_tags(soup)
            if p2:
                res["published_date"], res["date_confidence"] = p2
                res["extraction_method"] = "meta_tag"
                return res

            # Priority 3: Time tags
            p3 = extract_date_from_time_tags(soup)
            if p3:
                res["published_date"], res["date_confidence"] = p3
                res["extraction_method"] = "time_tag"
                return res

            # Priority 4: Visible text
            p4 = extract_date_from_visible_text(soup.get_text()[:3000])
            if p4:
                res["published_date"], res["date_confidence"] = p4
                res["extraction_method"] = "visible_text"
                return res

    except Exception as e:
        log.warning("page_date_extraction_http_error", url=url, error=str(e))
    finally:
        if close_client:
            await client.aclose()

    # Priority 5: Fall back to URL pattern if page fetch failed or produced no date
    if url_date:
        res["published_date"], res["date_confidence"] = url_date
        res["extraction_method"] = "url_pattern"

    return res
