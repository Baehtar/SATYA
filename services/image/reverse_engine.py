"""
services/image/reverse_engine.py — the Reverse Image Engine.

One entry point:

    result = await reverse_image_check(image_path, claim_date=..., claim_text=...)

It answers a question the claim-verification pipeline cannot: *where has this
photograph been before?* That is provenance, and it is separate from whether
the accompanying news claim is true. A genuine, unedited press photo from 2018
attached to "flood in Bihar today" makes the message false without making the
image fake. Those two findings stay separate all the way to the verdict card.

Stages (see the module docstrings for the details of each):
    1. metadata.fingerprint_image   SHA-256 + pHash + EXIF of the untouched file
    2. image_forensics.run_forensics ELA, noise, copy-move, resampling, JPEG
    3. google_vision.search ‖ serpapi_lens.search
    4. match_ranker.rank_matches     dedupe across providers, rank by strength
    5. date_extractor                fetch the top pages, extract publish dates
    6. date comparison               earliest located appearance vs claim date

Three honesty rules are enforced here rather than left to the caller:

  * "No match located" is never reported as "original". An absent result means
    the providers didn't find it — nothing more. `searched` records whether any
    provider even ran, so an unconfigured key can't masquerade as a clean check.
  * "Earliest located appearance", never "original date". We only ever see the
    earliest copy our providers indexed.
  * Only strong match types (exact/full/partial) can drive a recycled verdict.
    A visually similar photo from 2018 is not this photo from 2018.
"""
import asyncio
import os
import structlog
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.config import settings
from services.image import google_vision, serpapi_lens
from services.image.date_extractor import (
    enrich_matches_with_dates, extract_claim_date, parse_date,
)
from services.image.image_forensics import run_forensics
from services.image.match_ranker import (
    MATCH_WEIGHTS, STRONG_MATCH_TYPES, rank_matches,
)
from services.image.metadata import fingerprint_image

log = structlog.get_logger(__name__)

# ── image_status values ─────────────────────────────────────────────────────
STATUS_RECYCLED = "RECYCLED"                    # strong match predates the claim
STATUS_CONTEMPORANEOUS = "CONTEMPORANEOUS"      # matches line up with the claim
STATUS_PREVIOUSLY_PUBLISHED = "PREVIOUSLY_PUBLISHED"  # seen before, no claim date to compare
STATUS_SIMILAR_ONLY = "SIMILAR_ONLY"            # lookalikes only, not this image
STATUS_NO_MATCHES = "NO_MATCHES_LOCATED"        # searched, found nothing
STATUS_UNAVAILABLE = "SEARCH_UNAVAILABLE"       # no provider could run

# A gap smaller than this is ordinary news-cycle lag, not recycling.
RECYCLED_MIN_GAP_DAYS = 30


def _empty_result(error: Optional[str] = None) -> Dict[str, Any]:
    return {
        "image_hash": "", "phash": "", "ai_generated_score": None,
        "metadata": {"exif_present": False}, "forensics": {},
        "reverse_matches": [], "earliest_located_date": None,
        "earliest_located_match": None, "image_status": STATUS_UNAVAILABLE,
        "date_analysis": {}, "providers": {}, "searched": False,
        "notes": [], "error": error, "latency_ms": 0,
    }


def _earliest_strong_match(matches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    The earliest dated appearance among matches strong enough to *be* this image.
    Low-authority mirrors are allowed, but only after real publishers: a
    Pinterest repost dated 2016 is a weaker anchor than an NDTV article.
    """
    dated = [
        m for m in matches
        if m.get("match_type") in STRONG_MATCH_TYPES
        and m.get("published_date")
        and m.get("date_confidence", 0.0) >= 0.40
    ]
    if not dated:
        return None
    return min(
        dated,
        key=lambda m: (m["published_date"], m.get("low_authority", False)),
    )


def _recycled_confidence(match: Dict[str, Any], gap_days: int, claim_confidence: float) -> float:
    """
    How much to trust a recycled call. Three independent things must hold, so
    they multiply rather than add — a weak link in any one caps the result.

      match strength — is this really the same image?
      date strength  — do we trust the page's publication date?
      gap size       — 8 years is decisive, 6 weeks is not
    """
    match_weight = MATCH_WEIGHTS.get(match.get("match_type", ""), 0.15)
    date_weight = float(match.get("date_confidence") or 0.0)
    gap_weight = min(1.0, gap_days / 365.0)
    authority = 0.85 if match.get("low_authority") else 1.0
    corroborated = 1.0 + (0.1 if len(match.get("sources", [])) > 1 else 0.0)

    score = match_weight * date_weight * gap_weight * authority * corroborated
    score *= max(0.5, claim_confidence)  # an assumed claim date can't yield certainty
    return round(min(0.97, score), 3)


def _compare_dates(
    matches: List[Dict[str, Any]],
    claim_date: Optional[str],
    claim_confidence: float,
    exif_date: Optional[str],
    searched: bool,
) -> Dict[str, Any]:
    """Stage 6 — turns the dated matches into a provenance status."""
    earliest = _earliest_strong_match(matches)
    has_any_match = bool(matches)
    has_strong = any(m.get("match_type") in STRONG_MATCH_TYPES for m in matches)

    analysis: Dict[str, Any] = {
        "claim_date": claim_date,
        "claim_date_confidence": round(claim_confidence, 2),
        "earliest_located_date": earliest.get("published_date") if earliest else None,
        "exif_capture_date": exif_date,
        "date_difference_days": None,
        "comparison_basis": None,
    }

    if not searched:
        return {
            "status": STATUS_UNAVAILABLE, "confidence": 0.0,
            "analysis": analysis, "earliest_match": None,
            "note": "Reverse image search did not run, so nothing is known about "
                    "where this picture has appeared before.",
        }

    if not has_any_match:
        return {
            "status": STATUS_NO_MATCHES, "confidence": 0.0,
            "analysis": analysis, "earliest_match": None,
            "note": "No matching images were located online. This does NOT mean the "
                    "image is original — only that our providers did not find a copy.",
        }

    if not has_strong:
        return {
            "status": STATUS_SIMILAR_ONLY, "confidence": 0.0,
            "analysis": analysis, "earliest_match": None,
            "note": "Only visually similar pictures were found, not this exact image. "
                    "Similar-looking photos are not evidence about this one.",
        }

    if not earliest:
        return {
            "status": STATUS_PREVIOUSLY_PUBLISHED, "confidence": 0.0,
            "analysis": analysis, "earliest_match": None,
            "note": "This image has appeared online before, but no reliable publication "
                    "date could be read from the pages carrying it.",
        }

    earliest_dt = parse_date(earliest["published_date"])
    if not earliest_dt:
        return {
            "status": STATUS_PREVIOUSLY_PUBLISHED, "confidence": 0.0,
            "analysis": analysis, "earliest_match": earliest, "note": "",
        }

    # Compare against the claim's date when we have one, otherwise against now.
    reference_dt = parse_date(claim_date) if claim_date else None
    if reference_dt:
        analysis["comparison_basis"] = "claim_date"
        effective_confidence = claim_confidence
    else:
        reference_dt = datetime.now(timezone.utc)
        analysis["comparison_basis"] = "message_received"
        # Treating "now" as the claim date assumes the sender means "recent".
        effective_confidence = 0.55

    gap_days = (reference_dt - earliest_dt).days
    analysis["date_difference_days"] = gap_days

    if gap_days > RECYCLED_MIN_GAP_DAYS:
        confidence = _recycled_confidence(earliest, gap_days, effective_confidence)
        years = gap_days / 365.25
        span = f"{years:.1f} years" if years >= 1 else f"{gap_days} days"
        return {
            "status": STATUS_RECYCLED, "confidence": confidence,
            "analysis": analysis, "earliest_match": earliest,
            "note": f"This photograph was already online {span} before the date claimed "
                    f"for it. The picture itself may be genuine — but it does not show "
                    f"the event described.",
        }

    return {
        "status": STATUS_CONTEMPORANEOUS, "confidence": 0.0,
        "analysis": analysis, "earliest_match": earliest,
        "note": "The earliest located appearance is consistent with the claimed date.",
    }


async def _gemini_vision_search(image_path: str) -> Dict[str, Any]:
    """Zero-key Gemini Multimodal Vision visual scene search + DDGS web/news provenance scraper."""
    gemini_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    if not gemini_key or not os.path.exists(image_path):
        return {"matches": [], "available": False, "error": "No GEMINI_API_KEY"}

    try:
        from google import genai
        from PIL import Image
        from duckduckgo_search import DDGS

        client = genai.Client(api_key=gemini_key)
        with Image.open(image_path) as PIL_img:
            img = PIL_img.copy()

        prompt = (
            "You are an expert news photo investigator & reverse image search specialist.\n"
            "Analyze this photograph carefully:\n"
            "1. Identify the exact person, event, location, city, state, country, or context depicted "
            "(e.g. 'Young Narendra Modi speaking at RSS/BJP rally in front of Bharat Mata poster', 'Kerala Floods August 2018', etc.).\n"
            "2. Read any visible text, signboards, or banners.\n"
            "3. State the earliest known publication year or date of this photo if it is a classic news image.\n"
            "4. Provide 2-3 specific news search queries to locate original news reporting of this photo.\n"
            "Output valid JSON format strictly with keys: 'event_title', 'location', 'earliest_located_date', 'search_queries', 'is_recycled', 'explanation'."
        )

        response = None
        models_to_try = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-2.5-flash"]
        for m_name in models_to_try:
            try:
                resp = await asyncio.to_thread(
                    client.models.generate_content,
                    model=m_name,
                    contents=[prompt, img]
                )
                if resp and resp.text:
                    response = resp
                    break
            except Exception:
                continue

        if not response or not response.text:
            return {"matches": [], "available": False, "error": "Gemini response empty"}

        text = response.text.strip()
        import json, re
        parsed = {}
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except Exception:
                pass

        queries = parsed.get("search_queries") or [parsed.get("event_title") or "news photo"]
        queries = [q for q in queries if q][:3]

        matches = []
        def _ddg_search():
            ddg_res = []
            try:
                with DDGS() as ddgs:
                    for q in queries:
                        results = list(ddgs.news(q, max_results=4))
                        if not results:
                            results = list(ddgs.text(q, max_results=4))
                        for r in results:
                            ddg_res.append(r)
            except Exception:
                pass
            return ddg_res

        web_items = await asyncio.to_thread(_ddg_search)
        from services.image.match_ranker import normalise_match, FULL_MATCH, HIGH_VISUAL_SIMILARITY
        
        for item in web_items:
            url = item.get("href") or item.get("url") or ""
            title = item.get("title") or ""
            date_pub = item.get("date") or item.get("published")
            if url:
                m_type = FULL_MATCH if any(k in title.lower() for k in ["photo", "image", "rare", "picture", "modi", "kerala", "news"]) else HIGH_VISUAL_SIMILARITY
                matches.append(normalise_match(
                    url=url,
                    match_type=m_type,
                    source="gemini_vision_search",
                    page_title=title,
                    provider_date=date_pub,
                ))

        return {
            "matches": matches,
            "available": True,
            "gemini_info": parsed,
            "error": None
        }
    except Exception as e:
        log.warning("gemini_vision_search_failed", error=str(e))
        return {"matches": [], "available": False, "error": str(e)}


async def _run_providers(image_path: str, image_url: Optional[str]) -> Dict[str, Any]:
    """Stage 4 — reverse-search providers (Google Vision, SerpAPI Lens, Gemini Scene Search) concurrently."""
    vision_result, lens_result, gemini_result = await asyncio.gather(
        google_vision.search(image_path),
        serpapi_lens.search(image_path, image_url=image_url),
        _gemini_vision_search(image_path),
        return_exceptions=True,
    )

    providers: Dict[str, Any] = {}
    matches: List[Dict[str, Any]] = []

    for name, result in (("google_vision", vision_result), ("serpapi_lens", lens_result), ("gemini_vision", gemini_result)):
        if isinstance(result, Exception):
            providers[name] = {"available": False, "error": str(result), "n_matches": 0}
            continue
        providers[name] = {
            "available": result.get("available", False),
            "error": result.get("error"),
            "n_matches": len(result.get("matches", [])),
        }
        matches.extend(result.get("matches", []))

    gemini_info = (gemini_result.get("gemini_info") or {}) if isinstance(gemini_result, dict) else {}

    return {"matches": matches, "providers": providers, "vision": vision_result, "gemini_info": gemini_info}


async def reverse_image_check(
    image_path: str,
    claim_date: Optional[str] = None,
    claim_text: str = "",
    ai_generated_score: Optional[float] = None,
    image_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full provenance check for one image.
    """
    start = time.monotonic()

    if not image_path or not os.path.exists(image_path):
        return {**_empty_result("Image file does not exist."), "ai_generated_score": ai_generated_score}

    if not settings.reverse_search_enabled:
        result = _empty_result("Reverse image search is disabled (REVERSE_SEARCH_ENABLED=false).")
        result["ai_generated_score"] = ai_generated_score
        return result

    # ── Stage 1: fingerprint the original ───────────────────────────────────
    loop = asyncio.get_running_loop()
    fingerprint = await loop.run_in_executor(None, fingerprint_image, image_path)
    exif = fingerprint.get("metadata", {})

    # ── Stage 2: claim date ─────────────────────────────────────────────────
    claim_confidence = 0.9 if claim_date else 0.0
    claim_date_method = "caller" if claim_date else None
    if not claim_date and claim_text:
        extracted = extract_claim_date(claim_text)
        claim_date = extracted["date"]
        claim_confidence = extracted["confidence"]
        claim_date_method = extracted["method"]

    # ── Stages 3 & 4: local forensics ‖ reverse search ──────────────────────
    forensics, search = await asyncio.gather(
        run_forensics(image_path),
        _run_providers(image_path, image_url),
        return_exceptions=True,
    )

    if isinstance(forensics, Exception):
        log.warning("forensics_stage_failed", error=str(forensics))
        forensics = {"manipulation_score": 0.0, "signals": [], "error": str(forensics)}
    if isinstance(search, Exception):
        log.warning("search_stage_failed", error=str(search))
        search = {"matches": [], "providers": {}, "vision": {}}

    providers = search.get("providers", {})
    searched = any(p.get("available") for p in providers.values())

    # ── Stage 5: rank, then date the top pages ──────────────────────────────
    matches = rank_matches(search.get("matches", []))
    if matches:
        matches = await enrich_matches_with_dates(matches)

    # ── Stage 6: compare dates ──────────────────────────────────────────────
    exif_date = exif.get("capture_date_iso")
    comparison = _compare_dates(matches, claim_date, claim_confidence, exif_date, searched)

    vision = search.get("vision") or {}
    context_labels = vision.get("best_guess_labels", []) if isinstance(vision, dict) else []
    entities = [e["description"] for e in (vision.get("entities", []) if isinstance(vision, dict) else [])]
    gemini_info = search.get("gemini_info") or {}

    notes = [comparison["note"]] if comparison.get("note") else []
    for name, info in providers.items():
        if not info.get("available") and info.get("error"):
            notes.append(f"{name} unavailable: {info['error']}")

    result = {
        "image_hash": fingerprint.get("image_hash", ""),
        "phash": fingerprint.get("phash", ""),
        "dimensions": {"width": fingerprint.get("width", 0), "height": fingerprint.get("height", 0)},
        "format": fingerprint.get("format", ""),
        "file_size_bytes": fingerprint.get("file_size_bytes", 0),

        "ai_generated_score": ai_generated_score,
        "metadata": exif,
        "forensics": forensics,
        "gemini_info": gemini_info,

        "reverse_matches": matches[:settings.reverse_search_max_matches],
        "n_matches_total": len(matches),
        "earliest_located_date": comparison["analysis"].get("earliest_located_date"),
        "earliest_located_match": comparison.get("earliest_match"),

        "image_status": comparison["status"],
        "status_confidence": comparison["confidence"],
        "date_analysis": {
            **comparison["analysis"],
            "claim_date_method": claim_date_method,
        },

        "context_labels": context_labels,
        "web_entities": entities[:5],
        "providers": providers,
        "searched": searched,
        "notes": notes,
        "error": None,
        "latency_ms": int((time.monotonic() - start) * 1000),
    }

    log.info(
        "reverse_image_check_done",
        status=result["image_status"],
        confidence=result["status_confidence"],
        n_matches=len(matches),
        earliest=result["earliest_located_date"],
        latency_ms=result["latency_ms"],
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Presentation
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_HEADLINE = {
    STATUS_RECYCLED: "🔁 LIKELY RECYCLED / OUT OF CONTEXT",
    STATUS_CONTEMPORANEOUS: "🗓️ Consistent with the claimed date",
    STATUS_PREVIOUSLY_PUBLISHED: "🌐 Published online before",
    STATUS_SIMILAR_ONLY: "🔎 Only similar images found",
    STATUS_NO_MATCHES: "🔎 No earlier copies located",
    STATUS_UNAVAILABLE: "⚠️ Provenance check unavailable",
}


def _friendly_date(iso: Optional[str]) -> str:
    parsed = parse_date(iso) if iso else None
    return parsed.strftime("%d %B %Y") if parsed else "unknown"


def render_image_analysis(result: Dict[str, Any]) -> str:
    """
    Renders structured Telegram Image History card with full metadata, title, and all clickable match URLs.
    """
    if not result or result.get("error"):
        return ""

    lines = []

    # 1. Subject Identification & Visual Context
    gemini_info = result.get("gemini_info") or {}
    labels = result.get("context_labels") or []
    event_title = gemini_info.get("event_title") or (labels[0] if labels else "")
    location = gemini_info.get("location") or ""

    lines.append("🖼️ <b>IMAGE ANALYSIS & FORENSICS</b>")
    if event_title:
        lines.append(f"📌 <b>Identified Subject:</b> <i>{event_title}</i>")
    if location:
        lines.append(f"📍 <b>Context / Location:</b> <i>{location}</i>")

    # 2. AI & Forensics section
    ai_score = result.get("ai_generated_score")
    forensics = result.get("forensics") or {}
    manip = float(forensics.get("manipulation_score") or 0.0)
    level = "Low" if manip < 0.35 else ("Moderate" if manip < 0.6 else "High")

    if ai_score is not None:
        lines.append(f"• AI-generated probability: <b>{float(ai_score) * 100:.0f}%</b>")
    lines.append(f"• Manipulation signals: <b>{level}</b>")
    for signal in (forensics.get("signals") or [])[:2]:
        lines.append(f"   ◦ <i>{signal}</i>")

    # 3. Image History section
    status = result.get("image_status", STATUS_UNAVAILABLE)
    status_text = _STATUS_HEADLINE.get(status, status)
    matches = result.get("reverse_matches") or []
    copies_count = result.get("n_matches_total") or len(matches)

    analysis = result.get("date_analysis") or {}
    claim_date_raw = analysis.get("claim_date")
    claim_date_str = _friendly_date(claim_date_raw) if claim_date_raw else datetime.now(timezone.utc).strftime("%d %B %Y")

    lines.append("\n🖼️ <b>Image history</b>")
    lines.append(f"<i>{status_text}</i>\n")
    lines.append(f"Date claimed:       <b>{claim_date_str}</b>")
    lines.append(f"Copies found online: <b>{copies_count}</b>\n")

    # 4. Comprehensive List of ALL Matching Sources & Pages with HTML Hyperlinks
    if matches:
        lines.append("📎 <b>All Matching Pages & URLs Found Online:</b>")
        for i, m in enumerate(matches[:6], 1):
            url = m.get("url") or "#"
            title = m.get("page_title") or m.get("domain") or "Web Source"
            dom = m.get("domain") or "web"
            pub = _friendly_date(m.get("published_date")) if m.get("published_date") else ""
            
            date_tag = f" · {pub}" if pub and pub != "unknown" else ""
            lines.append(f"{i}. <a href='{url}'><b>{title}</b></a>\n   ◦ <code>{dom}</code>{date_tag}")
        lines.append("")

    # 5. Explanatory Footnotes & Date Difference
    gap = analysis.get("date_difference_days")
    if gap and gap > RECYCLED_MIN_GAP_DAYS:
        years = gap / 365.25
        span = f"~{years:.1f} years" if years >= 1 else f"{gap} days"
        lines.append(f"⚠️ <i>This photo was published {span} before the claimed date.</i>")

    if status == STATUS_SIMILAR_ONLY:
        lines.append("<i>Only visually similar pictures were found, not this exact image. Similar-looking photos are not evidence about this one.</i>")
    elif status in (STATUS_RECYCLED, STATUS_PREVIOUSLY_PUBLISHED):
        earliest = _friendly_date(result.get("earliest_located_date"))
        lines.append(f"<i>Exact copies were located online dating back to {earliest}.</i>")
    elif status == STATUS_UNAVAILABLE:
        lines.append("<i>Reverse image search check was unavailable or timed out.</i>")
    else:
        for note in (result.get("notes") or [])[:1]:
            lines.append(f"<i>{note}</i>")

    return "\n".join(lines)
