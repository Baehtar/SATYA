"""
services/image/reverse_engine.py — Unified Master Image Reverse Engine & Forensics Dispatcher.

Integrates:
  1. Image preservation (SHA256, dHash, pHash, EXIF)
  2. Local Forensics (ELA, EXIF anomalies, Noise splicing, Copy-move)
  3. Reverse Search (Google Cloud Vision Web Detection + SerpAPI Google Lens in parallel)
  4. Match Ranking & Canonical Deduplication
  5. Multi-tier Page Fetching & Publication Date Extraction
  6. Provenance Calculation & Date Comparison (Recycled image detection)
"""
import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import httpx
import structlog

from services.image.metadata import extract_metadata
from services.image.google_vision import search_google_vision_web
from services.image.serpapi_lens import search_serpapi_google_lens
from services.image.date_extractor import extract_page_date, parse_date_string
from services.image.match_ranker import rank_and_deduplicate_matches
from services.image.image_forensics import run_image_forensics

log = structlog.get_logger(__name__)

async def _analyze_image_with_gemini(image_path: str) -> Dict[str, Any]:
    gemini_key = os.getenv("GEMINI_API_KEY") or getattr(settings, "gemini_api_key", "")
    if not gemini_key or not os.path.exists(image_path):
        return {}

    try:
        from google import genai
        from PIL import Image

        client = genai.Client(api_key=gemini_key)
        start_time = time.monotonic()

        with Image.open(image_path) as PIL_img:
            img = PIL_img.copy()

        prompt = (
            "You are an expert news photo investigator & reverse image search specialist.\n"
            "Analyze this photograph carefully:\n"
            "1. Identify the exact event, location, city, state, country, or context depicted (e.g. 'Kerala Floods August 2018 in Ranni/Pathanamthitta', '2020 Delhi Farmer Protests', etc.).\n"
            "2. Read any visible storefront signboards, street names, vehicle license plates, or banners.\n"
            "3. State the earliest known publication year or date of this photo if it is a classic or previously published news image.\n"
            "4. Provide 2-3 specific news search queries to locate original news reporting of this photo.\n"
            "Output valid JSON format strictly with keys: 'event_title', 'location', 'earliest_located_date', 'search_queries', 'is_recycled', 'explanation'."
        )

        models_to_try = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-2.5-flash", "gemini-flash-latest"]
        response = None

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
            except Exception as m_err:
                if "429" in str(m_err):
                    await asyncio.sleep(0.5)
                continue

        if not response or not response.text:
            return {}

        text = (response.text or "").strip()
        elapsed = time.monotonic() - start_time

        import json
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                parsed["processing_time"] = round(elapsed, 2)
                return parsed
            except Exception:
                pass

        return {
            "explanation": text,
            "processing_time": round(elapsed, 2)
        }

    except Exception as e:
        log.warning("gemini_image_analysis_failed", error=str(e))
        return {}


async def reverse_image_check(
    image_path: str,
    claimed_date: Optional[str] = None,
    ai_generated_score: float = 0.0
) -> Dict[str, Any]:
    """
    Main entry point for reverse image engine and forensics.
    
    Returns structured dict per architectural spec:
      {
        "image_hash": str,
        "dhash": str,
        "phash": str,
        "ai_generated_score": float,
        "metadata": dict,
        "forensics": dict,
        "reverse_matches": List[Dict[str, Any]],
        "earliest_located_date": str | None,
        "earliest_located_source": str | None,
        "image_status": str,  # RECYCLED, ORIGINAL_OR_NEW, UNVERIFIABLE
        "date_analysis": dict,
        "latency_ms": int
      }
    """
    start_time = time.monotonic()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    log.info("reverse_engine_start", path=image_path, claimed_date=claimed_date)

    # ── STAGE 1: Metadata & Hashes ───────────────────────────────────────────
    meta = extract_metadata(image_path)

    # ── STAGE 2: Local Forensics ──────────────────────────────────────────────
    forensics = run_image_forensics(image_path, meta)

    # ── STAGE 3: Parallel Reverse Search (Gemini Visual + Vision + Lens) ──────
    gemini_task = asyncio.create_task(_analyze_image_with_gemini(image_path))
    vision_task = asyncio.create_task(search_google_vision_web(image_path))
    lens_task = asyncio.create_task(search_serpapi_google_lens(image_path))

    gemini_data, vision_res, lens_res = await asyncio.gather(
        gemini_task, vision_task, lens_task, return_exceptions=True
    )

    gemini_info = gemini_data if isinstance(gemini_data, dict) else {}
    google_vision_data = vision_res if isinstance(vision_res, dict) else {}
    serpapi_lens_data = lens_res if isinstance(lens_res, dict) else {}

    # ── STAGE 4: Free Web News Search via Gemini Extracted Queries ────────────
    from src.pipelines.text.adapters.web_news import search_web_news
    web_news_matches = []
    queries = gemini_info.get("search_queries", [])
    if not queries and gemini_info.get("event_title"):
        queries = [gemini_info["event_title"]]

    for q in queries[:2]:
        try:
            news_res = await search_web_news(q)
            for item in news_res:
                web_news_matches.append({
                    "url": item.source_url,
                    "canonical_url": item.source_url,
                    "match_type": "FULL_MATCH",
                    "similarity_weight": 0.85,
                    "source_provider": f"News ({item.source_name})",
                    "page_title": item.original_claim,
                    "published_date": item.fact_check_date,
                    "date_confidence": 0.90
                })
        except Exception:
            pass

    # ── STAGE 5: Normalize & Rank Matches ───────────────────────────────────────
    ranked_matches = rank_and_deduplicate_matches(google_vision_data, serpapi_lens_data)
    
    # Prepend web news matches if found
    for w_m in web_news_matches:
        if not any(m["url"] == w_m["url"] for m in ranked_matches):
            ranked_matches.append(w_m)

    # ── STAGE 6: Fetch Top Pages & Extract Publication Dates ───────────────────
    top_matches = ranked_matches[:8]
    if top_matches:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as http_client:
            fetch_tasks = [
                extract_page_date(m["url"], client=http_client)
                for m in top_matches if not m.get("published_date")
            ]
            if fetch_tasks:
                page_date_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

                for i, p_res in enumerate(page_date_results):
                    if isinstance(p_res, dict) and p_res.get("published_date"):
                        top_matches[i]["published_date"] = p_res["published_date"]
                        top_matches[i]["date_confidence"] = p_res["date_confidence"]
                        if not top_matches[i].get("page_title") and p_res.get("page_title"):
                            top_matches[i]["page_title"] = p_res["page_title"]

    # ── STAGE 7: Calculate Provenance & Date Comparison ─────────────────────────
    earliest_date_str = None
    earliest_dt = None
    earliest_source_url = None

    # Check if Gemini extracted a valid earliest date (e.g. 2018-08-14)
    gemini_date_str = gemini_info.get("earliest_located_date")
    if gemini_date_str:
        gemini_dt = parse_date_string(gemini_date_str)
        if gemini_dt:
            earliest_dt = gemini_dt
            earliest_date_str = gemini_date_str
            earliest_source_url = "Gemini Visual Archive Inspection"

    # Filter matches with valid publication dates
    matches_with_dates = []
    for m in ranked_matches:
        if m.get("published_date"):
            dt = parse_date_string(m["published_date"])
            if dt:
                matches_with_dates.append((dt, m["published_date"], m["url"]))

    if matches_with_dates:
        matches_with_dates.sort(key=lambda x: x[0])
        first_dt, first_date_str, first_url = matches_with_dates[0]
        if not earliest_dt or first_dt < earliest_dt:
            earliest_dt = first_dt
            earliest_date_str = first_date_str
            earliest_source_url = first_url

    # Calculate date analysis & recycled status
    image_status = "ORIGINAL_OR_NEW"
    date_diff_days = None
    recycled_conf = 0.0

    now_utc = datetime.now(timezone.utc)
    target_claim_dt = parse_date_string(claimed_date) if claimed_date else now_utc

    if earliest_dt:
        date_diff_days = (target_claim_dt - earliest_dt).days

        if claimed_date and date_diff_days > 30:
            image_status = "RECYCLED"
            recycled_conf = min(1.0, date_diff_days / 365.0)

        elif (now_utc - earliest_dt).days > 180 or gemini_info.get("is_recycled"):
            image_status = "RECYCLED"
            recycled_conf = 0.90
            if date_diff_days is None:
                date_diff_days = (now_utc - earliest_dt).days

    if not ranked_matches and not gemini_info.get("event_title") and (google_vision_data.get("error") and serpapi_lens_data.get("error")):
        image_status = "UNVERIFIABLE"

    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    log.info(
        "reverse_engine_complete",
        status=image_status,
        matches=len(ranked_matches),
        earliest_date=earliest_date_str,
        date_diff=date_diff_days,
        latency_ms=elapsed_ms
    )

    camera_str = f"{meta.get('camera_make', '')} {meta.get('camera_model', '')}".strip()

    return {
        "image_hash": meta.get("sha256", ""),
        "dhash": meta.get("dhash", ""),
        "phash": meta.get("phash", ""),
        "ai_generated_score": ai_generated_score,
        "metadata": {
            "exif_present": meta.get("exif_present", False),
            "camera": camera_str,
            "software": meta.get("software", ""),
            "creation_time": meta.get("creation_time", ""),
            "gps_present": meta.get("gps_present", False),
            "dimensions": meta.get("dimensions", [0, 0])
        },
        "forensics": forensics,
        "gemini_visual_info": gemini_info,
        "reverse_matches": ranked_matches,
        "earliest_located_date": earliest_date_str,
        "earliest_located_source": earliest_source_url,
        "image_status": image_status,
        "date_analysis": {
            "claim_date": claimed_date,
            "earliest_located_date": earliest_date_str,
            "date_difference_days": date_diff_days,
            "recycled_confidence": round(recycled_conf, 2)
        },
        "latency_ms": elapsed_ms
    }
