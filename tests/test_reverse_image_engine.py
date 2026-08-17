"""
tests/test_reverse_image_engine.py — the Reverse Image Engine.

Covers the 17 scenarios the engine was specified against. No network: the two
search providers and the page-date fetcher are monkeypatched, and date
extraction is exercised directly against HTML fixtures.

Run:
    pytest tests/test_reverse_image_engine.py -v
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image

from services.image import date_extractor, reverse_engine
from services.image.date_extractor import (
    METHOD_JSONLD, METHOD_META, METHOD_URL, METHOD_VISIBLE_TEXT,
    extract_claim_date, extract_dates_from_html,
)
from services.image.match_ranker import (
    EXACT_MATCH, FULL_MATCH, HIGH_VISUAL_SIMILARITY, LOW_VISUAL_SIMILARITY,
    PARTIAL_MATCH, canonical_url, normalise_match, rank_matches,
)
from services.image.metadata import fingerprint_image, hamming_distance, perceptual_hash
from services.image.reverse_engine import (
    STATUS_CONTEMPORANEOUS, STATUS_NO_MATCHES, STATUS_PREVIOUSLY_PUBLISHED,
    STATUS_RECYCLED, STATUS_SIMILAR_ONLY, STATUS_UNAVAILABLE, reverse_image_check,
)

NOW = datetime.now(timezone.utc)
TODAY = NOW.date().isoformat()
EIGHT_YEARS_AGO = (NOW - timedelta(days=365 * 8)).date().isoformat()
LAST_WEEK = (NOW - timedelta(days=7)).date().isoformat()


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def photo(tmp_path):
    """A JPEG with real structure — flat colour would defeat the forensics."""
    import numpy as np
    rng = np.random.default_rng(1234)
    noise = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    path = tmp_path / "photo.jpg"
    Image.fromarray(noise).save(path, "JPEG", quality=92)
    return str(path)


def match(url, match_type=FULL_MATCH, source="google_vision", date=None, confidence=0.9, **kw):
    m = normalise_match(url=url, match_type=match_type, source=source, **kw)
    m["published_date"] = date
    m["date_confidence"] = confidence if date else 0.0
    return m


def install_providers(monkeypatch, vision=None, lens=None, vision_available=True, lens_available=False):
    """Replaces both search providers with canned results."""
    async def fake_vision(image_path, timeout=None):
        return {
            "matches": vision or [], "entities": [], "best_guess_labels": ["kerala floods"],
            "available": vision_available,
            "error": None if vision_available else "GOOGLE_VISION_API_KEY is not configured.",
        }

    async def fake_lens(image_path, image_url=None, timeout=None):
        return {
            "matches": lens or [], "available": lens_available,
            "error": None if lens_available else "no public URL configured",
        }

    monkeypatch.setattr(reverse_engine.google_vision, "search", fake_vision)
    monkeypatch.setattr(reverse_engine.serpapi_lens, "search", fake_lens)


def freeze_dates(monkeypatch):
    """Matches arrive pre-dated by the fixtures, so skip the page fetches."""
    async def passthrough(matches, limit=None, timeout=None):
        return matches
    monkeypatch.setattr(reverse_engine, "enrich_matches_with_dates", passthrough)


def run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
#  1–4. The image was published before the date claimed for it
# ─────────────────────────────────────────────────────────────────────────────

def test_case_01_exact_match_with_old_news_photo(monkeypatch, photo):
    """An exact match on a dated news page, years before the claim → RECYCLED."""
    install_providers(monkeypatch, vision=[
        match("https://ndtv.com/news/kerala-floods-2018", FULL_MATCH, date=EIGHT_YEARS_AGO),
    ])
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(photo, claim_date=TODAY))

    assert result["image_status"] == STATUS_RECYCLED
    assert result["earliest_located_date"] == EIGHT_YEARS_AGO
    assert result["status_confidence"] > 0.5
    assert result["date_analysis"]["date_difference_days"] > 2800


def test_case_02_recycled_disaster_photo_reports_the_gap(monkeypatch, photo):
    install_providers(monkeypatch, vision=[
        match("https://thehindu.com/2018/08/16/kerala-floods", FULL_MATCH, date=EIGHT_YEARS_AGO),
        match("https://boomlive.in/fact-check/old-photo", FULL_MATCH, date=LAST_WEEK),
    ])
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(photo, claim_date=TODAY))

    # The EARLIEST located appearance anchors the comparison, not the newest.
    assert result["earliest_located_date"] == EIGHT_YEARS_AGO
    assert result["image_status"] == STATUS_RECYCLED


def test_case_03_crop_of_an_old_image_still_counts(monkeypatch, photo):
    """A partial match is a real match — a crop of the 2018 photo is that photo."""
    install_providers(monkeypatch, vision=[
        match("https://indianexpress.com/2018/08/kerala", PARTIAL_MATCH, date=EIGHT_YEARS_AGO),
    ])
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(photo, claim_date=TODAY))
    assert result["image_status"] == STATUS_RECYCLED
    # ...but with less confidence than a full match would give.
    assert result["status_confidence"] < 0.95


def test_case_04_meme_built_on_an_old_photo(monkeypatch, photo):
    install_providers(monkeypatch, vision=[
        match("https://knowyourmeme.com/photos/1234", PARTIAL_MATCH, date=EIGHT_YEARS_AGO),
        match("https://facebook.com/post/999", FULL_MATCH, date=LAST_WEEK),
    ])
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(photo, claim_date=TODAY))
    assert result["image_status"] == STATUS_RECYCLED
    assert result["earliest_located_date"] == EIGHT_YEARS_AGO


# ─────────────────────────────────────────────────────────────────────────────
#  5–8. Cases where the engine must NOT over-claim
# ─────────────────────────────────────────────────────────────────────────────

def test_case_05_no_matches_is_not_proof_of_originality(monkeypatch, photo):
    install_providers(monkeypatch, vision=[])
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(photo, claim_date=TODAY))

    assert result["image_status"] == STATUS_NO_MATCHES
    assert result["searched"] is True          # the search DID run
    assert result["status_confidence"] == 0.0
    assert "does NOT mean" in " ".join(result["notes"])


def test_case_06_ai_generated_score_passes_through_untouched(monkeypatch, photo):
    """The engine reports the AI score but never computes or overrides it."""
    install_providers(monkeypatch, vision=[])
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(photo, ai_generated_score=0.91))

    assert result["ai_generated_score"] == 0.91
    # An AI-generated picture with no matches is still "no matches" — the two
    # findings are independent.
    assert result["image_status"] == STATUS_NO_MATCHES


def test_case_07_edited_image_does_not_become_recycled(monkeypatch, tmp_path):
    """Resizing and re-saving raises forensics signals but proves nothing."""
    import numpy as np
    rng = np.random.default_rng(7)
    original = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    path = tmp_path / "resized.jpg"
    original.resize((200, 150)).save(path, "JPEG", quality=60)

    install_providers(monkeypatch, vision=[])
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(str(path)))

    assert result["image_status"] == STATUS_NO_MATCHES
    assert result["forensics"]["manipulation_score"] < 1.0
    assert isinstance(result["forensics"]["signals"], list)


def test_case_08_stripped_exif_is_reported_not_penalised(monkeypatch, photo):
    install_providers(monkeypatch, vision=[])
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(photo))

    # PIL writes no EXIF for our generated file — the normal case for a forward.
    assert result["metadata"]["exif_present"] is False
    assert result["metadata"]["capture_date_iso"] is None
    assert result["image_status"] == STATUS_NO_MATCHES


# ─────────────────────────────────────────────────────────────────────────────
#  9–13. Dates
# ─────────────────────────────────────────────────────────────────────────────

def test_case_09_false_publication_date_in_the_claim(monkeypatch, photo):
    """'Yesterday's flood' vs a 2018 photo — the claim's own date is the anchor."""
    install_providers(monkeypatch, vision=[
        match("https://ndtv.com/2017/08/bihar-floods", FULL_MATCH, date=EIGHT_YEARS_AGO),
    ])
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(photo, claim_text="This photo shows yesterday's flood in Bihar."))

    assert result["date_analysis"]["claim_date_method"] == "relative_term"
    assert result["date_analysis"]["comparison_basis"] == "claim_date"
    assert result["image_status"] == STATUS_RECYCLED


def test_case_10_multiple_source_pages_are_deduplicated(monkeypatch, photo):
    """The same page from both providers counts once, and records both."""
    install_providers(
        monkeypatch,
        vision=[match("https://www.ndtv.com/article?utm_source=x", FULL_MATCH, date=EIGHT_YEARS_AGO)],
        lens=[match("https://ndtv.com/article/", EXACT_MATCH, source="serpapi_lens", date=EIGHT_YEARS_AGO)],
        lens_available=True,
    )
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(photo, claim_date=TODAY))

    assert result["n_matches_total"] == 1
    merged = result["reverse_matches"][0]
    assert set(merged["sources"]) == {"google_vision", "serpapi_lens"}
    assert merged["match_type"] == EXACT_MATCH      # strongest type wins the merge


def test_case_11_conflicting_publication_dates_are_flagged():
    """JSON-LD says 2024, the URL says 2019 — report the conflict, prefer earliest."""
    html = """
    <html><head>
      <script type="application/ld+json">
        {"@type":"NewsArticle","datePublished":"2024-02-16T10:00:00+05:30"}
      </script>
      <meta property="article:published_time" content="2019-03-14T08:00:00Z">
    </head><body>Story</body></html>
    """
    result = extract_dates_from_html(html, "https://example.com/2019/03/14/story")

    assert result["conflicting"] is True
    assert result["date"] == "2019-03-14"           # earliest trusted candidate
    assert result["confidence"] < 0.95              # dinged for the disagreement


def test_case_12_provider_unavailable_is_not_a_finding(monkeypatch, photo):
    install_providers(monkeypatch, vision=[], vision_available=False, lens_available=False)
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(photo, claim_date=TODAY))

    assert result["image_status"] == STATUS_UNAVAILABLE
    assert result["searched"] is False
    assert result["status_confidence"] == 0.0
    assert any("unavailable" in n.lower() for n in result["notes"])


def test_case_13_page_without_a_machine_readable_date(monkeypatch, photo):
    """Found online, but no date could be read → say exactly that."""
    install_providers(monkeypatch, vision=[
        match("https://example.com/gallery", FULL_MATCH, date=None),
    ])
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(photo, claim_date=TODAY))

    assert result["image_status"] == STATUS_PREVIOUSLY_PUBLISHED
    assert result["earliest_located_date"] is None
    assert result["status_confidence"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  14–17. Claim integration and languages
# ─────────────────────────────────────────────────────────────────────────────

def test_case_14_image_with_extracted_news_claim(monkeypatch, photo):
    install_providers(monkeypatch, vision=[
        match("https://thehindu.com/2018/08/16/floods", FULL_MATCH, date=EIGHT_YEARS_AGO),
    ])
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(
        photo,
        claim_text="Massive flooding in Bihar on 16 August 2026 leaves thousands stranded",
        ai_generated_score=0.12,
    ))

    assert result["date_analysis"]["claim_date_method"] == "explicit_date"
    assert result["ai_generated_score"] == 0.12
    assert result["image_status"] == STATUS_RECYCLED
    block = reverse_engine.render_image_analysis(result)
    assert "IMAGE ANALYSIS" in block and "Earliest located appearance" in block


@pytest.mark.parametrize("text,expected_method", [
    ("यह तस्वीर कल की बाढ़ की है",              "relative_term"),   # Hindi: "kal"
    ("Aaj ki Bihar flood ki photo hai",         "relative_term"),   # Hinglish
    ("இந்த புகைப்படம் நேற்று வெள்ளத்தின்",        "relative_term"),   # Tamil: "நேற்று"
    ("Flood in Bihar on 16 August 2018",        "explicit_date"),   # English
    ("Some claim with no date at all",          None),
])
def test_cases_15_16_17_claim_dates_across_languages(text, expected_method):
    result = extract_claim_date(text)
    assert result["method"] == expected_method
    if expected_method:
        assert result["date"] is not None
        assert 0.0 < result["confidence"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Date extraction priority order
# ─────────────────────────────────────────────────────────────────────────────

def test_jsonld_beats_every_other_source():
    html = """
    <html><head>
      <script type="application/ld+json">
        {"@graph":[{"@type":"NewsArticle","datePublished":"2018-08-16"}]}
      </script>
      <meta property="article:published_time" content="2018-08-20">
      <time datetime="2018-08-25">25 Aug</time>
    </head><body>Published: 30 August 2018</body></html>
    """
    result = extract_dates_from_html(html, "https://x.com/2018/09/01/a")
    assert result["method"] == METHOD_JSONLD
    assert result["date"] == "2018-08-16"


def test_date_modified_is_never_used_as_publication_date():
    html = """
    <html><head><script type="application/ld+json">
      {"@type":"NewsArticle","dateModified":"2026-01-01"}
    </script></head><body>x</body></html>
    """
    result = extract_dates_from_html(html, "https://example.com/story")
    assert result["date"] is None


def test_meta_then_time_then_text_then_url():
    meta = extract_dates_from_html(
        '<html><head><meta name="pubdate" content="2019-05-02"></head><body>x</body></html>', "")
    assert meta["method"] == METHOD_META and meta["date"] == "2019-05-02"

    text = extract_dates_from_html(
        "<html><body>Published: 14 March 2019 by staff</body></html>", "")
    assert text["method"] == METHOD_VISIBLE_TEXT and text["date"] == "2019-03-14"

    url = extract_dates_from_html("<html><body>no dates</body></html>",
                                  "https://site.com/2019/03/14/headline")
    assert url["method"] == METHOD_URL and url["date"] == "2019-03-14"


def test_body_dates_without_a_publication_cue_are_ignored():
    """An article ABOUT the 2018 floods isn't an article published in 2018."""
    html = "<html><body>The 16 August 2018 floods were devastating.</body></html>"
    assert extract_dates_from_html(html, "https://example.com/story")["date"] is None


def test_implausible_dates_are_rejected():
    future = (NOW + timedelta(days=400)).date().isoformat()
    assert date_extractor.parse_date(future) is None      # the future
    assert date_extractor.parse_date("1887-01-01") is None  # before photography indexing
    assert date_extractor.parse_date("not a date") is None
    assert date_extractor.parse_date("") is None


def test_indian_day_first_date_parsing():
    """03/04/2019 is 3 April in Indian sources, not 4 March."""
    parsed = date_extractor.parse_date("03/04/2019")
    assert parsed is not None and parsed.month == 4 and parsed.day == 3


# ─────────────────────────────────────────────────────────────────────────────
#  Ranking
# ─────────────────────────────────────────────────────────────────────────────

def test_canonical_url_collapses_variants():
    assert canonical_url("https://www.NDTV.com/story/?utm_source=wa#top") == \
           canonical_url("http://ndtv.com/story")


def test_ranking_puts_strong_matches_first():
    ranked = rank_matches([
        match("https://a.com/1", LOW_VISUAL_SIMILARITY),
        match("https://b.com/2", EXACT_MATCH),
        match("https://c.com/3", HIGH_VISUAL_SIMILARITY),
        match("https://d.com/4", FULL_MATCH),
    ])
    assert [m["match_type"] for m in ranked] == [
        EXACT_MATCH, FULL_MATCH, HIGH_VISUAL_SIMILARITY, LOW_VISUAL_SIMILARITY,
    ]


def test_visual_similarity_alone_never_yields_recycled(monkeypatch, photo):
    """A different photo that looks alike says nothing about this one."""
    install_providers(monkeypatch, vision=[
        match("https://old.com/2015/pic", HIGH_VISUAL_SIMILARITY, date=EIGHT_YEARS_AGO),
        match("https://other.com/2016/pic", LOW_VISUAL_SIMILARITY, date=EIGHT_YEARS_AGO),
    ])
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(photo, claim_date=TODAY))

    assert result["image_status"] == STATUS_SIMILAR_ONLY
    assert result["status_confidence"] == 0.0


def test_recent_match_is_contemporaneous_not_recycled(monkeypatch, photo):
    install_providers(monkeypatch, vision=[
        match("https://news.com/today", FULL_MATCH, date=LAST_WEEK),
    ])
    freeze_dates(monkeypatch)

    result = run(reverse_image_check(photo, claim_date=TODAY))
    assert result["image_status"] == STATUS_CONTEMPORANEOUS


# ─────────────────────────────────────────────────────────────────────────────
#  Fingerprinting
# ─────────────────────────────────────────────────────────────────────────────

def test_sha256_and_phash_are_computed(photo):
    fingerprint = fingerprint_image(photo)
    assert len(fingerprint["image_hash"]) == 64
    assert len(fingerprint["phash"]) == 16
    assert fingerprint["width"] == 320 and fingerprint["height"] == 240


def test_phash_survives_resize_and_recompression(tmp_path):
    """The property the whole engine leans on: a forward is still the same image."""
    import numpy as np
    rng = np.random.default_rng(42)
    base = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).filter(
        __import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(3)
    )
    original = tmp_path / "a.jpg"
    base.save(original, "JPEG", quality=95)

    forwarded = tmp_path / "b.jpg"
    base.resize((180, 180)).save(forwarded, "JPEG", quality=45)

    a = perceptual_hash(Image.open(original))
    b = perceptual_hash(Image.open(forwarded))
    assert hamming_distance(a, b) <= 10

    different = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    assert hamming_distance(a, perceptual_hash(different)) > 10


# ─────────────────────────────────────────────────────────────────────────────
#  Forensics calibration
#
#  These thresholds were tuned against real files (a 12 MP Samsung original, a
#  rendered screenshot, a WhatsApp-style downscale). They are the difference
#  between a signal and a light that is always on, so they get tests.
# ─────────────────────────────────────────────────────────────────────────────

def _textured(width=800, height=600, seed=11):
    """Photo-like content — flat or pure-noise images are degenerate here."""
    import numpy as np
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.sin(xx / 17.0) * 40 + np.cos(yy / 23.0) * 40 + 128
    base = base[:, :, None].repeat(3, axis=2) + rng.normal(0, 12, (height, width, 3))
    return Image.fromarray(np.clip(base, 0, 255).astype("uint8")).filter(
        ImageFilter.GaussianBlur(0.6)
    )


def test_copy_move_detects_a_pasted_region(tmp_path):
    from services.image.image_forensics import _load_grey_native, copy_move_detection

    clean = _textured()
    clean_path = tmp_path / "clean.jpg"
    clean.save(clean_path, "JPEG", quality=92)

    forged = clean.copy()
    forged.paste(forged.crop((100, 100, 220, 220)), (500, 380))
    forged_path = tmp_path / "forged.jpg"
    forged.save(forged_path, "JPEG", quality=92)

    clean_score, clean_pairs = copy_move_detection(_load_grey_native(str(clean_path), 1536))
    forged_score, forged_pairs = copy_move_detection(_load_grey_native(str(forged_path), 1536))

    assert forged_score >= 0.9 and forged_pairs > 100
    assert clean_score == 0.0 and clean_pairs < 12   # below the support threshold


def test_resampling_fires_on_a_resize_not_on_an_original(tmp_path):
    from services.image.image_forensics import _load_grey_native, resampling_score

    original = _textured()
    original_path = tmp_path / "original.jpg"
    original.save(original_path, "JPEG", quality=92)

    resized_path = tmp_path / "resized.jpg"
    original.resize((320, 240)).save(resized_path, "JPEG", quality=40)

    assert resampling_score(_load_grey_native(str(original_path), 2048)) < 0.3
    assert resampling_score(_load_grey_native(str(resized_path), 2048)) > 0.5


def test_forensics_reads_real_camera_exif():
    """A genuine Samsung original: EXIF present, and no manipulation signals."""
    import os
    photo = "UI/AJWH6830.JPG"
    if not os.path.exists(photo):
        pytest.skip("sample photo not in the repo")

    from services.image.metadata import fingerprint_image
    fingerprint = fingerprint_image(photo)
    exif = fingerprint["metadata"]

    assert exif["exif_present"] is True
    assert exif["camera"] and exif["capture_date_iso"]
    # An untouched camera file must not be flagged as edited.
    assert exif["editing_software_detected"] is None


def test_jpeg_quality_estimate_tracks_the_save_setting(tmp_path):
    from services.image.image_forensics import jpeg_analysis

    image = _textured(320, 240)
    high, low = tmp_path / "high.jpg", tmp_path / "low.jpg"
    image.save(high, "JPEG", quality=95)
    image.save(low, "JPEG", quality=40)

    high_q = jpeg_analysis(str(high))["estimated_quality"]
    low_q = jpeg_analysis(str(low))["estimated_quality"]

    assert high_q > low_q
    assert 85 <= high_q <= 100 and 30 <= low_q <= 55
    assert jpeg_analysis(str(low))["recompressed"] is True


def test_png_is_not_treated_as_jpeg(tmp_path):
    from services.image.image_forensics import jpeg_analysis
    path = tmp_path / "a.png"
    _textured(120, 90).save(path, "PNG")
    result = jpeg_analysis(str(path))
    assert result["is_jpeg"] is False and result["estimated_quality"] is None


def test_missing_file_is_handled_everywhere(monkeypatch):
    install_providers(monkeypatch, vision=[])
    result = run(reverse_image_check("/nonexistent/image.jpg"))
    assert result["error"] and result["image_status"] == STATUS_UNAVAILABLE
