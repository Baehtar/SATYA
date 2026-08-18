"""
tests/test_provenance_integration.py — provenance reaching the front-ends.

The engine itself is covered in test_reverse_image_engine.py. This file checks
the wiring: that services/ml_service.py runs it, that a recycled image can
change the message verdict when — and only when — the message asserted a date,
and that both the Telegram card and the web card carry the finding.

Run:
    pytest tests/test_provenance_integration.py -v
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from services import ml_service
from services.ml_service import apply_provenance_fusion, check_image, check_mixed

NOW = datetime.now(timezone.utc)
TODAY = NOW.date().isoformat()
OLD = (NOW - timedelta(days=365 * 8)).date().isoformat()


def recycled_provenance(claim_method="relative_term", basis="claim_date", confidence=0.82):
    return {
        "image_status": "RECYCLED",
        "status_confidence": confidence,
        "searched": True,
        "ai_generated_score": 0.12,
        "earliest_located_date": OLD,
        "earliest_located_match": {"domain": "ndtv.com", "url": "https://ndtv.com/2018/floods"},
        "n_matches_total": 3,
        "reverse_matches": [{
            "url": "https://ndtv.com/2018/floods", "domain": "ndtv.com",
            "page_title": "Kerala floods", "match_type": "FULL_MATCH",
            "published_date": OLD, "date_confidence": 0.95,
        }],
        "date_analysis": {
            "claim_date": TODAY, "claim_date_method": claim_method,
            "comparison_basis": basis, "date_difference_days": 2920,
            "earliest_located_date": OLD,
        },
        "forensics": {"manipulation_score": 0.11, "signals": ["Resampling artefacts"]},
        "context_labels": ["kerala floods 2018"],
        "notes": ["This photograph was already online 8.0 years before the date claimed for it."],
    }


@pytest.fixture
def stub_pipeline(monkeypatch, tmp_path):
    """Replaces every external call check_image makes, except provenance."""
    from PIL import Image
    import numpy as np
    path = tmp_path / "img.jpg"
    rng = np.random.default_rng(3)
    Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(path, "JPEG")

    async def fake_ai(image_path):
        return {"verdict": "LIKELY_TRUE", "confidence": 0.88,
                "artificial_score": 0.12, "explanation": "Image AI-generation probability: 12.0%."}

    monkeypatch.setattr(ml_service, "check_image_ai", fake_ai)
    return str(path)


def stub_provenance(monkeypatch, provenance):
    async def fake(image_path, claim_text="", ai_score=None, progress_callback=None):
        return provenance
    monkeypatch.setattr(ml_service, "run_provenance", fake)


def run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
#  The fusion rule
# ─────────────────────────────────────────────────────────────────────────────

def test_recycled_image_flips_an_unverifiable_claim():
    """A dated claim contradicted by the photo's history is a false message."""
    verdict, confidence, note = apply_provenance_fusion(
        "UNVERIFIABLE", 0.5, recycled_provenance()
    )
    assert verdict == "LIKELY_FALSE"
    assert confidence == 0.82
    assert "does not show the event described" in note


def test_recycled_without_an_asserted_date_changes_nothing():
    """
    An old photo with no date claimed for it proves nothing — file photos are
    legitimate. This is the guard that keeps the engine from crying wolf.
    """
    provenance = recycled_provenance(claim_method=None, basis="message_received")
    verdict, confidence, note = apply_provenance_fusion("UNVERIFIABLE", 0.5, provenance)
    assert (verdict, confidence, note) == ("UNVERIFIABLE", 0.5, "")


def test_low_confidence_provenance_changes_nothing():
    provenance = recycled_provenance(confidence=0.30)
    verdict, _, note = apply_provenance_fusion("UNVERIFIABLE", 0.5, provenance)
    assert verdict == "UNVERIFIABLE" and note == ""


def test_true_claim_keeps_its_verdict_but_gains_the_note():
    """The words can be true while the picture is borrowed. Say both."""
    verdict, confidence, note = apply_provenance_fusion(
        "LIKELY_TRUE", 0.9, recycled_provenance()
    )
    assert verdict == "LIKELY_TRUE" and confidence == 0.9 and note


def test_non_recycled_statuses_are_inert():
    for status in ("NO_MATCHES_LOCATED", "SEARCH_UNAVAILABLE", "CONTEMPORANEOUS", "SIMILAR_ONLY"):
        provenance = {**recycled_provenance(), "image_status": status}
        assert apply_provenance_fusion("UNVERIFIABLE", 0.5, provenance) == ("UNVERIFIABLE", 0.5, "")


def test_missing_provenance_is_safe():
    assert apply_provenance_fusion("LIKELY_TRUE", 0.7, None) == ("LIKELY_TRUE", 0.7, "")
    assert apply_provenance_fusion("LIKELY_TRUE", 0.7, {}) == ("LIKELY_TRUE", 0.7, "")


# ─────────────────────────────────────────────────────────────────────────────
#  check_image
# ─────────────────────────────────────────────────────────────────────────────

def test_ai_image_mode_still_runs_provenance(monkeypatch, stub_pipeline):
    """
    The commonest fake is a real photo out of context — it scores 0% on an AI
    detector, so the dedicated AI mode must run provenance too.
    """
    stub_provenance(monkeypatch, recycled_provenance())

    result = run(check_image(stub_pipeline, mode="ai_image", caption="yesterday's flood"))

    assert result["provenance"]["image_status"] == "RECYCLED"
    assert "IMAGE ANALYSIS" in result["explanation"]
    assert "Earliest located appearance" in result["explanation"]
    # The AI verdict itself is untouched by provenance in this mode.
    assert result["image_ai_score"] == 0.12


def test_image_without_readable_text_still_reports_provenance(monkeypatch, stub_pipeline):
    async def fake_ocr(image_path):
        return {"raw_text": ""}

    monkeypatch.setattr("services.ocr.extract_text_from_image", fake_ocr, raising=False)
    import services.ocr as ocr_module
    monkeypatch.setattr(ocr_module, "extract_text_from_image", fake_ocr)
    monkeypatch.setattr(ocr_module, "normalize_ocr_result",
                        lambda text: {"has_readable_text": False, "cleaned_text": "", "language": "EN"})
    stub_provenance(monkeypatch, recycled_provenance())

    result = run(check_image(stub_pipeline))

    assert result["provenance"]["image_status"] == "RECYCLED"
    assert "IMAGE ANALYSIS" in result["explanation"]


def test_unavailable_provenance_adds_no_noise(monkeypatch, stub_pipeline):
    """No key configured → no IMAGE ANALYSIS block rather than an empty one."""
    stub_provenance(monkeypatch, {
        "image_status": "SEARCH_UNAVAILABLE", "searched": False,
        "forensics": {"signals": []}, "reverse_matches": [], "date_analysis": {},
        "notes": [], "ai_generated_score": 0.12,
    })

    result = run(check_image(stub_pipeline, mode="ai_image"))
    assert "IMAGE ANALYSIS" not in result["explanation"]


def test_provenance_failure_never_breaks_the_check(monkeypatch, stub_pipeline):
    stub_provenance(monkeypatch, None)
    result = run(check_image(stub_pipeline, mode="ai_image"))
    assert result["verdict"] == "LIKELY_TRUE"
    assert result["provenance"] is None


def test_run_provenance_times_out_gracefully(monkeypatch, stub_pipeline):
    """A slow publisher must never hold up a Telegram reply."""
    async def slow(*a, **kw):
        await asyncio.sleep(5)

    # run_provenance imports from services.image at call time, so patching the
    # package attribute is enough.
    import services.image as image_pkg
    monkeypatch.setattr(image_pkg, "reverse_image_check", slow)

    from src.config import settings
    monkeypatch.setattr(settings, "image_pipeline_timeout", 1)

    result = run(ml_service.run_provenance(stub_pipeline, ai_score=0.1))
    assert result["image_status"] == "SEARCH_UNAVAILABLE"
    assert result["searched"] is False


# ─────────────────────────────────────────────────────────────────────────────
#  check_mixed — image + caption
# ─────────────────────────────────────────────────────────────────────────────

def test_check_mixed_passes_the_caption_to_the_image_check(monkeypatch, stub_pipeline):
    """The caption is where the claimed date lives, so provenance needs it."""
    seen = {}

    async def fake_check_image(image_path, progress_callback=None, mode=None, caption=""):
        seen["caption"] = caption
        return {"type": "image", "verdict": "LIKELY_TRUE", "confidence": 0.8,
                "image_ai_score": 0.12, "explanation": "img",
                "provenance": recycled_provenance()}

    async def fake_check_text(text, progress_callback=None):
        return {"type": "text", "verdict": "UNVERIFIABLE", "confidence": 0.5,
                "explanation": "no fact-check found", "sources": []}

    monkeypatch.setattr(ml_service, "check_image", fake_check_image)
    monkeypatch.setattr(ml_service, "check_text", fake_check_text)

    result = run(check_mixed(stub_pipeline, "This shows yesterday's flood in Bihar"))

    assert seen["caption"] == "This shows yesterday's flood in Bihar"
    # Unverifiable claim + recycled photo with an asserted date → false message.
    assert result["verdict"] == "LIKELY_FALSE"
    assert result["provenance"]["image_status"] == "RECYCLED"


# ─────────────────────────────────────────────────────────────────────────────
#  Web card
# ─────────────────────────────────────────────────────────────────────────────

def test_web_card_carries_provenance_separately(monkeypatch):
    from UI.src import adapter

    async def no_llm(verdict_slug, claim, evidence, fallback=None):
        return ("en", "hi")
    monkeypatch.setattr(adapter, "_bilingual", no_llm)

    card = run(adapter.build_card({
        "type": "image", "verdict": "LIKELY_FALSE", "confidence": 0.82,
        "explanation": "x", "sources": [], "image_ai_score": 0.12,
        "extracted_claim": "flood in Bihar today",
        "provenance": recycled_provenance(),
    }))

    assert "RECYCLED_IMAGE" in card["image_flags"]
    assert card["provenance"]["status"] == "RECYCLED"
    assert card["provenance"]["earliest_located_date"] == OLD
    assert card["provenance"]["matches"][0]["domain"] == "ndtv.com"
    # The claim verdict is still the claim's, not the image's.
    assert card["verdict"] == "likely_false"


def test_web_card_flags_manipulation_signals(monkeypatch):
    from UI.src import adapter

    async def no_llm(verdict_slug, claim, evidence, fallback=None):
        return ("en", "hi")
    monkeypatch.setattr(adapter, "_bilingual", no_llm)

    provenance = recycled_provenance()
    provenance["forensics"] = {"manipulation_score": 0.72, "copy_move_score": 0.6, "signals": []}

    card = run(adapter.build_card({
        "type": "image", "verdict": "UNVERIFIABLE", "confidence": 0.5,
        "explanation": "x", "sources": [], "provenance": provenance,
    }))

    assert "MANIPULATION_SIGNALS" in card["image_flags"]
    assert "CLONED_REGION" in card["image_flags"]


def test_telegram_card_keeps_image_and_claim_apart():
    """The two findings must read as separate sections, not one blended verdict."""
    from bot.response import format_verdict
    from services.image import render_image_analysis

    block = render_image_analysis(recycled_provenance())
    message = format_verdict({
        "verdict": "LIKELY_FALSE", "confidence": 0.82,
        "explanation": f"<b>Claim Analysis: Likely False</b>\n\n{block}",
        "sources": [],
    })

    assert "IMAGE ANALYSIS" in message
    assert "Claim Analysis" in message
    assert "Earliest located appearance" in message
    # Never phrased as the original date — we only know the earliest we located.
    assert "Original image date" not in message
