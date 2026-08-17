"""
tests/test_api.py — Integration tests for the Satya backend API.

Run the API first:
    uvicorn src.api.main:app --port 8000

Then run tests:
    pytest tests/test_api.py -v
    
Or test manually with the quick_test() function at the bottom.
"""
import asyncio
import pytest
import httpx
from src.api.client import SatyaClient, VerdictResult

BASE_URL = "http://localhost:8000"


# ─────────────────────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health():
    client = SatyaClient(BASE_URL)
    assert await client.health(), "Backend is not reachable — is it running?"


# ─────────────────────────────────────────────────────────────
#  Text endpoint tests
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_text_false_claim():
    """Classic ₹15 lakh claim — should come back LIKELY_FALSE."""
    client = SatyaClient(BASE_URL)
    result = await client.check_text(
        "BREAKING: Government announces ₹15 lakh will be deposited in every Indian's account!"
    )
    assert isinstance(result, VerdictResult)
    assert result.verdict in ("likely_false", "unverifiable"), f"Expected false/unverifiable, got {result.verdict}"
    assert result.total_latency_ms < 60_000, f"Too slow: {result.total_latency_ms}ms"
    assert result.explanation_english
    assert result.explanation_hindi
    print(f"  ✅ Verdict: {result.verdict_emoji} {result.verdict} ({result.confidence_level}, {result.total_latency_ms}ms)")


@pytest.mark.asyncio
async def test_text_unverifiable():
    """Vague local claim — must be UNVERIFIABLE, not FALSE."""
    client = SatyaClient(BASE_URL)
    result = await client.check_text(
        "My neighbour said the new bridge in our area will collapse next week. Be careful!"
    )
    assert result.verdict == "unverifiable", (
        f"Vague local claim should be UNVERIFIABLE, got {result.verdict}. "
        "This is a calibration failure — check confidence.py"
    )
    print(f"  ✅ Correctly returned UNVERIFIABLE for local claim ({result.total_latency_ms}ms)")


@pytest.mark.asyncio
async def test_text_hinglish():
    """Hinglish forward — claim extractor must handle it."""
    client = SatyaClient(BASE_URL)
    result = await client.check_text(
        "Yaar sun, COVID vaccine mein microchip hai jo government use karti hai tracking ke liye. "
        "Sach mein! Ek doctor ne bataya. Abhi forward karo!"
    )
    assert result.verdict in ("likely_false", "unverifiable")
    assert result.explanation_hindi, "Hindi explanation must not be empty for Hinglish input"
    print(f"  ✅ Hinglish forward: {result.verdict} ({result.total_latency_ms}ms)")


@pytest.mark.asyncio
async def test_text_opinion_not_false():
    """Pure opinion — should be UNVERIFIABLE (not FALSE)."""
    client = SatyaClient(BASE_URL)
    result = await client.check_text(
        "I think the current government is doing a terrible job. Everything is getting worse."
    )
    assert result.verdict == "unverifiable", (
        f"Opinions must not be marked FALSE — got {result.verdict}"
    )
    print(f"  ✅ Opinion correctly returned UNVERIFIABLE")


@pytest.mark.asyncio
async def test_text_response_schema():
    """All required fields must be present in the response."""
    client = SatyaClient(BASE_URL)
    result = await client.check_text("India won the 1983 Cricket World Cup.")
    assert result.request_id
    assert result.verdict
    assert result.confidence_level in ("high", "moderate", "low")
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.explanation_english
    assert result.explanation_hindi
    assert isinstance(result.signals_used, list)
    assert isinstance(result.sources, list)
    print(f"  ✅ Schema valid: all fields present")


# ─────────────────────────────────────────────────────────────
#  Image endpoint tests
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_image_upload(tmp_path):
    """Upload a real PNG — pipeline must not crash."""
    # Create a minimal valid 1x1 PNG
    import struct, zlib
    def make_png():
        def chunk(tag, data):
            c = struct.pack(">I", len(data)) + tag + data
            return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        png += chunk(b"IEND", b"")
        return png

    img_bytes = make_png()
    client = SatyaClient(BASE_URL)
    result = await client.check_image(img_bytes, filename="test.png")
    assert isinstance(result, VerdictResult)
    assert result.verdict  # any verdict is fine
    print(f"  ✅ Image upload: {result.verdict} ({result.total_latency_ms}ms)")


# ─────────────────────────────────────────────────────────────
#  Unified endpoint test
# ─────────────────────name────────────────────────────────────

@pytest.mark.asyncio
async def test_unified_text():
    """Unified /check/ endpoint with text."""
    client = SatyaClient(BASE_URL)
    result = await client.check_auto(
        text="The Earth is flat and NASA is hiding it from us."
    )
    assert result.verdict in ("likely_false", "unverifiable")
    print(f"  ✅ Unified endpoint: {result.verdict}")


# ─────────────────────────────────────────────────────────────
#  Verdict cache test
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verdict_cache():
    """Verdict should be retrievable by request_id."""
    client = SatyaClient(BASE_URL)
    result = await client.check_text("India is in Asia.")
    
    async with httpx.AsyncClient() as http:
        r = await http.get(f"{BASE_URL}/check/verdict/{result.request_id}")
        assert r.status_code == 200
        cached = r.json()
        assert cached["request_id"] == result.request_id
    print(f"  ✅ Cache hit for request_id: {result.request_id}")


# ─────────────────────────────────────────────────────────────
#  Manual quick-test (no pytest needed)
# ─────────────────────────────────────────────────────────────

async def quick_test():
    """Run this directly: python tests/test_api.py"""
    print("\n🔍 Satya API Quick Test\n" + "=" * 50)

    client = SatyaClient(BASE_URL)

    if not await client.health():
        print("❌ Backend not reachable. Run: uvicorn src.api.main:app --port 8000")
        return

    test_cases = [
        ("₹15 lakh claim", "BREAKING: ₹15 lakh will be deposited in everyone's account! Forward this now!"),
        ("Hinglish misinformation", "Yaar, vaccine mein microchip hai! Doctor ne confirm kiya hai!"),
        ("True historical fact", "India won independence on August 15, 1947."),
        ("Pure opinion", "I think politicians are all corrupt and nothing works in this country."),
        ("Local unverifiable", "The electricity will be cut in our area for 3 days next week."),
    ]

    for name, text in test_cases:
        try:
            result = await client.check_text(text)
            print(
                f"{result.verdict_emoji} [{name}]\n"
                f"   Verdict: {result.verdict:25} Confidence: {result.confidence_bar} {int(result.confidence_score*100)}%\n"
                f"   EN: {result.explanation_english[:80]}...\n"
                f"   HI: {result.explanation_hindi[:80]}...\n"
                f"   Latency: {result.total_latency_ms}ms  |  Sources: {len(result.sources)}\n"
            )
        except Exception as e:
            print(f"❌ [{name}] Error: {e}\n")

    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(quick_test())
