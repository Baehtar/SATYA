"""
tests/test_judging_set.py — 8-item live judging set runner.

Run with: python -m pytest tests/test_judging_set.py -v

Contains:
  2x true claims
  2x false claims
  2x unverifiable claims
  2x adversarial trick forwards

Each test measures verdict accuracy + latency.
"""
import asyncio
import time
import pytest
from src.models.schemas import Verdict, CheckRequest, MessageType

# ─── Judging Set Fixtures ────────────────────────────────────────────────────

JUDGING_SET = [
    # ── True Claims ──────────────────────────────────────────────────────────
    {
        "id": "TC-01",
        "label": "TRUE",
        "type": "text",
        "content": "India won the Cricket World Cup in 1983 under Kapil Dev's captaincy.",
        "expected_verdict": Verdict.LIKELY_TRUE,
        "description": "Verifiable historical fact — should match fact sources",
    },
    {
        "id": "TC-02",
        "label": "TRUE",
        "type": "text",
        "content": "The Indian Constitution was adopted on 26 November 1949.",
        "expected_verdict": Verdict.LIKELY_TRUE,
        "description": "Established constitutional fact",
    },

    # ── False Claims ─────────────────────────────────────────────────────────
    {
        "id": "FC-01",
        "label": "FALSE",
        "type": "text",
        "content": "BREAKING: Government announces ₹15 lakh will be deposited in every Indian citizen's bank account by December. Share immediately! सच्चाई जानो!",
        "expected_verdict": Verdict.LIKELY_FALSE,
        "description": "Classic ₹15 lakh misinformation — PIB has fact-checked this",
    },
    {
        "id": "FC-02",
        "label": "FALSE",
        "type": "text",
        "content": "COVID vaccine contains microchips to track Indians. This has been confirmed by a Pune doctor. Forward to everyone you know!",
        "expected_verdict": Verdict.LIKELY_FALSE,
        "description": "Vaccine misinformation — extensively fact-checked",
    },

    # ── Unverifiable Claims ───────────────────────────────────────────────────
    {
        "id": "UV-01",
        "label": "UNVERIFIABLE",
        "type": "text",
        "content": "My neighbour told me that the new metro line in our city will be cancelled next month. Very sad news.",
        "expected_verdict": Verdict.UNVERIFIABLE,
        "description": "Hyper-local claim — no fact-check possible",
    },
    {
        "id": "UV-02",
        "label": "UNVERIFIABLE",
        "type": "text",
        "content": "Suno suno! Kal raat ek badi company ka server hack hua hai. Bahut log affected hain. Details abhi nahi bata sakta.",
        "expected_verdict": Verdict.UNVERIFIABLE,
        "description": "Vague Hinglish claim with no verifiable details",
    },

    # ── Adversarial Forwards ──────────────────────────────────────────────────
    {
        "id": "ADV-01",
        "label": "ADVERSARIAL",
        "type": "text",
        "content": (
            "The Prime Minister of India has officially announced a new ₹10,000 "
            "monthly stipend scheme for all Indians above 60. "
            "This is 100% verified by Times of India."  # Fake attribution
        ),
        "expected_verdict": Verdict.LIKELY_FALSE,
        "description": "False claim with fake credible attribution — adversarial",
    },
    {
        "id": "ADV-02",
        "label": "ADVERSARIAL",
        "type": "text",
        "content": (
            "PIB FACT CHECK: The claim that onion prices will rise is FALSE. "  # True PIB claim
            "But ALSO: all ATMs in India will be shut down for 3 days next week."  # Appended false claim
        ),
        "expected_verdict": Verdict.LIKELY_FALSE,
        "description": "Mixed true+false in one message — tests claim extraction precision",
    },
]


# ─── Test Runner ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("item", JUDGING_SET, ids=[i["id"] for i in JUDGING_SET])
async def test_judging_item(item):
    """Runs a single judging set item through the full text pipeline."""
    from src.pipelines.text.pipeline import run_text_pipeline
    from src.verdict.aggregator import aggregate_evidence
    from src.verdict.confidence import calibrate_confidence
    from src.models.schemas import EvidenceBundle

    start = time.monotonic()

    request = CheckRequest(
        request_id=item["id"],
        message_type=MessageType.TEXT,
        user_id=0,
        chat_id=0,
        text_content=item["content"],
    )

    # Run text pipeline
    claim_analysis = await run_text_pipeline(request)

    # Build bundle and run verdict engine
    bundle = EvidenceBundle(
        request_id=item["id"],
        message_type=MessageType.TEXT,
        claim_analysis=claim_analysis,
    )
    evidence = aggregate_evidence(bundle)
    score = calibrate_confidence(evidence)

    latency_ms = int((time.monotonic() - start) * 1000)

    print(f"\n{'='*60}")
    print(f"ID: {item['id']} ({item['label']})")
    print(f"Description: {item['description']}")
    print(f"Extracted claim: {claim_analysis.extracted_claim[:80]}")
    print(f"Verdict: {score.verdict.value}")
    print(f"Expected: {item['expected_verdict'].value}")
    print(f"Confidence: {int(score.confidence_score * 100)}% ({score.confidence_level.value})")
    print(f"Latency: {latency_ms}ms")
    print(f"Match: {'✅' if score.verdict == item['expected_verdict'] else '❌'}")

    # Assertions
    assert latency_ms < 60_000, f"Exceeded 60s timeout: {latency_ms}ms"
    assert score.verdict == item["expected_verdict"], (
        f"Wrong verdict: got {score.verdict.value}, expected {item['expected_verdict'].value}"
    )


if __name__ == "__main__":
    """Direct run: python tests/test_judging_set.py"""
    async def run_all():
        print("\n🔍 SATYA — Judging Set Evaluation\n" + "="*60)
        results = []
        for item in JUDGING_SET:
            try:
                from src.pipelines.text.pipeline import run_text_pipeline
                from src.verdict.aggregator import aggregate_evidence
                from src.verdict.confidence import calibrate_confidence
                from src.models.schemas import EvidenceBundle

                start = time.monotonic()
                request = CheckRequest(
                    request_id=item["id"],
                    message_type=MessageType.TEXT,
                    user_id=0, chat_id=0,
                    text_content=item["content"],
                )
                claim_analysis = await run_text_pipeline(request)
                bundle = EvidenceBundle(
                    request_id=item["id"],
                    message_type=MessageType.TEXT,
                    claim_analysis=claim_analysis,
                )
                evidence = aggregate_evidence(bundle)
                score = calibrate_confidence(evidence)
                latency_ms = int((time.monotonic() - start) * 1000)
                correct = score.verdict == item["expected_verdict"]
                results.append({"id": item["id"], "correct": correct, "latency_ms": latency_ms})
                status = "✅" if correct else "❌"
                print(f"{status} {item['id']:8} | {score.verdict.value:25} | {latency_ms:5}ms | {item['description'][:50]}")
            except Exception as e:
                print(f"💥 {item['id']:8} | ERROR: {e}")
                results.append({"id": item["id"], "correct": False, "latency_ms": -1})

        correct = sum(1 for r in results if r["correct"])
        avg_latency = sum(r["latency_ms"] for r in results if r["latency_ms"] > 0) // max(1, len(results))
        print(f"\n{'='*60}")
        print(f"Score: {correct}/{len(JUDGING_SET)} correct  |  Avg latency: {avg_latency}ms")

    asyncio.run(run_all())
