"""
src/pipelines/text/claim_matcher.py — Semantic claim matching using async google.genai SDK with retries.
Evaluates Who, What, When, Where, and determines verdict across fact-checks & mainstream news.
"""
import asyncio
import structlog
from google import genai
from google.genai import types
from src.config import settings
from src.models.schemas import (
    ClaimAnalysis,
    FactCheckMatch,
    Verdict,
    ClaimType,
    LanguageCode,
    ClaimMatchSchema,
)

log = structlog.get_logger(__name__)


def _get_client():
    if not settings.gemini_api_key:
        return None
    return genai.Client(api_key=settings.gemini_api_key)


MATCH_PROMPT = """You are a lead fact-checker and senior journalist evaluating search evidence against an extracted claim.

USER'S CLAIM:
{claim}

CLAIM CONTEXT (HINDI / TAMIL / OTHER):
Hindi: {claim_hindi}
Tamil: {claim_tamil}

NEWS AND FACT-CHECK ARTICLES FOUND:
{fact_checks}

Instructions:
1. For each article, evaluate semantic equivalence:
   - Does this article address the EXACT SAME event, person, statement, or quote?
   - Evaluate: Who, What happened, When, Where, How much, To whom?
   - Assign match_confidence (0.0 to 1.0). > 0.70 means direct match.
2. Determine overall verdict:
   - "likely_true": Credible news coverage or fact-checks confirm this event/claim actually occurred or is factual.
   - "likely_false": Credible fact-checks or news coverage explicitly debunk or refute this claim as false/fabricated.
   - "unverifiable": No confident matches (>0.50), weak evidence, or conflicting reports.

CRITICAL RULES:
- If multiple mainstream news outlets report the exact event as reported news, return "likely_true" with high confidence (0.85-0.95).
- If credible fact-checks explicitly debunk the claim, return "likely_false" with high confidence (0.85-0.95).
- Absence of evidence means UNVERIFIABLE.
"""


async def match_and_summarise(
    raw_text: str,
    claim_info: dict,
    matches: list[FactCheckMatch],
) -> ClaimAnalysis:
    """
    Semantically matches extracted claim against retrieved fact-checks & news asynchronously with retries.
    Returns a complete ClaimAnalysis object.
    """
    claim = claim_info.get("claim", raw_text[:200])
    claim_hindi = claim_info.get("claim_hindi", "")
    claim_tamil = claim_info.get("claim_tamil", "")
    lang = claim_info.get("language", LanguageCode.EN)

    claim_type_str = claim_info.get("claim_type", "other").lower()
    try:
        claim_type = ClaimType(claim_type_str)
    except ValueError:
        claim_type = ClaimType.OTHER

    entities = claim_info.get("entities", {})

    if not matches:
        return ClaimAnalysis(
            raw_text=raw_text,
            extracted_claim=claim,
            claim_type=claim_type,
            language=lang,
            is_checkable=claim_info.get("is_checkable", True),
            entities_people=entities.get("people", []),
            entities_places=entities.get("places", []),
            entities_dates=entities.get("dates", []),
            matches=[],
            text_verdict=Verdict.UNVERIFIABLE,
            text_verdict_confidence=0.5,
            no_match_reason="No matching news coverage or fact-checks found. "
                            "This claim could not be independently verified.",
        )

    fc_summary = "\n".join([
        f"[{i}] SOURCE: {m.source_name}\n"
        f"    TITLE: {m.original_claim[:150]}\n"
        f"    SNIPPET: {m.snippet[:200]}\n"
        f"    URL: {m.source_url}"
        for i, m in enumerate(matches[:8])
    ])

    client = _get_client()
    if not client:
        return ClaimAnalysis(
            raw_text=raw_text,
            extracted_claim=claim,
            claim_type=claim_type,
            language=lang,
            matches=matches,
            text_verdict=Verdict.UNVERIFIABLE,
            text_verdict_confidence=0.5,
            error="No GEMINI_API_KEY configured",
        )

    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=MATCH_PROMPT.format(
                    claim=claim,
                    claim_hindi=claim_hindi,
                    claim_tamil=claim_tamil,
                    fact_checks=fc_summary
                ),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ClaimMatchSchema,
                    temperature=0.1,
                    max_output_tokens=600,
                ),
            )

            match_res: ClaimMatchSchema = getattr(response, "parsed", None)
            if not match_res:
                break

            updated_matches = []
            for m_result in match_res.matches:
                idx = m_result.index
                if idx < len(matches):
                    conf = float(m_result.match_confidence)
                    if conf >= 0.40:
                        m = matches[idx].model_copy(update={
                            "match_confidence": conf,
                            "fact_check_verdict": m_result.verdict_extracted,
                            "reason": m_result.verdict_explanation,
                        })
                        updated_matches.append(m)

            updated_matches.sort(key=lambda x: x.match_confidence, reverse=True)
            best_match = updated_matches[0] if updated_matches else (matches[0] if matches else None)

            verdict_str = match_res.overall_verdict.lower()
            if verdict_str in ["likely_false", "false", "fake", "misleading"]:
                verdict = Verdict.LIKELY_FALSE
            elif verdict_str in ["likely_true", "true", "correct"]:
                verdict = Verdict.LIKELY_TRUE
            elif verdict_str == "misleading_context":
                verdict = Verdict.MISLEADING_CONTEXT
            else:
                verdict = Verdict.UNVERIFIABLE

            confidence = float(match_res.overall_confidence)
            if confidence < 0.50 or not updated_matches:
                verdict = Verdict.UNVERIFIABLE

            return ClaimAnalysis(
                raw_text=raw_text,
                extracted_claim=claim,
                claim_type=claim_type,
                language=lang,
                is_checkable=claim_info.get("is_checkable", True),
                entities_people=entities.get("people", []),
                entities_places=entities.get("places", []),
                entities_dates=entities.get("dates", []),
                matches=updated_matches or matches,
                best_match=best_match,
                text_verdict=verdict,
                text_verdict_confidence=confidence,
                no_match_reason=match_res.reasoning if verdict == Verdict.UNVERIFIABLE else None,
            )

        except Exception as e:
            if ("429" in str(e) or "503" in str(e)) and attempt < 2:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            log.error("semantic_claim_matching_failed", error=str(e))
            return ClaimAnalysis(
                raw_text=raw_text,
                extracted_claim=claim,
                claim_type=claim_type,
                language=lang,
                matches=matches,
                text_verdict=Verdict.UNVERIFIABLE,
                text_verdict_confidence=0.5,
                error=str(e),
            )

    return ClaimAnalysis(
        raw_text=raw_text,
        extracted_claim=claim,
        claim_type=claim_type,
        language=lang,
        matches=matches,
        text_verdict=Verdict.UNVERIFIABLE,
        text_verdict_confidence=0.5,
        no_match_reason="Automated matching fallback.",
    )
