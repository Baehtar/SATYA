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
1. For each article, evaluate semantic relevance:
   - Does this article address this specific claim, rumor, viral forward, or hoax (even if the headline refutes or debunks it)?
   - Articles from fact-checkers (PIB, Alt News, BOOM, Google FactCheck) or news organizations discussing and refuting this viral claim are direct matches (match_confidence 0.75 - 0.95).
2. Determine overall verdict:
   - "likely_true": Credible news coverage, official government notices, or fact-checks confirm this event/claim actually occurred or is true.
   - "likely_false": Credible fact-checks, official government statements (PIB), or investigative reporting confirm this claim is false, fabricated, a known hoax, a fake scheme, or a viral rumor.
   - "unverifiable": Pure personal hearsay, hyper-local rumors with zero public coverage, or ambiguous conflicting reports.

CRITICAL RULES:
- If search results include fact-checks or reports debunking the viral rumor (e.g., vaccine microchips, fake cash deposits, fake PM stipend, ATM closures), return "likely_false" with confidence 0.85-0.95.
- If mainstream news outlets report the exact event as verified factual news (e.g., historical facts, official laws), return "likely_true" with confidence 0.85-0.95.
- Genuine absence of any coverage for local/vague personal claims means "unverifiable".
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
        for i, m in enumerate(matches[:6])
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
                    max_output_tokens=2000,
                ),
            )

            match_res: ClaimMatchSchema = getattr(response, "parsed", None)
            if not match_res and getattr(response, "text", None):
                try:
                    import json
                    raw_json = response.text.strip()
                    if raw_json.startswith("```json"):
                        raw_json = raw_json[7:]
                    if raw_json.startswith("```"):
                        raw_json = raw_json[3:]
                    if raw_json.endswith("```"):
                        raw_json = raw_json[:-3]
                    data = json.loads(raw_json.strip())
                    match_res = ClaimMatchSchema(**data)
                except Exception as parse_err:
                    log.warning("claim_matcher_json_parse_fallback_failed", error=str(parse_err))

            if not match_res:
                log.warning("claim_matcher_no_parsed_response", raw_text=getattr(response, "text", "")[:200])
                continue

            updated_matches = []
            for m_result in match_res.matches:
                idx = m_result.index
                if idx < len(matches):
                    conf = float(m_result.match_confidence)
                    if conf >= 0.25:
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
            if confidence < 0.50 or (not updated_matches and not matches):
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
            err_msg = str(e).lower()
            if ("429" in err_msg or "503" in err_msg or "quota" in err_msg or "resource_exhausted" in err_msg) and attempt < 3:
                wait_time = 2.5 * (attempt + 1)
                log.info("gemini_matcher_rate_limit_retry", wait_seconds=wait_time)
                await asyncio.sleep(wait_time)
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
