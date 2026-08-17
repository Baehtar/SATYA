"""
src/pipelines/text/claim_matcher.py — Semantic claim matching using Gemini.
Owned by: Person 2

Matches the user's extracted claim against retrieved fact-checks,
handles paraphrasing and cross-language matching (Hindi claim ↔ English fact-check).
"""
import json
import structlog
import google.generativeai as genai
from src.config import settings
from src.models.schemas import ClaimAnalysis, FactCheckMatch, Verdict, ClaimType

log = structlog.get_logger(__name__)
_model = genai.GenerativeModel(settings.gemini_model)

MATCH_PROMPT = """You are an expert fact-checker. Your job is to determine if a fact-check article matches a given claim.

USER'S CLAIM (extracted from a WhatsApp forward):
{claim}

FACT-CHECK ARTICLES FOUND:
{fact_checks}

For each fact-check article, rate how well it matches the user's claim on a scale of 0.0 to 1.0.
Also extract the verdict from each article.

Respond ONLY with a valid JSON object:
{{
  "matches": [
    {{
      "index": 0,
      "match_confidence": 0.95,
      "verdict_extracted": "false|true|misleading|out of context|satire|unverifiable",
      "verdict_explanation": "One sentence explaining what the fact-check found"
    }}
  ],
  "overall_verdict": "likely_true|likely_false|unverifiable|misleading_context",
  "overall_confidence": 0.85,
  "reasoning": "2-3 sentences explaining the overall verdict"
}}

Rules:
- Match confidence > 0.75 means the fact-check directly addresses this claim
- Match confidence 0.5–0.75 means the fact-check is related but not exact
- Match confidence < 0.5 means the fact-check is tangentially related
- If no matches score > 0.5, set overall_verdict to "unverifiable"
- Never set overall_verdict to "likely_false" unless a credible fact-check explicitly debunks it
"""


async def match_and_summarise(
    raw_text: str,
    claim_info: dict,
    matches: list[FactCheckMatch],
) -> ClaimAnalysis:
    """
    Semantically matches the claim against retrieved fact-checks.
    Returns a complete ClaimAnalysis.
    """
    claim = claim_info.get("claim", raw_text[:200])

    # Map claim type
    claim_type_map = {
        "political": ClaimType.POLITICAL,
        "health": ClaimType.HEALTH,
        "disaster": ClaimType.DISASTER,
        "religious": ClaimType.RELIGIOUS,
        "financial": ClaimType.FINANCIAL,
    }
    claim_type = claim_type_map.get(claim_info.get("claim_type", "other"), ClaimType.OTHER)

    entities = claim_info.get("entities", {})

    # If no matches found, return UNVERIFIABLE
    if not matches:
        return ClaimAnalysis(
            raw_text=raw_text,
            extracted_claim=claim,
            claim_type=claim_type,
            is_checkable=claim_info.get("is_checkable", True),
            entities_people=entities.get("people", []),
            entities_places=entities.get("places", []),
            entities_dates=entities.get("dates", []),
            matches=[],
            text_verdict=Verdict.UNVERIFIABLE,
            text_verdict_confidence=0.8,
            no_match_reason="No relevant fact-checks found in our sources (PIB, AltNews, BOOM). "
                           "This doesn't mean the claim is false — it may be too recent or too local to have been fact-checked.",
        )

    # Build fact-check summary for the prompt
    fc_summary = "\n".join([
        f"[{i}] SOURCE: {m.source_name}\n"
        f"    TITLE: {m.original_claim[:150]}\n"
        f"    SNIPPET: {m.snippet[:150]}\n"
        f"    URL: {m.source_url}"
        for i, m in enumerate(matches[:6])  # limit to top 6
    ])

    try:
        response = await _model.generate_content_async(
            MATCH_PROMPT.format(claim=claim, fact_checks=fc_summary),
            generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=512),
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)

    except Exception as e:
        log.error("claim_matching_failed", error=str(e))
        return ClaimAnalysis(
            raw_text=raw_text,
            extracted_claim=claim,
            claim_type=claim_type,
            matches=matches,
            text_verdict=Verdict.UNVERIFIABLE,
            text_verdict_confidence=0.5,
            error=str(e),
        )

    # Update match confidence scores from Gemini
    updated_matches = []
    for m_result in result.get("matches", []):
        idx = m_result.get("index", 0)
        if idx < len(matches):
            m = matches[idx].model_copy(update={
                "match_confidence": m_result.get("match_confidence", 0.0),
                "fact_check_verdict": m_result.get("verdict_extracted", ""),
            })
            if m.match_confidence >= 0.5:
                updated_matches.append(m)

    # Sort by confidence
    updated_matches.sort(key=lambda x: x.match_confidence, reverse=True)
    best = updated_matches[0] if updated_matches else None

    # Map verdict string to enum
    verdict_map = {
        "likely_true": Verdict.LIKELY_TRUE,
        "likely_false": Verdict.LIKELY_FALSE,
        "unverifiable": Verdict.UNVERIFIABLE,
        "misleading_context": Verdict.MISLEADING_CONTEXT,
    }
    verdict = verdict_map.get(result.get("overall_verdict", "unverifiable"), Verdict.UNVERIFIABLE)

    return ClaimAnalysis(
        raw_text=raw_text,
        extracted_claim=claim,
        claim_type=claim_type,
        is_checkable=claim_info.get("is_checkable", True),
        entities_people=entities.get("people", []),
        entities_places=entities.get("places", []),
        entities_dates=entities.get("dates", []),
        matches=updated_matches,
        best_match=best,
        text_verdict=verdict,
        text_verdict_confidence=result.get("overall_confidence", 0.5),
        no_match_reason=None if updated_matches else "No confident match found in fact-check sources.",
    )
