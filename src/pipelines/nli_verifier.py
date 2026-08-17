"""
src/pipelines/nli_verifier.py — Multilingual Natural Language Inference (NLI) Verifier.
Evaluates semantic relationship between extracted claim and evidence: ENTAILMENT | CONTRADICTION | NEUTRAL.
"""
import asyncio
import structlog
from typing import Dict, Any
from google import genai
from google.genai import types
from src.config import settings
from src.models.schemas import ClaimMatchSchema, MatchItemSchema

log = structlog.get_logger(__name__)


def _get_client():
    if not settings.gemini_api_key:
        return None
    return genai.Client(api_key=settings.gemini_api_key)


NLI_PROMPT = """You are a precision Natural Language Inference (NLI) classifier for fact-checking.

CLAIM:
"{claim}"

EVIDENCE:
"{evidence}"

Instructions:
Classify the relationship of the EVIDENCE with respect to the CLAIM into exactly ONE of:
- "ENTAILMENT": The evidence directly confirms or proves the claim is TRUE.
- "CONTRADICTION": The evidence directly refutes, debunks, or proves the claim is FALSE.
- "NEUTRAL": The evidence is unrelated, neutral, or does not provide sufficient info to confirm or deny.

Respond ONLY in valid JSON format:
{{
  "nli_label": "ENTAILMENT|CONTRADICTION|NEUTRAL",
  "nli_score": 0.95,
  "explanation": "1 sentence explanation."
}}
"""


async def verify_claim_nli(claim: str, evidence_snippet: str) -> Dict[str, Any]:
    """
    Evaluates NLI classification for a single claim-evidence pair.
    Returns dict with nli_label, nli_score, and explanation.
    """
    if not claim.strip() or not evidence_snippet.strip():
        return {
            "nli_label": "NEUTRAL",
            "nli_score": 0.5,
            "explanation": "Empty claim or evidence provided."
        }

    client = _get_client()
    if not client:
        return {
            "nli_label": "NEUTRAL",
            "nli_score": 0.5,
            "explanation": "NLI client unconfigured."
        }

    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=NLI_PROMPT.format(claim=claim[:500], evidence=evidence_snippet[:800]),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=200,
            ),
        )

        import json
        raw = response.text.strip()
        start_idx = raw.find('{')
        end_idx = raw.rfind('}')
        if start_idx != -1 and end_idx != -1:
            raw = raw[start_idx:end_idx + 1]

        data = json.loads(raw)
        label = data.get("nli_label", "NEUTRAL").upper()
        if label not in ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]:
            label = "NEUTRAL"

        score = float(data.get("nli_score", 0.5))

        return {
            "nli_label": label,
            "nli_score": score,
            "explanation": data.get("explanation", "")
        }

    except Exception as e:
        log.warning("nli_verification_failed", error=str(e))
        return {
            "nli_label": "NEUTRAL",
            "nli_score": 0.5,
            "explanation": f"NLI error: {e}"
        }
