import logging
import json
import re
from typing import Dict, Any
from google import genai
from src.config import settings
from src.models.schemas import ClaimType

logger = logging.getLogger(__name__)


async def extract_claim(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {
            "claim": "",
            "entities": [],
            "claim_type": ClaimType.OTHER,
            "is_checkable": False,
        }

    clean_text = text.strip()

    # Try Gemini 2.5 Flash if API key is provided
    if settings.gemini_api_key:
        try:
            client = genai.Client(api_key=settings.gemini_api_key)

            prompt = """You are a fact-check assistant. Given a forwarded message, extract:
1. The core factual claim (one sentence, in English)
2. Named entities (people, places, events, dates)
3. Claim type: POLITICAL, HEALTH, DISASTER, RELIGIOUS, FINANCIAL, OTHER
4. Is this a verifiable factual claim? (true/false)

Respond in JSON format only.
Example: {"claim": "A new virus was discovered in NY.", "entities": ["NY"], "claim_type": "HEALTH", "is_checkable": true}"""

            response = client.models.generate_content(
                model=settings.gemini_model, contents=prompt + "\n\nMessage:\n" + clean_text
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            result = json.loads(raw)

            try:
                claim_type = ClaimType(result.get("claim_type", "OTHER").lower())
            except Exception:
                claim_type = ClaimType.OTHER

            return {
                "claim": result.get("claim", clean_text[:200]),
                "entities": result.get("entities", []),
                "claim_type": claim_type,
                "is_checkable": result.get("is_checkable", True),
            }
        except Exception as e:
            logger.error(f"Gemini claim extraction failed: {e}")

    # Fallback NLP & Rule-based extraction (works without API key)
    # Extract entities via regex (capitalized words, numbers, key terms)
    entities = list(set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", clean_text)))

    # Classify claim type via heuristics
    lowered = clean_text.lower()
    claim_type = ClaimType.OTHER

    if any(k in lowered for k in ["modi", "gandhi", "bjp", "congress", "pm", "government", "election", "pib", "minister", "scheme", "15 lakh"]):
        claim_type = ClaimType.POLITICAL
    elif any(k in lowered for k in ["virus", "covid", "vaccine", "doctor", "cure", "health", "hospital", "cancer", "disease", "who"]):
        claim_type = ClaimType.HEALTH
    elif any(k in lowered for k in ["cyclone", "flood", "earthquake", "tsunami", "rain", "disaster", "storm", "landslide"]):
        claim_type = ClaimType.DISASTER
    elif any(k in lowered for k in ["bank", "rbi", "rupee", "₹", "note", "tax", "money", "loan", "atm"]):
        claim_type = ClaimType.FINANCIAL
    elif any(k in lowered for k in ["temple", "mosque", "church", "religion", "hindu", "muslim", "god"]):
        claim_type = ClaimType.RELIGIOUS

    # Extract first strong sentence as the core claim
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", clean_text) if len(s.strip()) > 10]
    core_claim = sentences[0] if sentences else clean_text[:200]

    return {
        "claim": core_claim,
        "entities": entities,
        "claim_type": claim_type,
        "is_checkable": True,  # Mark as checkable so pipeline proceeds to search fact check DB/sites!
    }
