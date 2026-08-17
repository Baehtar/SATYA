"""
src/pipelines/text/claim_extractor.py — Core claim extraction using Gemini.
Owned by: Person 2

Extracts the central verifiable factual claim from a rambling WhatsApp forward.
Works in Hindi, English, and Hinglish.
"""
import json
import structlog
import google.generativeai as genai
from src.config import settings

log = structlog.get_logger(__name__)

genai.configure(api_key=settings.gemini_api_key)
_model = genai.GenerativeModel(settings.gemini_model)

CLAIM_EXTRACTION_PROMPT = """You are an expert fact-checker specialising in Indian misinformation.

Analyse the following forwarded WhatsApp message and extract structured information.
The message may be in Hindi, English, or a mix (Hinglish).

FORWARD TEXT:
---
{text}
---

Respond ONLY with a valid JSON object in this exact format:
{{
  "claim": "The single core factual claim in plain English (1-2 sentences max). If in Hindi, translate.",
  "claim_hindi": "The single core factual claim in plain Hindi.",
  "claim_type": "political|health|disaster|religious|financial|other",
  "is_checkable": true,
  "checkability_reason": "Why this is/isn't a checkable claim (1 sentence)",
  "entities": {{
    "people": ["list of named people mentioned"],
    "places": ["list of named places"],
    "dates": ["list of dates or time references mentioned"],
    "organisations": ["list of organisations mentioned"]
  }},
  "urgency_language": true,
  "urgency_phrases": ["list of fear/urgency phrases like 'share now', 'breaking', 'viral']"
}}

Rules:
- is_checkable = false ONLY for pure opinions or satire
- Always extract a claim even from vague text; summarise the main fear/assertion
- entities should be specific names, not generic descriptions
- urgency_language = true if the forward uses phrases like "share karo", "forward करो", "breaking", "सच्चाई"
"""


async def extract_claim(text: str) -> dict:
    """
    Extracts the core factual claim from a WhatsApp forward.
    Returns a dict with: claim, claim_hindi, claim_type, is_checkable, entities, etc.
    """
    log.info("extracting_claim", text_len=len(text))

    prompt = CLAIM_EXTRACTION_PROMPT.format(text=text[:3000])  # cap at 3000 chars

    try:
        response = await _model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,   # low temp for structured extraction
                max_output_tokens=512,
            ),
        )
        raw = response.text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)
        log.info("claim_extracted", claim=result.get("claim", "")[:80])
        return result

    except json.JSONDecodeError as e:
        log.warning("claim_json_parse_failed", error=str(e), raw=raw[:200])
        # Fallback: return the text itself as the claim
        return {
            "claim": text[:200],
            "claim_hindi": text[:200],
            "claim_type": "other",
            "is_checkable": True,
            "entities": {"people": [], "places": [], "dates": [], "organisations": []},
            "urgency_language": False,
            "urgency_phrases": [],
        }
    except Exception as e:
        log.error("claim_extraction_failed", error=str(e))
        raise
