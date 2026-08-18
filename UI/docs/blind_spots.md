# Satya Web UI — Blind Spots & Failure Modes

What this front-end can and cannot see. Scoped to the signals the shared backend
(`services/ml_service.py`) actually computes — the web UI adds no detection of its own.

## 1. Signals available to a web check

| Signal | Source | Notes |
|---|---|---|
| AI-generated image | HuggingFace `Organika/sdxl-detector` | Single classifier. Needs `HF_API_KEY`. |
| Text in an image | Gemini Vision OCR (`pytesseract` fallback) | Needs `GEMINI_API_KEY`. |
| Claim verification | PIB · Alt News · BOOM · Google FactCheck API · Google News RSS | Only Google News RSS works with no key at all. |
| Voice → text | Gemini speech-to-text | Needs `GEMINI_API_KEY`, plus ffmpeg for WebM recordings. |

**Not wired into this path:** ELA/noise manipulation forensics, EXIF anomaly checks,
and Google Lens reverse image search. That code exists in `src/pipelines/image/` but
needs torch/transformers and a SerpAPI key, so the web UI does not call it. Recycled
old photos and Photoshop edits therefore go undetected here — the card never claims
otherwise, and `image_flags` only ever reports what was measured.

## 2. Image AI-detection limits

- **Platform compression.** WhatsApp/Telegram re-encoding alters the high-frequency
  statistics the detector keys on, in both directions.
- **Newer generators.** Models released after the classifier's training data (e.g.
  recent Midjourney/FLUX versions) can pass as authentic.
- **A synthetic image is not a false claim.** An AI illustration attached to a true
  report does not make the report false. The verdict follows the claim; the AI signal
  stays in `image_flags`.

## 3. Claim understanding

- **Satire and hyperbole** can be read as factual assertions.
- **Cross-lingual paraphrase.** Hinglish/Tanglish transliteration and regional
  languages obscure entity names, so retrieval can miss an existing fact-check.
- **Absence of evidence is not evidence.** No match returns `unverifiable`, never
  `likely_false`, and the card says so explicitly in the disclaimer.

## 4. Retrieval and rate limits

- Fact-check sites are queried by **live search and HTML scraping**; a layout change
  or a block silently reduces a source to zero results. Google News RSS is the most
  reliable path and needs no key.
- SerpAPI and Google FactCheck free tiers can be exhausted during a breaking-news
  spike; adapters fail independently and the check continues with fewer sources.
- There is **no offline fact-check database.** With no network and no keys, a check
  returns `unverifiable` rather than a cached answer.

## 5. Degradation is visible, not silent

`GET /api/health` reports which keys are configured. When a stage cannot run, its
explanation says so (e.g. "HF API key missing"), the confidence stays low, and the
verdict stays `unverifiable` — the UI never converts a missing capability into a
confident answer.
