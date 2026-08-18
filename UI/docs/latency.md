# Satya Web UI — Latency

## Budgets (enforced in code)

These are the real timeouts, from `src/config.py` and `UI/src/server.py`:

| Stage | Budget | Where |
|---|---|---|
| Text pipeline (extract → search → match) | 45s | `settings.text_pipeline_timeout` |
| Whole check, end to end | 58s | `settings.total_timeout`, enforced in `_run_analysis` |
| Browser dead-socket guard | 90s | `frontend/js/api.js` |

A check that exceeds the budget sends a `failed` event with an explanation — it never
hangs the stepper.

## What the time is actually spent on

| Work | Notes |
|---|---|
| Claim extraction | 1 Gemini call. Falls back to a truncation heuristic without a key. |
| Evidence retrieval | 4–5 sources in parallel (`asyncio.gather`): Google FactCheck API, Alt News, BOOM, Google News RSS, and PIB for government/political/financial claims. Slowest source sets the floor. |
| Semantic matching | 1 Gemini call over up to 8 retrieved articles. |
| AI-image detection | 1 HuggingFace Inference API call (`Organika/sdxl-detector`), run concurrently with OCR. |
| OCR | 1 Gemini Vision call; `pytesseract` locally if installed. |
| Voice transcription | 1 Gemini call, plus an ffmpeg transcode when the browser records WebM. |
| Verdict card | 1 Gemini call for the bilingual explanation; curated bilingual text without a key. |

## Measured

Locally, with no API keys configured (so retrieval ran but every Gemini/HF call was
skipped), a text check completed end to end in **~3.9s**, dominated by the parallel
source search. Expect meaningfully more with keys configured — each Gemini call adds
its own round trip, which is why the total budget is 58s rather than the observed few
seconds.

Repeated claims are served from the in-process search cache in
`src/pipelines/text/fact_check_search.py` (SHA-256 of the normalised claim), so a
re-check of the same text skips retrieval entirely.

## Perceived latency

Progress is streamed over SSE as each stage starts, so the stepper moves within
milliseconds of a stage beginning rather than after the whole check. Stages a flow
does not need (e.g. no source search for a photo with no readable text) are marked
*skipped* instead of spinning forever.
