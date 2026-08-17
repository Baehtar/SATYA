# Satya 🔍 — AI Forward-Checker for Viral Misinformation

> **Hackathon submission for PS-S03 — Satya**  
> Forward a suspicious image, text, or voice note → receive a credibility verdict in < 60 seconds, in Hindi + English.

---

## What is Satya?

Satya is a Telegram bot that acts as an AI-powered fact-checker inside your messaging flow. Forward any suspicious content and get back a **plain-language credibility card** with:

- 🟢 Likely True / 🔴 Likely False / 🟡 Unverifiable / 🤖 AI Generated / ✂️ Manipulated
- A confidence level (with honest uncertainty)
- A two-line explanation a grandparent can understand — in Hindi + English
- Source links from PIB Fact Check, AltNews, and BOOM

---


## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/satya.git
cd satya
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ **GPU machine only**: PyTorch CUDA will be installed automatically. For CPU-only machines, replace the torch line in requirements.txt with the CPU version.

### 4. Set up environment variables
```bash
cp .env.example .env
# Fill in your API keys in .env
```

**Required keys:**
| Key | Where to get it |
|---|---|
| `TELEGRAM_BOT_TOKEN` | [@BotFather](https://t.me/BotFather) on Telegram |
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) |
| `SERPAPI_KEY` | [serpapi.com](https://serpapi.com) (free: 100 searches/month) |
| `GOOGLE_FACTCHECK_API_KEY` | [Google Cloud Console](https://console.cloud.google.com/) → Fact Check Tools API (free) |

### 5. Run the bot
```bash
python -m src.bot.telegram_bot
```

### 6. Run the dashboard (optional)
```bash
uvicorn src.dashboard.app:app --port 8080 --reload
# Open http://localhost:8080
```

---

## Project Structure

```
satya/
├── src/
│   ├── config.py                    # Centralised settings (all teammates import from here)
│   ├── models/
│   │   └── schemas.py               # 🚨 SHARED DATA CONTRACTS — read before coding
│   ├── bot/
│   │   ├── telegram_bot.py          # Main bot entry point
│   │   ├── router.py                # Message type detection + pipeline dispatch
│   │   └── formatter.py             # Verdict card → Telegram HTML
│   ├── pipelines/
│   │   ├── image/                   # Person 1
│   │   │   ├── pipeline.py          # Orchestrator (runs all image checks in parallel)
│   │   │   ├── ai_detector.py       # HuggingFace ViT AI-generation detection
│   │   │   ├── manipulation_detector.py  # ELA + EXIF + noise analysis
│   │   │   └── reverse_search.py    # SerpAPI Google Lens + date comparison
│   │   ├── text/                    # Person 2
│   │   │   ├── pipeline.py          # Orchestrator
│   │   │   ├── claim_extractor.py   # Gemini: extract core claim from forward
│   │   │   ├── fact_check_search.py # PIB + AltNews + BOOM + Google Fact Check API
│   │   │   └── claim_matcher.py     # Semantic matching + verdict extraction
│   │   ├── audio/                   # Person 4 (stretch)
│   │   │   └── voice_analyzer.py    # Whisper STT + voice clone detection
│   │   └── screenshot/              # Person 4 (stretch)
│   │       └── chyron_detector.py   # News channel + OCR + tampering check
│   ├── verdict/                     # Person 3
│   │   ├── aggregator.py            # Merge all pipeline outputs
│   │   ├── confidence.py            # Calibrated confidence scoring (key judging criterion)
│   │   └── card_generator.py        # Gemini: write grandparent-readable card in EN + HI
│   ├── db/
│   │   ├── models.py                # SQLModel tables
│   │   └── database.py              # Async session management
│   └── dashboard/
│       ├── app.py                   # FastAPI trend dashboard API
│       └── static/index.html        # Dashboard frontend
├── tests/
│   └── test_judging_set.py          # 8-item judging evaluation runner
├── docs/
│   └── (architecture + blind spots docs go here)
├── .env.example                     # Copy to .env and fill in keys
├── .gitignore
└── requirements.txt
```

---

## Running Tests

```bash
# Full judging set evaluation
python tests/test_judging_set.py

# Pytest
pytest tests/ -v

# Single pipeline test
pytest tests/test_judging_set.py -k "TC-01" -v
```

---

## Architecture

See `docs/architecture.md` for the full pipeline diagram.

**Summary:**
1. Telegram message → **Router** (classifies type)
2. Image → AI detector ‖ Manipulation detector ‖ Reverse image search (parallel)
3. Text → Claim extractor → Fact-check search (PIB/AltNews/BOOM/Google) → Claim matcher
4. Voice → Whisper STT → Text pipeline ‖ Voice clone detector
5. All results → **Evidence Aggregator** → **Confidence Calibrator** → **Card Generator** (Gemini)
6. Verdict card → Telegram (HTML + inline buttons)

---

## Supported Input Types

| Type | Pipeline | Stretch |
|---|---|---|
| 📷 Image | AI detection + Manipulation + Reverse search | No |
| 💬 Text forward | Claim extraction + Fact-check search | No |
| 📷+💬 Image + caption | Both in parallel | No |
| 🎤 Voice note | Whisper STT + Voice clone detection | Yes |
| 📰 News screenshot | Channel detection + Chyron OCR + Tampering | Yes |

---

## Key Design Decisions

1. **UNVERIFIABLE is a first-class verdict** — the system never falls back to LIKELY_FALSE when evidence is insufficient. A confident "we don't know" is better than a wrong verdict.

2. **Parallel pipelines** — image and text pipelines run concurrently for mixed-content messages, keeping latency under 30s even for complex forwards.

3. **Calibrated confidence** — confidence scores are explicitly weighted: single weak signal → LOW confidence; multiple corroborating signals → HIGH confidence. Conflicting signals reduce confidence rather than defaulting to one verdict.

4. **Honest fallbacks** — if any pipeline fails (timeout, API error), the system returns partial results with a transparency note, rather than crashing or returning a false verdict.

---

## Blind Spots

See `docs/blind_spots.md` for the full list. Key ones:
- Video deepfakes (image-only pipeline)
- First-time misinformation (no fact-check exists yet → UNVERIFIABLE by design ✓)
- Highly localised claims not covered by national fact-checkers
- Satire that isn't clearly labelled
- Bleeding-edge AI image models the detector hasn't seen

---

## Latency Targets (GPU)

| Forward Type | Target |
|---|---|
| Text only | 5–10s |
| Image only | 8–20s |
| Image + caption | 10–25s |
| Voice note | 12–25s |
| News screenshot | 8–18s |
