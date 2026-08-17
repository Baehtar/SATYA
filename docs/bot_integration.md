# Integration Guide for Telegram Bot

## How to call the Satya backend from your bot

Install the client (already in requirements.txt — just import it):

```python
from src.api.client import SatyaClient

client = SatyaClient(base_url="http://localhost:8000")
```

---

## Endpoint Quick Reference

| What you have | Call |
|---|---|
| Text forward | `await client.check_text(text)` |
| Photo | `await client.check_image(bytes, filename, caption)` |
| Voice note | `await client.check_audio(bytes, filename)` |
| News screenshot | `await client.check_screenshot(bytes)` |
| Don't care — just check it | `await client.check_auto(text=..., file_bytes=..., filename=...)` |

All calls return a `VerdictResult` dataclass.

---

## VerdictResult fields

```python
result.verdict              # "likely_true" | "likely_false" | "unverifiable" | "ai_generated" | "manipulated" | "misleading_context"
result.confidence_score     # float 0.0–1.0
result.confidence_level     # "high" | "moderate" | "low"
result.explanation_english  # 2-3 plain English sentences
result.explanation_hindi    # 2-3 plain Hindi sentences
result.signals_used         # list of human-readable signals
result.sources              # list of dicts: {source_name, source_url, verdict, snippet}
result.source_urls          # list of URLs
result.blind_spot_warning   # str or None — honest limitation caveat
result.total_latency_ms     # int — end-to-end ms

# Convenience
result.verdict_emoji        # 🟢 / 🔴 / 🟡 / 🤖 / ✂️ / 🟠
result.confidence_bar       # "██████░░░░" (10-char bar)
result.is_flagged           # True if verdict is false/ai/manipulated/misleading
```

---

## Minimal bot integration example

```python
from telegram import Update
from telegram.ext import ContextTypes
from src.api.client import SatyaClient

client = SatyaClient("http://localhost:8000")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    thinking = await msg.reply_text("🔍 Checking...")

    try:
        if msg.photo:
            photo = msg.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            img_bytes = await file.download_as_bytearray()
            result = await client.check_image(
                bytes(img_bytes),
                caption=msg.caption,
            )
        elif msg.voice:
            file = await context.bot.get_file(msg.voice.file_id)
            audio_bytes = await file.download_as_bytearray()
            result = await client.check_audio(bytes(audio_bytes), "voice.ogg")
        else:
            result = await client.check_text(msg.text or "")

        # Format and send
        text = (
            f"{result.verdict_emoji} <b>{result.verdict.replace('_', ' ').upper()}</b>\n"
            f"<code>{result.confidence_bar}</code> {int(result.confidence_score*100)}%\n\n"
            f"🔍 {result.explanation_english}\n\n"
            f"🔍 {result.explanation_hindi}\n"
        )
        if result.sources:
            text += f"\n📎 <a href='{result.sources[0]['source_url']}'>{result.sources[0]['source_name']}</a>"

        await thinking.edit_text(text, parse_mode="HTML", disable_web_page_preview=True)

    except Exception as e:
        await thinking.edit_text(f"❌ Error: {e}")
```

---

## Starting the backend

```bash
# From the repo root, with .env filled in:
uvicorn src.api.main:app --port 8000 --reload

# Verify it's up:
curl http://localhost:8000/health
# {"status":"ok","service":"satya-api","version":"1.0.0"}

# Interactive docs:
# http://localhost:8000/docs
```

---

## Running both bot + backend together

```bash
# Terminal 1 — backend
uvicorn src.api.main:app --port 8000

# Terminal 2 — bot
python -m src.bot.telegram_bot
```
