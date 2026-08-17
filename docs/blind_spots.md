# Satya — Documented Blind Spots

Honest limitations of the system. Required deliverable for judging.

## What the system CANNOT catch

### 1. First-time / Novel Misinformation
- If a claim has never been fact-checked by PIB, AltNews, or BOOM, the verdict will be `UNVERIFIABLE`
- **This is by design** — we never guess FALSE without evidence
- Affects: breaking news misinformation in the first few hours

### 2. Highly Localised Claims
- "The XYZ hospital in Nagpur is shutting down" — not covered by national fact-checkers
- Verdict: `UNVERIFIABLE` (correct and honest)

### 3. Video Deepfakes
- We process images only. Video clips sent as files are not analysed
- Workaround: extract a keyframe manually and submit that

### 4. Bleeding-Edge AI Image Models
- The HuggingFace detector (`umm-maybe/AI-image-detector`) was trained on a specific set of generators
- Images from the newest models (released after the training data) may score low
- Workaround: ELA + noise analysis provides a secondary signal

### 5. Satire Without Labels
- Satirical content (The Onion-style) may be extracted as a factual claim
- The claim extractor looks for `is_checkable=false` but this relies on Gemini identifying satire
- Adversarial satire designed to look real will be harder to catch

### 6. Multi-Claim Forwards
- A single forward with 5 different claims: we extract the **primary** claim only
- The adversarial test ADV-02 tests this — mixed true+false in one message

### 7. Audio in Non-Hindi/English Languages
- Whisper handles Hindi/English/Hinglish well
- Regional languages (Tamil, Bengali, Telugu, Marathi) will transcribe with lower accuracy
- The fact-check search is English/Hindi only

### 8. Encrypted/Forwarded-Many-Times Context
- We cannot access the original sender, forward chain, or group context
- A legitimate photo forwarded with false context requires the reverse image search to catch it

### 9. Compound Manipulations
- An AI-generated image that is **also** made to look like a real news screenshot
- The two detectors run independently — the verdict engine may not combine them optimally in edge cases

### 10. Real-Time Prices / Stock Claims
- "Onion price is ₹200/kg today" — these change daily; fact-check databases don't cover them
- Verdict: `UNVERIFIABLE`

---

## What gives us honest confidence

| Signal | What it proves |
|---|---|
| AI generation score > 0.85 | Strong evidence this image was never real |
| Recycled image + date gap > 1 year | Strong evidence of misleading context |
| 2+ independent fact-checkers agree | High-confidence FALSE verdict |
| No signals found | Honest UNVERIFIABLE — not a lazy FALSE |
| Conflicting signals | Reduced confidence, flag for adversarial review |
