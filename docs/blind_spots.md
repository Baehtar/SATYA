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

### 11. Reverse Image Search Coverage
- "No matching images found" means **our providers did not locate a copy** — never that the
  image is original. Google's index does not cover most WhatsApp groups, Telegram channels,
  regional news sites or anything behind a login
- The engine reports `NO_MATCHES_LOCATED` and `searched: true/false` separately so an
  unconfigured API key can never masquerade as a clean result

### 12. Publication Dates Are Not Upload Dates
- We read the date a *page* claims to have been published, which can be back-dated, missing,
  auto-updated by a CMS, or simply wrong
- `dateModified` is deliberately ignored — a 2019 article edited last week would otherwise
  look brand new. Conflicting dates on one page are flagged, not silently resolved
- A photo agency page dated 2015 may itself be reusing an older photograph

### 13. Google Lens Needs a Public Image URL
- SerpAPI's Lens engine searches *by URL*, so it only runs when `PUBLIC_IMAGE_BASE_URL` or
  `SERPAPI_LENS_ALLOW_UPLOAD` is set — both publish the user's picture to a third party
- Default deployment runs Google Vision alone, which accepts the bytes directly. Fewer
  matches, no privacy cost

### 14. Forensics Are Signals, Not Proof
- ELA fires on any re-compression; every WhatsApp forward is re-compressed
- Copy-move fires on genuinely repeating content (crowds, tiles, UI chrome in screenshots)
- Resampling fires on any resize — universal for anything through a social platform
- "Photoshop" in EXIF means the file was *saved* by an editor; cropping counts
- On images above ~1536px only a native-resolution centre window is examined, so a forgery
  at the very edge of a large frame can be missed

### 15. Cropped and Recomposed Images
- A heavy crop, a mirror flip, or an image placed inside a collage may not match the original
  in either provider's index, and pHash tolerance is limited
- Partial matches are weighted below full matches for exactly this reason

---

## What gives us honest confidence

| Signal | What it proves |
|---|---|
| AI generation score > 0.85 | Strong evidence this image was never real |
| Exact match on a dated news page + gap > 1 year + a date asserted in the claim | Strong evidence of misleading context — about the framing, not the photo |
| The same page found independently by Vision *and* Lens | Corroborated provenance, recorded but not double-counted |
| 2+ independent fact-checkers agree | High-confidence FALSE verdict |
| No signals found | Honest UNVERIFIABLE — not a lazy FALSE |
| Conflicting signals | Reduced confidence, flag for adversarial review |

| Signal | What it does NOT prove |
|---|---|
| No reverse-search match | That the image is original — only that no copy was located |
| An old match with no date in the claim | Misuse. File photos are legitimate |
| Visual similarity to an old photo | Anything about *this* photo |
| High ELA / resampling / stripped EXIF | Manipulation. All three are normal for any forward |
