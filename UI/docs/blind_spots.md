# Satya AI Misinformation Checker — Blind Spots & Failure Modes

Fact-checking tools relying on AI models face inherent limitations. This document outlines known blind spots, adversarial evasion risks, and how Satya mitigates them.

## 1. Image AI-Detection Blind Spots
- **Compression Artifacts**: WhatsApp and Telegram compress images aggressively, which can alter high-frequency noise patterns relied on by ELA (Error Level Analysis) and synthetic detector models.
- **Novel Generative Models**: Zero-day diffusion models (e.g., FLUX, Midjourney v6) may not be recognized by older classifier weights.
- **Mitigation**: Satya never relies on a single signal. It combines AI classification, EXIF software tags, noise variance, AND Google Lens reverse image search.

## 2. Text Claim Extraction & Paraphrasing
- **Sarcasm and Hyperbole**: Satirical posts or meme text can be misclassified as factual claims.
- **Cross-Lingual Paraphrasing**: Translations between regional Indian languages (e.g., Bhojpuri, Tamil, Bengali) can obscure entity names.
- **Mitigation**: Satya's confidence calibrator enforces an `UNVERIFIABLE` verdict whenever single weak signals are present, preventing high-confidence false accusations.

## 3. Rate Limits & Third-Party APIs
- **SerpAPI / Google Fact Check**: High volume requests during breaking news events may hit free-tier rate limits.
- **Mitigation**: Satya incorporates an offline curated fact-check database of viral claims (e.g., Kerala flood photos, RBI nano-chip hoaxes) to deliver immediate responses even during network isolation.
