# Satya Architecture Documentation

Satya is built as a single-page web app + FastAPI backend providing real-time AI misinformation verification in < 60 seconds with bilingual (English + Hindi) output cards.

## Data Pipeline Architecture

```mermaid
graph TD
    User([User Request: Text / Image / Voice]) --> Router[src/pipelines/router.py]
    
    Router -->|Voice Recording| VoicePipeline[src/pipelines/audio/voice_analyzer.py]
    VoicePipeline -->|Transcribed Text| TextPipeline
    
    Router -->|Image Input| ImagePipeline[src/pipelines/image/pipeline.py]
    Router -->|Text Input| TextPipeline[src/pipelines/text/pipeline.py]
    
    subgraph Image Pipeline (Parallel 30s Budget)
        ImagePipeline --> AIDetector[ai_detector.py: HF AI Detector + ELA]
        ImagePipeline --> Manipulation[manipulation.py: EXIF + Noise Analysis]
        ImagePipeline --> ReverseSearch[reverse_search.py: SerpAPI Google Lens]
    end
    
    subgraph Text Pipeline (Sequential 25s Budget)
        TextPipeline --> ClaimExtract[claim_extractor.py: Gemini 2.5 Flash]
        ClaimExtract --> FactSearch[fact_checker.py: Google Fact Check + Local Index]
        FactSearch --> ClaimMatch[claim_matcher.py: Semantic Matching]
    end
    
    ImagePipeline --> Aggregator[src/verdict/aggregator.py]
    TextPipeline --> Aggregator
    
    Aggregator --> ConfidenceEngine[src/verdict/confidence.py: Calibrated Scoring]
    ConfidenceEngine --> CardGen[src/verdict/card_generator.py: Gemini Bilingual Explanation]
    
    CardGen --> SSEStream[SSE Stream: /api/check/{id}/stream]
    SSEStream --> UI[Frontend Single Page App]
```

## Latency Budgets & Timeouts

| Pipeline Component | Max Timeout | Strategy / Fallback |
|---|---|---|
| Image Analysis | 30.0s | `asyncio.gather` with PIL ImageChops acceleration |
| Text Claim Extraction | 25.0s | Rule-based regex NLP extractor fallback |
| Reverse Search | 15.0s | Recycled metadata heuristic fallback |
| Verdict Card Generation | 15.0s | Curated bilingual template fallback |
| Total Request Budget | < 60.0s | Async SSE streaming updates |
