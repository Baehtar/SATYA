"""
src/pipelines/image/ai_detector.py — AI-generation & deepfake detection.
Owned by: Person 1

Uses HuggingFace ViT model to detect AI-generated images.
Falls back to secondary model if primary fails.
GPU-accelerated when available.
"""
import asyncio
import functools
import structlog
from pathlib import Path

log = structlog.get_logger(__name__)

# Models are loaded once and cached (expensive to load per-request)
_primary_model = None
_primary_processor = None
_secondary_model = None
_device = None


def _get_device():
    global _device
    if _device is None:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("ai_detector_device", device=_device)
    return _device


def _load_primary_model():
    """Load umm-maybe/AI-image-detector (ViT-based, ~350MB)."""
    global _primary_model, _primary_processor
    if _primary_model is None:
        from transformers import ViTForImageClassification, ViTImageProcessor
        model_name = "umm-maybe/AI-image-detector"
        log.info("loading_ai_detector_model", model=model_name)
        _primary_processor = ViTImageProcessor.from_pretrained(model_name)
        _primary_model = ViTForImageClassification.from_pretrained(model_name)
        _primary_model = _primary_model.to(_get_device())
        _primary_model.eval()
        log.info("ai_detector_model_loaded")
    return _primary_model, _primary_processor


def _load_secondary_model():
    """Fallback: Organika/sdxl-detector."""
    global _secondary_model
    if _secondary_model is None:
        from transformers import pipeline as hf_pipeline
        log.info("loading_secondary_ai_detector")
        _secondary_model = hf_pipeline(
            "image-classification",
            model="Organika/sdxl-detector",
            device=0 if _get_device() == "cuda" else -1,
        )
        log.info("secondary_ai_detector_loaded")
    return _secondary_model


def _run_primary_inference(image_path: str) -> dict:
    """Synchronous inference — runs in thread pool."""
    import torch
    from PIL import Image

    model, processor = _load_primary_model()
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(_get_device())
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    # Label mapping: check model config for exact labels
    # umm-maybe/AI-image-detector: label 0 = artificial, label 1 = human
    labels = model.config.id2label
    scores = {labels[i]: probs[0][i].item() for i in range(len(labels))}

    # Normalise to "AI score" regardless of label naming
    ai_score = max(
        scores.get("artificial", 0),
        scores.get("AI", 0),
        scores.get("fake", 0),
        1 - scores.get("human", 1),  # fallback
    )

    return {"score": ai_score, "model": "umm-maybe/AI-image-detector", "raw_scores": scores}


def _run_secondary_inference(image_path: str) -> dict:
    """Synchronous secondary inference — runs in thread pool."""
    pipeline = _load_secondary_model()
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    results = pipeline(image)

    # Map to AI score
    for r in results:
        if "sdxl" in r["label"].lower() or "artificial" in r["label"].lower():
            return {"score": r["score"], "model": "Organika/sdxl-detector"}
    return {"score": 0.0, "model": "Organika/sdxl-detector"}


async def detect_ai_generation(image_path: str) -> dict:
    """
    Async wrapper for AI generation detection.
    Returns: {"score": float, "model": str, "raw_scores": dict}
    """
    if not image_path or not Path(image_path).exists():
        return {"score": 0.0, "model": "none", "error": "No image path"}

    loop = asyncio.get_running_loop()

    try:
        # Run in thread pool to avoid blocking the event loop
        result = await loop.run_in_executor(
            None,
            functools.partial(_run_primary_inference, image_path)
        )
        log.info("ai_detection_done", score=result["score"], model=result["model"])
        return result

    except Exception as primary_err:
        log.warning("primary_ai_detector_failed", error=str(primary_err))
        try:
            result = await loop.run_in_executor(
                None,
                functools.partial(_run_secondary_inference, image_path)
            )
            log.info("secondary_ai_detection_done", score=result["score"])
            return result
        except Exception as secondary_err:
            log.error("all_ai_detectors_failed", error=str(secondary_err))
            return {"score": 0.0, "model": "failed", "error": str(secondary_err)}
