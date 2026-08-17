import logging
import asyncio
from PIL import Image
import io
import math
from typing import Optional
from src.models.schemas import AIDetectionResult

logger = logging.getLogger(__name__)

_model_cache = {}

def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert('RGB')

def run_hf_pipeline(image_path: str):
    # Synchronous HuggingFace call
    try:
        from transformers import pipeline
        
        if 'ai_detector' not in _model_cache:
            _model_cache['ai_detector'] = pipeline("image-classification", model="umm-maybe/AI-image-detector")
            
        detector = _model_cache['ai_detector']
        image = load_image(image_path)
        result = detector(image)
        
        # Format depends on the exact model
        ai_score = 0.0
        for item in result:
            if item['label'] == 'artificial':
                ai_score = item['score']
                break
                
        return ai_score
    except Exception as e:
        logger.warning(f"Error running HF pipeline: {e}")
        return 0.5 # fallback

def calculate_ela_score(image_path: str) -> float:
    try:
        image = load_image(image_path)
        image.thumbnail((512, 512))
        buffer = io.BytesIO()
        image.save(buffer, 'JPEG', quality=90)
        buffer.seek(0)
        resaved = Image.open(buffer)
        
        from PIL import ImageChops, ImageStat
        diff = ImageChops.difference(image, resaved)
        stat = ImageStat.Stat(diff)
        mean_diff = sum(stat.mean) / max(len(stat.mean), 1)
        return min(mean_diff / 20.0, 1.0)
    except Exception as e:
        logger.warning(f"Error calculating ELA: {e}")
        return 0.0

async def detect_ai_generated(image_path: str) -> AIDetectionResult:
    try:
        # Run HF model in thread
        ai_score = await asyncio.to_thread(run_hf_pipeline, image_path)
        ela_score = await asyncio.to_thread(calculate_ela_score, image_path)
        
        is_ai = ai_score > 0.8
        return AIDetectionResult(
            score=ai_score,
            ela_score=ela_score,
            model_used="umm-maybe/AI-image-detector"
        )
    except Exception as e:
        logger.error(f"AI detection failed: {e}")
        return AIDetectionResult(
            score=0.5,
            ela_score=0.0,
            model_used="fallback"
        )
