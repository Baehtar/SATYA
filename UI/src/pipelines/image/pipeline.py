import logging
import asyncio
from src.models.schemas import ImageAnalysis
from src.pipelines.image.ai_detector import detect_ai_generated
from src.pipelines.image.manipulation import detect_manipulation
from src.pipelines.image.reverse_search import reverse_image_search

logger = logging.getLogger(__name__)

async def run_image_pipeline(image_path: str, progress_cb) -> ImageAnalysis:
    await progress_cb("image_analysis", "started", "Starting image analysis pipeline")
    
    # Run all three detectors in PARALLEL using asyncio.gather
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                detect_ai_generated(image_path),
                detect_manipulation(image_path),
                reverse_image_search(image_path),
                return_exceptions=True
            ),
            timeout=30.0
        )
        
        ai_result, manipulation_result, reverse_result = results
        
        if isinstance(ai_result, Exception):
            logger.error(f"AI detection failed: {ai_result}")
            ai_result = None
            
        if isinstance(manipulation_result, Exception):
            logger.error(f"Manipulation detection failed: {manipulation_result}")
            manipulation_result = None
            
        if isinstance(reverse_result, Exception):
            logger.error(f"Reverse search failed: {reverse_result}")
            reverse_result = None
            
        await progress_cb("image_analysis", "completed", "Completed image analysis")
        
        return ImageAnalysis(
            ai_detection=ai_result,
            manipulation=manipulation_result,
            reverse_search=reverse_result
        )
    except asyncio.TimeoutError:
        logger.error("Image pipeline timed out")
        await progress_cb("image_analysis", "error", "Image analysis timed out")
        return ImageAnalysis()
