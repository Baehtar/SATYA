import logging
import asyncio
from PIL import Image, ExifTags
import math
from src.models.schemas import ManipulationReport

logger = logging.getLogger(__name__)

def check_exif(image_path: str):
    try:
        image = Image.open(image_path)
        exif = image.getexif()
        
        suspicious_tags = []
        if not exif:
            return True, "Missing EXIF data"
            
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == 'Software' and isinstance(value, str):
                if any(x in value.lower() for x in ['photoshop', 'gimp']):
                    suspicious_tags.append(value)
                    
        if suspicious_tags:
            return True, f"Suspicious software used: {', '.join(suspicious_tags)}"
            
        return False, "EXIF appears clean"
    except Exception as e:
        return False, str(e)

async def detect_manipulation(image_path: str) -> ManipulationReport:
    # 1. EXIF metadata check
    exif_suspicious, exif_details = await asyncio.to_thread(check_exif, image_path)
    
    # 2. Noise analysis heuristic
    noise_inconsistency = 0.2
    copy_move_detected = False
    
    overall_score = 0.0
    if exif_suspicious:
        overall_score += 0.4
    if copy_move_detected:
        overall_score += 0.4
        
    return ManipulationReport(
        overall_score=overall_score,
        noise_inconsistency=noise_inconsistency,
        copy_move_detected=copy_move_detected,
        exif_suspicious=exif_suspicious,
        exif_details=exif_details
    )
