"""
services/image/google_vision.py — Google Cloud Vision Web Detection REST client.
Uploads local images directly via base64 payload to Google Cloud Vision API.
"""
import base64
import os
import httpx
import structlog
from typing import Dict, Any, List, Optional
from src.config import settings

log = structlog.get_logger(__name__)

VISION_API_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"


async def search_google_vision_web(image_path: str) -> Dict[str, Any]:
    """
    Performs Google Cloud Vision Web Detection on a local image file.
    
    Returns:
      {
        "provider": "google_vision",
        "best_guess_labels": List[str],
        "full_matching_images": List[Dict[str, str]],
        "partial_matching_images": List[Dict[str, str]],
        "pages_with_matching_images": List[Dict[str, str]],
        "visually_similar_images": List[Dict[str, str]],
        "web_entities": List[Dict[str, Any]],
        "error": Optional[str]
      }
    """
    api_key = (
        os.getenv("GOOGLE_VISION_API_KEY")
        or os.getenv("GOOGLE_CLOUD_VISION_API_KEY")
        or getattr(settings, "google_vision_api_key", "")
    )

    result: Dict[str, Any] = {
        "provider": "google_vision",
        "best_guess_labels": [],
        "full_matching_images": [],
        "partial_matching_images": [],
        "pages_with_matching_images": [],
        "visually_similar_images": [],
        "web_entities": [],
        "error": None
    }

    if not api_key:
        log.warning("google_vision_key_missing")
        result["error"] = "Google Vision API key missing"
        return result

    if not os.path.exists(image_path):
        result["error"] = f"Image file not found: {image_path}"
        return result

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        base64_encoded = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "requests": [
                {
                    "image": {
                        "content": base64_encoded
                    },
                    "features": [
                        {
                            "type": "WEB_DETECTION",
                            "maxResults": 20
                        }
                    ]
                }
            ]
        }

        url = f"{VISION_API_ENDPOINT}?key={api_key}"

        async with httpx.AsyncClient(timeout=25.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

        if response.status_code != 200:
            log.warning("google_vision_http_error", status_code=response.status_code, body=response.text[:200])
            result["error"] = f"HTTP {response.status_code}: {response.text[:100]}"
            return result

        data = response.json()

        responses = data.get("responses", [])
        if not responses:
            return result

        web_detection = responses[0].get("webDetection", {})

        # 1. Best guess labels
        for bg in web_detection.get("bestGuessLabels", []):
            label = bg.get("label", "").strip()
            if label:
                result["best_guess_labels"].append(label)

        # 2. Full matching images
        for img in web_detection.get("fullMatchingImages", []):
            u = img.get("url", "")
            if u:
                result["full_matching_images"].append({"url": u, "page_title": ""})

        # 3. Partial matching images
        for img in web_detection.get("partialMatchingImages", []):
            u = img.get("url", "")
            if u:
                result["partial_matching_images"].append({"url": u, "page_title": ""})

        # 4. Pages with matching images
        for page in web_detection.get("pagesWithMatchingImages", []):
            p_url = page.get("url", "")
            p_title = page.get("pageTitle", "")
            if p_url:
                result["pages_with_matching_images"].append({
                    "url": p_url,
                    "page_title": p_title,
                    "full_matching_images": [i.get("url", "") for i in page.get("fullMatchingImages", [])],
                    "partial_matching_images": [i.get("url", "") for i in page.get("partialMatchingImages", [])],
                })

        # 5. Visually similar images
        for img in web_detection.get("visuallySimilarImages", []):
            u = img.get("url", "")
            if u:
                result["visually_similar_images"].append({"url": u, "page_title": ""})

        # 6. Web entities
        for entity in web_detection.get("webEntities", []):
            desc = entity.get("description", "")
            score = entity.get("score", 0.0)
            if desc:
                result["web_entities"].append({"description": desc, "score": score})

        log.info(
            "google_vision_web_detection_done",
            pages=len(result["pages_with_matching_images"]),
            full=len(result["full_matching_images"]),
            partial=len(result["partial_matching_images"])
        )

    except Exception as e:
        log.error("google_vision_failed", error=str(e))
        result["error"] = str(e)

    return result
