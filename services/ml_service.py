import asyncio
import os
import httpx
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "Organika/sdxl-detector")


async def check_text(text):

    print("TEXT SENT TO ML:")
    print(text)

    # TODO: Replace with text model/API if needed
    await asyncio.sleep(1)

    return {
        "type": "text",
        "verdict": "UNVERIFIABLE",
        "confidence": 0.0,
        "explanation": "The text analysis model has not been connected yet.",
        "sources": []
    }


async def check_image(image_path):

    print(f"IMAGE SENT TO ML: {image_path}")

    api_key = os.getenv("HF_API_KEY", HF_API_KEY)
    model = os.getenv("HF_IMAGE_MODEL", HF_IMAGE_MODEL)

    if not api_key:
        return {
            "type": "image",
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "explanation": "Hugging Face API key is not configured.",
            "sources": []
        }

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"Error reading image file {image_path}: {e}")
        return {
            "type": "image",
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "explanation": f"Failed to read image file: {e}",
            "sources": []
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "image/jpeg"
    }

    if image_path.lower().endswith(".png"):
        headers["Content-Type"] = "image/png"

    # Endpoints to try (router domain first, fallback to legacy api-inference)
    endpoints = [
        f"https://router.huggingface.co/hf-inference/models/{model}",
        f"https://api-inference.huggingface.co/models/{model}"
    ]

    max_retries = 3
    response_data = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        for endpoint in endpoints:
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        endpoint,
                        headers=headers,
                        content=image_bytes
                    )

                    if response.status_code == 200:
                        response_data = response.json()
                        break
                    elif response.status_code == 503 or "estimated_time" in response.text:
                        # Model is loading on HF serverless infrastructure
                        try:
                            error_info = response.json()
                            wait_time = min(error_info.get("estimated_time", 5.0), 10.0)
                        except Exception:
                            wait_time = 5.0
                        print(f"Model is loading, waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"HF API returned status {response.status_code}: {response.text}")
                        break
                except httpx.RequestError as req_err:
                    print(f"Network error trying endpoint {endpoint}: {req_err}")
                    break

            if response_data is not None:
                break

    if not response_data or not isinstance(response_data, list):
        error_msg = "Could not get valid response from Hugging Face model."
        if isinstance(response_data, dict) and "error" in response_data:
            error_msg = response_data["error"]

        return {
            "type": "image",
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "explanation": f"Image analysis failed: {error_msg}",
            "sources": [
                {
                    "name": f"Hugging Face ({model})",
                    "url": f"https://huggingface.co/{model}"
                }
            ]
        }

    # Parse Hugging Face classification scores
    # Example format: [{'label': 'human', 'score': 0.999}, {'label': 'artificial', 'score': 0.001}]
    top_prediction = response_data[0]
    top_label = str(top_prediction.get("label", "")).lower()
    top_score = float(top_prediction.get("score", 0.0))

    scores_by_label = {
        str(item.get("label", "")).lower(): float(item.get("score", 0.0))
        for item in response_data if isinstance(item, dict)
    }

    # Determine if artificial/AI or human/real score dominates
    human_score = scores_by_label.get("human", scores_by_label.get("real", 0.0))
    artificial_score = scores_by_label.get("artificial", scores_by_label.get("fake", scores_by_label.get("sdxl", 0.0)))

    # If scores sum to 0 or labels were different, derive from top_prediction
    if human_score == 0.0 and artificial_score == 0.0:
        if top_label in ["artificial", "fake", "sdxl", "ai"]:
            artificial_score = top_score
            human_score = max(0.0, 1.0 - top_score)
        else:
            human_score = top_score
            artificial_score = max(0.0, 1.0 - top_score)

    if artificial_score > human_score or top_label in ["artificial", "fake", "sdxl", "ai"]:
        confidence = artificial_score if artificial_score > 0 else top_score
        verdict = "LIKELY_FALSE"
        explanation = (
            f"The image was analyzed and is classified as "
            f"<b>AI-Generated / Synthetic</b> with <b>{confidence * 100:.1f}%</b> certainty."
        )
    else:
        confidence = human_score if human_score > 0 else top_score
        verdict = "LIKELY_TRUE"
        explanation = (
            f"The image was analyzed and is classified as "
            f"<b>Genuine / Real (Human-Created)</b> with <b>{confidence * 100:.1f}%</b> certainty."
        )

    return {
        "type": "image",
        "verdict": verdict,
        "confidence": confidence,
        "human_score": human_score,
        "artificial_score": artificial_score,
        "explanation": explanation,
        "sources": []
    }


async def check_voice(audio_path):

    print("VOICE SENT TO ML:")
    print(audio_path)

    # TODO: Replace with voice model if needed
    await asyncio.sleep(1)

    return {
        "type": "voice",
        "verdict": "UNVERIFIABLE",
        "confidence": 0.0,
        "explanation": "The voice analysis model has not been connected yet.",
        "sources": []
    }


async def check_mixed(image_path, caption):

    image_task = check_image(image_path)
    text_task = check_text(caption)

    image_result, text_result = await asyncio.gather(image_task, text_task)

    return {
        "type": "mixed",
        "image": image_result,
        "text": text_result
    }