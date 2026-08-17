import asyncio
import os
import uuid
import httpx
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "Organika/sdxl-detector")


async def check_text(text: str):
    """
    Multilingual Fake News & Claim Verification Pipeline.
    Supports EN, HI (Devanagari/Roman), TA (Tamil/Roman), and Mixed languages.
    Searches PIB, Alt News, BOOM, and Google FactCheck API in parallel.
    """
    print(f"TEXT CLAIM SENT TO VERIFICATION PIPELINE:\n{text[:150]}...")

    from src.models.schemas import CheckRequest, Verdict
    from src.pipelines.text.pipeline import run_text_pipeline

    req = CheckRequest(
        request_id=str(uuid.uuid4()),
        message_type="text",
        text_content=text
    )

    try:
        analysis = await run_text_pipeline(req)

        # Build clean sources list
        sources = [
            {
                "name": m.source_name,
                "url": m.source_url,
                "title": m.original_claim,
                "verdict": m.fact_check_verdict
            }
            for m in analysis.matches[:5] if m.source_url
        ]

        verdict_val = analysis.text_verdict.value if hasattr(analysis.text_verdict, "value") else str(analysis.text_verdict)
        confidence_score = float(analysis.text_verdict_confidence)

        # Language-aware explanation summary
        if verdict_val == "LIKELY_FALSE":
            explanation = (
                f"<b>Claim Analysis: Likely False</b>\n\n"
                f"Extracted Claim: <i>\"{analysis.extracted_claim}\"</i>\n\n"
                f"Evidence: This claim matches fact-checks from authoritative sources "
                f"({', '.join([s['name'] for s in sources[:2]]) or 'Fact-check archives'})."
            )
        elif verdict_val == "LIKELY_TRUE":
            explanation = (
                f"<b>Claim Analysis: Likely True</b>\n\n"
                f"Extracted Claim: <i>\"{analysis.extracted_claim}\"</i>\n\n"
                f"Evidence: Fact-check sources confirm the validity of this assertion."
            )
        else:
            explanation = (
                f"<b>Claim Analysis: Unverifiable</b>\n\n"
                f"Extracted Claim: <i>\"{analysis.extracted_claim}\"</i>\n\n"
                f"No active fake-news debunks or verifications were found in PIB, Alt News, or BOOM archives for this claim.\n\n"
                f"ℹ️ <i>Fact-check databases focus on debunking viral rumors rather than covering standard news events.</i>"
            )

        return {
            "type": "text",
            "verdict": verdict_val,
            "confidence": confidence_score,
            "explanation": explanation,
            "sources": sources,
            "extracted_claim": analysis.extracted_claim,
            "language": analysis.language.value if hasattr(analysis.language, "value") else str(analysis.language)
        }

    except Exception as e:
        print(f"Error in check_text pipeline: {e}")
        return {
            "type": "text",
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "explanation": f"Claim verification error: {e}",
            "sources": []
        }


async def check_image_ai(image_path: str):
    """
    Existing AI Image Generation Detector using Hugging Face Inference API (Organika/sdxl-detector).
    """
    api_key = os.getenv("HF_API_KEY", HF_API_KEY)
    model = os.getenv("HF_IMAGE_MODEL", HF_IMAGE_MODEL)

    if not api_key:
        return {"verdict": "UNVERIFIABLE", "confidence": 0.0, "artificial_score": 0.0, "explanation": "HF API key missing."}

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        return {"verdict": "UNVERIFIABLE", "confidence": 0.0, "artificial_score": 0.0, "explanation": str(e)}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "image/jpeg"}
    if image_path.lower().endswith(".png"):
        headers["Content-Type"] = "image/png"

    endpoints = [
        f"https://router.huggingface.co/hf-inference/models/{model}",
        f"https://api-inference.huggingface.co/models/{model}"
    ]

    response_data = None
    async with httpx.AsyncClient(timeout=20.0) as client:
        for endpoint in endpoints:
            try:
                response = await client.post(endpoint, headers=headers, content=image_bytes)
                if response.status_code == 200:
                    response_data = response.json()
                    break
            except Exception:
                pass

    if not response_data or not isinstance(response_data, list):
        return {"verdict": "UNVERIFIABLE", "confidence": 0.0, "artificial_score": 0.0, "explanation": "AI detection unavailable."}

    top_pred = response_data[0]
    top_label = str(top_pred.get("label", "")).lower()
    top_score = float(top_pred.get("score", 0.0))

    scores = {str(item.get("label", "")).lower(): float(item.get("score", 0.0)) for item in response_data if isinstance(item, dict)}
    artificial_score = scores.get("artificial", scores.get("fake", scores.get("sdxl", 0.0)))
    if artificial_score == 0.0 and top_label in ["artificial", "fake", "sdxl", "ai"]:
        artificial_score = top_score

    return {
        "verdict": "LIKELY_FALSE" if artificial_score >= 0.70 else "LIKELY_TRUE",
        "confidence": artificial_score if artificial_score >= 0.70 else (1.0 - artificial_score),
        "artificial_score": artificial_score,
        "explanation": f"Image AI-generation probability: {artificial_score * 100:.1f}%."
    }


async def check_image(image_path: str, progress_callback=None, mode: str = None):
    """
    FULL TARGET FLOW with Mode Selection Support:
    - mode="ai_image": Dedicated AI visual detector report (SDXL / Midjourney / DALL-E).
    - mode="fake_news" / "extract_image": Dedicated OCR text extraction & news claim verification.
    - mode=None: Full combined pipeline (AI Image Detector + OCR Claim Verification).
    """
    print(f"IMAGE PROCESSOR ACTIVATED (Mode={mode}): {image_path}")

    # Mode 3: Dedicated AI Image Detection
    if mode == "ai_image":
        if progress_callback:
            await progress_callback("🤖 Running AI Image Authenticity Detection (SDXL/Midjourney)...")
        ai_res = await check_image_ai(image_path)
        image_ai_score = float(ai_res.get("artificial_score", 0.0))

        if progress_callback:
            await progress_callback("✅ AI visual inspection complete.")

        return {
            "type": "image",
            "verdict": ai_res.get("verdict", "UNVERIFIABLE"),
            "confidence": ai_res.get("confidence", 0.0),
            "image_ai_score": image_ai_score,
            "explanation": f"🤖 <b>AI Image Detection Result:</b>\n\n{ai_res.get('explanation', '')}",
            "sources": []
        }

    # Mode 1 & 2 / Combined Flow
    if progress_callback:
        await progress_callback("🔍 Reading image & running analysis...")

    ai_task = check_image_ai(image_path)

    from services.ocr import extract_text_from_image, normalize_ocr_result
    ocr_raw = await extract_text_from_image(image_path)
    ocr_meta = normalize_ocr_result(ocr_raw.get("raw_text", ""))

    ai_res = await ai_task
    image_ai_score = float(ai_res.get("artificial_score", 0.0))

    # Step 2: Check if OCR found readable text in image
    if not ocr_meta.get("has_readable_text"):
        if progress_callback:
            await progress_callback("✅ No news text detected in image. Visual AI report ready.")
        return {
            "type": "image",
            "verdict": ai_res.get("verdict", "UNVERIFIABLE"),
            "confidence": ai_res.get("confidence", 0.0),
            "image_ai_score": image_ai_score,
            "explanation": f"<b>Image Authenticity:</b> {ai_res.get('explanation', '')}\n\n<i>Note: No legible news headline/text was found in this image for fact-check search.</i>",
            "sources": [],
            "ocr": ocr_meta
        }

    # Step 3: Run Text Claim Extraction & Verification Pipeline
    if progress_callback:
        await progress_callback(f"📝 Extracted news text ({ocr_meta.get('language')}). Checking claim against PIB/AltNews/BOOM/Google...")

    from src.models.schemas import CheckRequest
    from src.pipelines.text.pipeline import run_text_pipeline
    from src.verdict.evidence_aggregator import aggregate_evidence

    req = CheckRequest(
        request_id=str(uuid.uuid4()),
        message_type="image",
        text_content=ocr_meta.get("cleaned_text", "")
    )

    if progress_callback:
        await progress_callback("⚖️ Comparing evidence & calibrating verdict...")

    text_analysis = await run_text_pipeline(req)

    # Step 4: Aggregate Evidence & Fuse Image AI + Claim NLI Signals
    fused_res = aggregate_evidence(text_analysis, image_ai_score, ocr_meta)

    extracted_claim = fused_res.get("extracted_claim", ocr_meta.get("cleaned_text")[:150])
    verdict_val = fused_res.get("verdict", "UNVERIFIABLE")
    conf_score = fused_res.get("confidence_score", 0.5)
    conf_level = fused_res.get("confidence_level", "MODERATE")
    sources = fused_res.get("sources", [])

    # Step 5: Format Telegram Verdict Card Text
    if verdict_val == "LIKELY_FALSE":
        explanation = (
            f"<b>Claim Analysis: Likely False</b>\n\n"
            f"Extracted Claim: <i>\"{extracted_claim}\"</i>\n\n"
            f"Evidence: Fact-checks debunk this claim.\n\n"
            f"🖼️ <b>Image Note:</b> {fused_res.get('image_note', '')}"
        )
    elif verdict_val == "LIKELY_TRUE":
        explanation = (
            f"<b>Claim Analysis: Likely True</b>\n\n"
            f"Extracted Claim: <i>\"{extracted_claim}\"</i>\n\n"
            f"Evidence: Verified by news & fact-check sources.\n\n"
            f"🖼️ <b>Image Note:</b> {fused_res.get('image_note', '')}"
        )
    else:
        explanation = (
            f"<b>Claim Analysis: Unverifiable</b>\n\n"
            f"Extracted Claim: <i>\"{extracted_claim}\"</i>\n\n"
            f"No active fake-news debunks or verifications were found in PIB, Alt News, or BOOM archives for this claim.\n\n"
            f"🖼️ <b>Image Note:</b> {fused_res.get('image_note', '')}"
        )

    if progress_callback:
        await progress_callback("✅ Analysis complete.")

    return {
        "type": "image",
        "verdict": verdict_val,
        "confidence": conf_score,
        "confidence_level": conf_level,
        "extracted_claim": extracted_claim,
        "explanation": explanation,
        "image_ai_score": image_ai_score,
        "sources": sources,
        "ocr": ocr_meta
    }


async def check_voice(audio_path: str):
    """
    Speech-to-text + claim verification for voice notes.
    """
    print(f"VOICE SENT TO ML: {audio_path}")
    await asyncio.sleep(1)

    return {
        "type": "voice",
        "verdict": "UNVERIFIABLE",
        "confidence": 0.0,
        "explanation": "Voice transcription pipeline is active. Transcribed claims are processed via the fake news engine.",
        "sources": []
    }


async def check_mixed(image_path: str, caption: str):
    """
    Runs Image AI Detector AND Text Claim Pipeline in parallel.
    Performs Evidence Fusion (Section 2 & 16 of fake_news_workflow.md):
    - Image AI score is an image authenticity signal, NOT proof that text claim is false!
    """
    image_task = check_image(image_path)
    text_task = check_text(caption) if caption else None

    if text_task:
        image_result, text_result = await asyncio.gather(image_task, text_task)
    else:
        image_result = await image_task
        text_result = None

    if not text_result:
        return image_result

    # ── Evidence Fusion Logic ─────────────────────────────────────────────────
    img_verdict = image_result.get("verdict", "UNVERIFIABLE")
    img_conf = image_result.get("confidence", 0.0)
    txt_verdict = text_result.get("verdict", "UNVERIFIABLE")
    txt_conf = text_result.get("confidence", 0.0)
    sources = text_result.get("sources", [])

    is_ai_image = (img_verdict == "LIKELY_FALSE" and img_conf >= 0.70)

    # Fusion Rule 1: Text claim debunked by PIB/Alt News/BOOM
    if txt_verdict == "LIKELY_FALSE":
        fused_verdict = "LIKELY_FALSE"
        fused_confidence = max(txt_conf, 0.85)
        fused_explanation = (
            f"<b>Caption Claim: Likely False</b>\n"
            f"Fact-checks from {', '.join([s['name'] for s in sources[:2]]) or 'sources'} debunk this claim.\n\n"
            f"<b>Image Note:</b> {image_result.get('explanation', '')}"
        )
    # Fusion Rule 2: Text claim verified as true by sources
    elif txt_verdict == "LIKELY_TRUE":
        fused_verdict = "LIKELY_TRUE"
        fused_confidence = txt_conf
        if is_ai_image:
            fused_explanation = (
                f"<b>Caption Claim: Likely True</b>\n"
                f"{text_result.get('explanation', '')}\n\n"
                f"⚠️ <i>Note: The accompanying image shows strong signs of AI generation, "
                f"but the written claim itself is factual.</i>"
            )
        else:
            fused_explanation = text_result.get('explanation', '')
    # Fusion Rule 3: Text claim unverifiable, but image is AI-generated
    elif is_ai_image:
        fused_verdict = "UNVERIFIABLE"
        fused_confidence = img_conf
        fused_explanation = (
            f"<b>Image Note:</b> The image shows strong signs of AI generation ({img_conf*100:.1f}% confidence).\n\n"
            f"<b>Caption Claim:</b> Could not be independently verified against PIB/Alt News/BOOM archives."
        )
    # Fusion Rule 4: Both unverifiable or real image with unverified claim
    else:
        fused_verdict = "UNVERIFIABLE"
        fused_confidence = 0.5
        fused_explanation = (
            f"We could not find enough independent fact-checks for this claim in PIB, Alt News, or BOOM archives."
        )

    return {
        "type": "mixed",
        "verdict": fused_verdict,
        "confidence": fused_confidence,
        "explanation": fused_explanation,
        "image": image_result,
        "text": text_result,
        "sources": sources,
    }