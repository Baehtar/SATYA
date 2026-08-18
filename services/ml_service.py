import asyncio
import os
import uuid
import httpx
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "Organika/sdxl-detector")


async def check_text(text: str, progress_callback=None):
    """
    Multilingual Fake News & Claim Verification Pipeline.
    Supports EN, HI (Devanagari/Roman), TA (Tamil/Roman), and Mixed languages.
    Searches PIB, Alt News, BOOM, and Google FactCheck API in parallel.

    progress_callback(message: str, step: str) — optional async hook for streaming UIs.
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
        analysis = await run_text_pipeline(req, progress_callback=progress_callback)

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


async def run_provenance(image_path: str, claim_text: str = "", ai_score: float = None, progress_callback=None):
    """
    Reverse Image Engine — where has this picture been before?

    Wrapped in a hard timeout: it calls two search providers and then fetches
    the matching pages, so a single slow publisher must never hold up a
    Telegram reply. A timeout degrades to "unavailable", which the renderer
    reports honestly rather than as "no earlier copies found".
    """
    from src.config import settings
    from services.image import reverse_image_check

    if not settings.reverse_search_enabled:
        return None

    if progress_callback:
        await progress_callback("🌐 Checking where this image has appeared before…", "image_analysis")

    try:
        return await asyncio.wait_for(
            reverse_image_check(
                image_path,
                claim_text=claim_text,
                ai_generated_score=ai_score,
            ),
            timeout=settings.image_pipeline_timeout,
        )
    except asyncio.TimeoutError:
        print("Provenance check timed out")
        return {
            "image_status": "SEARCH_UNAVAILABLE",
            "searched": False,
            "ai_generated_score": ai_score,
            "notes": ["The provenance check timed out before it finished."],
            "reverse_matches": [],
            "forensics": {},
            "date_analysis": {},
        }
    except Exception as e:
        print(f"Provenance check failed: {e}")
        return None


def _with_provenance(explanation: str, provenance: dict) -> str:
    """
    Appends the IMAGE ANALYSIS block to a claim explanation.

    The two stay visually separate on purpose: provenance describes the
    photograph, the claim verdict describes the sentence. An old photo does not
    make a statement false, and a true statement does not make the photo recent.
    """
    if not provenance:
        return explanation

    from services.image import render_image_analysis

    forensics_signals = (provenance.get("forensics") or {}).get("signals") or []
    if not provenance.get("searched") and not forensics_signals:
        return explanation

    block = render_image_analysis(provenance)
    return f"{explanation}\n\n━━━━━━━━━━\n{block}" if block else explanation


def apply_provenance_fusion(verdict: str, confidence: float, provenance: dict):
    """
    Lets a recycled image change the MESSAGE-level verdict — but only when the
    message actually made a dated assertion for the photograph to contradict.

    The distinction is the whole point of the engine. If someone sends an old
    photo with no date attached, the photo being old proves nothing; it may be a
    file photo, perfectly legitimately. It is the combination of "this shows
    yesterday's cyclone" with "this photo was published in 2018" that is false —
    and even then, what's false is the framing, not necessarily the photograph.

    Returns (verdict, confidence, note). A claim already debunked on its own
    evidence is left alone; nothing here ever flips a verdict to LIKELY_TRUE.
    """
    if not provenance or provenance.get("image_status") != "RECYCLED":
        return verdict, confidence, ""

    prov_confidence = float(provenance.get("status_confidence") or 0.0)
    date_analysis = provenance.get("date_analysis") or {}
    # "caller"/"explicit_date"/"relative_term" mean a date was actually asserted.
    # Falling back to "now" (comparison_basis=message_received) does not count.
    asserted_date = (
        date_analysis.get("claim_date_method") in ("caller", "explicit_date", "relative_term")
        and date_analysis.get("comparison_basis") == "claim_date"
    )

    if not asserted_date or prov_confidence < 0.55:
        return verdict, confidence, ""

    note = (
        "🔁 <b>The image does not show the event described.</b> It was already "
        "online well before the date claimed for it. The photograph itself may be "
        "genuine — it is the context around it that is wrong."
    )

    if verdict == "LIKELY_FALSE":
        return verdict, max(confidence, prov_confidence), note
    if verdict == "LIKELY_TRUE":
        # The words check out but the picture is borrowed — say exactly that.
        return verdict, confidence, note
    return "LIKELY_FALSE", prov_confidence, note


async def check_image(image_path: str, progress_callback=None, mode: str = None, caption: str = ""):
    """
    FULL TARGET FLOW with Mode Selection Support:
    - mode="ai_image": Dedicated AI visual detector report (SDXL / Midjourney / DALL-E).
    - mode="fake_news" / "extract_image": Dedicated OCR text extraction & news claim verification.
    - mode=None: Full combined pipeline (AI Image Detector + OCR Claim Verification).

    Every mode also runs the Reverse Image Engine (services/image) for
    provenance: AI-generation score, local forensics and reverse search answer
    three different questions and are reported separately.

    `caption` is the message text sent alongside the image — the usual home of
    the date being claimed ("yesterday's flood in Bihar"), which is what the
    earliest located appearance gets compared against.

    progress_callback(message: str, step: str) — optional async hook for streaming UIs.
    """
    print(f"IMAGE PROCESSOR ACTIVATED (Mode={mode}): {image_path}")

    # Mode 3: Dedicated AI Image Detection
    if mode == "ai_image":
        if progress_callback:
            await progress_callback("🤖 Running AI image authenticity detection (SDXL / Midjourney / DALL-E)…", "image_analysis")
        ai_res = await check_image_ai(image_path)
        image_ai_score = float(ai_res.get("artificial_score", 0.0))

        # Provenance still matters here: a real photograph reused out of context
        # is the commonest fake, and it scores 0% on an AI detector.
        provenance = await run_provenance(
            image_path, claim_text=caption, ai_score=image_ai_score,
            progress_callback=progress_callback,
        )

        if progress_callback:
            await progress_callback("✅ AI visual inspection complete.", "image_analysis")

        return {
            "type": "image",
            "verdict": ai_res.get("verdict", "UNVERIFIABLE"),
            "confidence": ai_res.get("confidence", 0.0),
            "image_ai_score": image_ai_score,
            "explanation": _with_provenance(
                f"🤖 <b>AI Image Detection Result:</b>\n\n{ai_res.get('explanation', '')}",
                provenance,
            ),
            "provenance": provenance,
            "sources": []
        }

    # Mode 1 & 2 / Combined Flow
    if progress_callback:
        await progress_callback("🔍 Reading the image & extracting any text…", "image_analysis")

    ai_task = check_image_ai(image_path)

    from services.ocr import extract_text_from_image, normalize_ocr_result
    ocr_raw = await extract_text_from_image(image_path)
    ocr_meta = normalize_ocr_result(ocr_raw.get("raw_text", ""))

    ai_res = await ai_task
    image_ai_score = float(ai_res.get("artificial_score", 0.0))

    # The date being claimed lives in the caption if there is one, otherwise in
    # the text printed on the image itself.
    claim_text = caption or ocr_meta.get("cleaned_text", "")
    provenance_task = asyncio.create_task(
        run_provenance(image_path, claim_text=claim_text, ai_score=image_ai_score,
                       progress_callback=progress_callback)
    )

    # Step 2: Check if OCR found readable text in image
    if not ocr_meta.get("has_readable_text"):
        provenance = await provenance_task
        if progress_callback:
            await progress_callback("✅ No readable text in the image — visual report ready.", "image_analysis")
        return {
            "type": "image",
            "verdict": ai_res.get("verdict", "UNVERIFIABLE"),
            "confidence": ai_res.get("confidence", 0.0),
            "image_ai_score": image_ai_score,
            "explanation": _with_provenance(
                f"<b>Image Authenticity:</b> {ai_res.get('explanation', '')}\n\n"
                f"<i>Note: No legible news headline/text was found in this image for fact-check search.</i>",
                provenance,
            ),
            "provenance": provenance,
            "sources": [],
            "ocr": ocr_meta
        }

    # Step 3: Run Text Claim Extraction & Verification Pipeline
    if progress_callback:
        await progress_callback(f"📝 Text extracted ({ocr_meta.get('language')}) — verifying the claim…", "image_analysis")

    from src.models.schemas import CheckRequest
    from src.pipelines.text.pipeline import run_text_pipeline
    from src.verdict.evidence_aggregator import aggregate_evidence

    req = CheckRequest(
        request_id=str(uuid.uuid4()),
        message_type="image",
        text_content=ocr_meta.get("cleaned_text", "")
    )

    # Claim verification and provenance are independent questions — run them
    # concurrently rather than making the user wait for one then the other.
    text_analysis, provenance = await asyncio.gather(
        run_text_pipeline(req, progress_callback=progress_callback),
        provenance_task,
    )

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

    verdict_val, conf_score, provenance_note = apply_provenance_fusion(
        verdict_val, conf_score, provenance
    )
    if provenance_note:
        explanation = f"{explanation}\n\n{provenance_note}"

    if progress_callback:
        await progress_callback("✅ Analysis complete.", "generating_verdict")

    return {
        "type": "image",
        "verdict": verdict_val,
        "confidence": conf_score,
        "confidence_level": conf_level,
        "extracted_claim": extracted_claim,
        "explanation": _with_provenance(explanation, provenance),
        "image_ai_score": image_ai_score,
        "provenance": provenance,
        "sources": sources,
        "ocr": ocr_meta
    }


async def check_voice(audio_path: str, progress_callback=None):
    """
    Speech-to-text (Gemini) → the same multilingual claim verification pipeline
    used for text forwards. Returns the text verdict plus the transcript.
    """
    print(f"VOICE SENT TO ML: {audio_path}")

    from services.audio import transcribe_audio

    if progress_callback:
        await progress_callback("🎤 Transcribing the voice note…", "audio_analysis")

    stt = await transcribe_audio(audio_path)

    if not stt.get("success") or not stt.get("text"):
        raw_err = str(stt.get("error") or "")
        if "429" in raw_err or "RESOURCE_EXHAUSTED" in raw_err:
            err_msg = (
                "⚠️ <b>Gemini Free Quota Limit Reached</b>\n\n"
                "The free tier daily limit for the primary AI model was reached. "
                "The system automatically cycled to fallback models. Please wait 30 seconds and retry, "
                "or paste your claim as text."
            )
        else:
            err_msg = (
                "<b>Could not transcribe this voice note.</b>\n\n"
                f"{raw_err or 'No intelligible speech was found.'}\n\n"
                "Try a clearer recording, or paste the claim as text instead."
            )

        return {
            "type": "voice",
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "transcript": "",
            "explanation": err_msg,
            "sources": []
        }

    transcript = stt["text"]

    if progress_callback:
        await progress_callback(
            f"🗣️ Heard: “{transcript[:60]}…” — verifying the claim…", "audio_analysis"
        )

    result = await check_text(transcript, progress_callback=progress_callback)

    result["type"] = "voice"
    result["transcript"] = transcript
    result["explanation"] = (
        f"🎤 <b>Transcript</b>\n<i>\"{transcript[:400]}\"</i>\n\n"
        f"{result.get('explanation', '')}"
    )
    return result


async def check_mixed(image_path: str, caption: str, progress_callback=None):
    """
    Runs Image AI Detector AND Text Claim Pipeline in parallel.
    Performs Evidence Fusion (Section 2 & 16 of fake_news_workflow.md):
    - Image AI score is an image authenticity signal, NOT proof that text claim is false!
    """
    # The caption goes to the image check too: it is where the claimed date
    # usually lives, and the provenance engine needs it to compare against.
    image_task = check_image(image_path, progress_callback=progress_callback, caption=caption)
    text_task = check_text(caption, progress_callback=progress_callback) if caption else None

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

    # Fusion Rule 5: provenance. A photograph that was already online before the
    # date the caption claims for it is evidence about the message even when the
    # fact-check archives have nothing on the claim itself.
    provenance = image_result.get("provenance")
    fused_verdict, fused_confidence, provenance_note = apply_provenance_fusion(
        fused_verdict, fused_confidence, provenance
    )
    if provenance_note:
        fused_explanation = f"{fused_explanation}\n\n{provenance_note}"

    return {
        "type": "mixed",
        "verdict": fused_verdict,
        "confidence": fused_confidence,
        "explanation": fused_explanation,
        "image": image_result,
        "text": text_result,
        "provenance": provenance,
        "image_ai_score": image_result.get("image_ai_score", 0.0),
        "sources": sources,
    }