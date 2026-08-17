import logging
from src.models.schemas import CheckRequest, MessageType, EvidenceBundle, VerdictCard
from src.pipelines.image.pipeline import run_image_pipeline
from src.pipelines.text.pipeline import run_text_pipeline
from src.verdict.aggregator import aggregate_evidence
from src.verdict.confidence import calibrate_confidence
from src.verdict.card_generator import generate_verdict_card
import asyncio

logger = logging.getLogger(__name__)

def classify(text: str = None, image_path: str = None, audio_path: str = None) -> MessageType:
    if audio_path:
        return MessageType.VOICE
    if image_path and text:
        return MessageType.MIXED
    if image_path:
        return MessageType.IMAGE
    if text:
        return MessageType.TEXT
    return MessageType.TEXT

async def dispatch(request: CheckRequest, progress_callback) -> tuple[EvidenceBundle, VerdictCard]:
    image_analysis = None
    claim_analysis = None
    
    # Process voice recording if present
    if request.message_type == MessageType.VOICE and request.audio_path:
        await progress_callback("audio_analysis", "running", "Transcribing voice recording...")
        from src.pipelines.audio.voice_analyzer import analyze_voice
        transcription, _ = await analyze_voice(request.audio_path)
        if transcription:
            request.text = transcription
            await progress_callback("audio_analysis", "completed", f"Transcribed: \"{transcription[:60]}...\"")
        else:
            await progress_callback("audio_analysis", "completed", "Audio processed.")

    tasks = []
    
    if request.message_type in [MessageType.IMAGE, MessageType.MIXED]:
        tasks.append(run_image_pipeline(request.image_path, progress_callback))
    else:
        tasks.append(asyncio.sleep(0)) # placeholder
        
    if request.message_type in [MessageType.TEXT, MessageType.MIXED, MessageType.VOICE] and request.text:
        tasks.append(run_text_pipeline(request.text, progress_callback))
    else:
        tasks.append(asyncio.sleep(0)) # placeholder
        
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    if request.message_type in [MessageType.IMAGE, MessageType.MIXED]:
        image_analysis = results[0]
        if isinstance(image_analysis, Exception):
            logger.error(f"Image pipeline failed: {image_analysis}")
            image_analysis = None
            
    if request.message_type in [MessageType.TEXT, MessageType.MIXED, MessageType.VOICE] and request.text:
        claim_analysis = results[1]
        if isinstance(claim_analysis, Exception):
            logger.error(f"Text pipeline failed: {claim_analysis}")
            claim_analysis = None
            
    evidence = await aggregate_evidence(request, image_analysis, claim_analysis)
    
    verdict, conf_float, conf_level = calibrate_confidence(evidence)
    verdict_card = await generate_verdict_card(evidence, verdict, conf_float, conf_level)
    
    return evidence, verdict_card
