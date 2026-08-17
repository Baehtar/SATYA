import logging
from typing import Optional
from src.models.schemas import EvidenceBundle, VerdictLevel, ConfidenceLevel

logger = logging.getLogger(__name__)

def calibrate_confidence(evidence: EvidenceBundle) -> tuple[VerdictLevel, float, ConfidenceLevel]:
    # Default fallback
    verdict_level = VerdictLevel.UNVERIFIABLE
    confidence = 0.5
    
    # Simple rule based engine
    has_image_evidence = evidence.image_analysis is not None
    has_text_evidence = evidence.claim_analysis is not None
    
    if has_image_evidence and not has_text_evidence:
        ai_score = evidence.image_analysis.ai_detection.score if evidence.image_analysis.ai_detection else 0
        manipulation_score = evidence.image_analysis.manipulation.overall_score if evidence.image_analysis.manipulation else 0
        is_recycled = evidence.image_analysis.reverse_search.is_recycled if evidence.image_analysis.reverse_search else False
        
        if ai_score > 0.8:
            verdict_level = VerdictLevel.LIKELY_FALSE
            confidence = float(ai_score)
        elif is_recycled:
            verdict_level = VerdictLevel.LIKELY_FALSE
            confidence = 0.75
        elif manipulation_score > 0.7:
            verdict_level = VerdictLevel.LIKELY_FALSE
            confidence = float(manipulation_score)
        else:
            verdict_level = VerdictLevel.UNVERIFIABLE
            confidence = 0.5
            
    elif has_text_evidence and not has_image_evidence:
        text_verdict = evidence.claim_analysis.overall_verdict
        if text_verdict == VerdictLevel.LIKELY_FALSE:
            verdict_level = VerdictLevel.LIKELY_FALSE
            confidence = 0.85
        elif text_verdict == VerdictLevel.LIKELY_TRUE:
            verdict_level = VerdictLevel.LIKELY_TRUE
            confidence = 0.85
        else:
            verdict_level = VerdictLevel.UNVERIFIABLE
            confidence = 0.5
            
    elif has_image_evidence and has_text_evidence:
        # Simplistic combination for MIXED
        verdict_level = VerdictLevel.UNVERIFIABLE
        confidence = 0.5
        
        if evidence.claim_analysis.overall_verdict == VerdictLevel.LIKELY_FALSE:
            verdict_level = VerdictLevel.LIKELY_FALSE
            confidence = 0.9
            
    confidence_level = ConfidenceLevel.LOW
    if confidence > 0.85:
        confidence_level = ConfidenceLevel.HIGH
    elif confidence >= 0.6:
        confidence_level = ConfidenceLevel.MODERATE
        
    # Calibrate: single weak signal -> NEVER say LIKELY_FALSE with high confidence
    if verdict_level == VerdictLevel.LIKELY_FALSE and confidence < 0.6:
        verdict_level = VerdictLevel.UNVERIFIABLE
        
    return verdict_level, confidence, confidence_level
