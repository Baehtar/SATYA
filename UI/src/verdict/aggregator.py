import logging
from typing import Optional
from src.models.schemas import CheckRequest, EvidenceBundle, ImageAnalysis, ClaimAnalysis

logger = logging.getLogger(__name__)

async def aggregate_evidence(request: CheckRequest, image_analysis: Optional[ImageAnalysis], claim_analysis: Optional[ClaimAnalysis]) -> EvidenceBundle:
    return EvidenceBundle(
        request_id=request.id,
        message_type=request.message_type,
        image_analysis=image_analysis,
        claim_analysis=claim_analysis
    )
