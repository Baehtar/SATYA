import logging
import asyncio
from src.models.schemas import ClaimAnalysis, VerdictLevel
from src.pipelines.text.claim_extractor import extract_claim
from src.pipelines.text.fact_checker import search_fact_checks
from src.pipelines.text.claim_matcher import match_claims

logger = logging.getLogger(__name__)

async def run_text_pipeline(text: str, progress_cb) -> ClaimAnalysis:
    await progress_cb("text_analysis", "started", "Extracting claim")
    
    try:
        async with asyncio.timeout(25.0):
            # Step 1
            claim_data = await extract_claim(text)
            
            if not claim_data["is_checkable"]:
                await progress_cb("text_analysis", "completed", "Claim is not verifiable")
                return ClaimAnalysis(
                    extracted_claim=claim_data["claim"],
                    overall_verdict=VerdictLevel.UNVERIFIABLE,
                    fact_checks=[]
                )
                
            await progress_cb("text_analysis", "running", "Searching fact-checkers")
            
            # Step 2
            fact_checks = await search_fact_checks(claim_data["claim"], claim_data["entities"])
            
            await progress_cb("text_analysis", "running", "Matching claims")
            
            # Step 3
            analysis = await match_claims(claim_data["claim"], fact_checks)
            
            await progress_cb("text_analysis", "completed", "Completed text analysis")
            return analysis
            
    except asyncio.TimeoutError:
        logger.error("Text pipeline timed out")
        await progress_cb("text_analysis", "error", "Text analysis timed out")
        return ClaimAnalysis()
    except Exception as e:
        logger.error(f"Text pipeline failed: {e}")
        await progress_cb("text_analysis", "error", f"Error: {str(e)}")
        return ClaimAnalysis()
