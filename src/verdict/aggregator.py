 """
src/verdict/aggregator.py — Evidence aggregator.
Owned by: Person 3

Merges outputs from all pipelines into a structured evidence summary
that the confidence calibrator can reason over.
"""
import structlog
from src.models.schemas import EvidenceBundle, Verdict

log = structlog.get_logger(__name__)


class AggregatedEvidence:
    """Intermediate representation for the confidence calibrator."""

    def __init__(self, bundle: EvidenceBundle):
        self.bundle = bundle
        self.signals: list[str] = []          # human-readable signal list
        self.raw_verdicts: list[Verdict] = [] # verdicts from individual pipelines
        self.raw_confidences: list[float] = []
        self.is_ai_generated: bool = False
        self.is_manipulated: bool = False
        self.is_recycled: bool = False
        self.recycled_confidence: float = 0.0
        self.has_fact_check: bool = False
        self.fact_check_verdict: Verdict = Verdict.UNVERIFIABLE
        self.fact_check_confidence: float = 0.0
        self.is_voice_clone: bool = False
        self.is_screenshot_tampered: bool = False
        self.best_sources: list = []
        self.no_evidence: bool = False

        self._process()

    def _process(self):
        b = self.bundle
        img = b.image_analysis
        text = b.claim_analysis
        audio = b.audio_analysis
        ss = b.screenshot_analysis

        # ── Image signals ─────────────────────────────────────────────────────
        if img:
            if img.ai_generation_score >= 0.75:
                self.is_ai_generated = True
                self.signals.append(
                    f"AI-generated image detected ({int(img.ai_generation_score*100)}% confidence)"
                )
                self.raw_verdicts.append(Verdict.AI_GENERATED)
                self.raw_confidences.append(img.ai_generation_score)

            elif img.ai_generation_score >= 0.45:
                self.signals.append(
                    f"Possible AI image (borderline, {int(img.ai_generation_score*100)}%)"
                )

            if img.manipulation_score >= 0.65:
                self.is_manipulated = True
                self.signals.append(
                    f"Image manipulation detected (ELA + noise analysis, {int(img.manipulation_score*100)}%)"
                )
                self.raw_verdicts.append(Verdict.MANIPULATED)
                self.raw_confidences.append(img.manipulation_score)

            if img.exif_anomalies:
                self.signals.extend([f"EXIF: {a}" for a in img.exif_anomalies[:2]])

            if img.recycled_image:
                self.is_recycled = True
                self.recycled_confidence = img.recycled_confidence
                self.signals.append(
                    f"Recycled image — earliest appearance: {img.earliest_appearance_date}"
                )
                self.raw_verdicts.append(Verdict.MISLEADING_CONTEXT)
                self.raw_confidences.append(img.recycled_confidence)

            if img.reverse_search_results:
                self.signals.append(
                    f"Found {len(img.reverse_search_results)} similar images online"
                )

        # ── Text signals ──────────────────────────────────────────────────────
        if text and text.text_verdict:
            self.has_fact_check = bool(text.matches)
            self.fact_check_verdict = text.text_verdict
            self.fact_check_confidence = text.text_verdict_confidence
            self.best_sources = text.matches[:3]

            if text.matches:
                self.signals.append(
                    f"Matched {len(text.matches)} fact-check article(s) "
                    f"(best: {text.best_match.source_name if text.best_match else 'unknown'})"
                )
                self.raw_verdicts.append(text.text_verdict)
                self.raw_confidences.append(text.text_verdict_confidence)
            else:
                self.signals.append("No matching fact-check found (claim may be too recent or local)")

        # ── Audio signals ─────────────────────────────────────────────────────
        if audio:
            if audio.voice_clone_score >= 0.70:
                self.is_voice_clone = True
                self.signals.append(
                    f"AI voice clone detected ({int(audio.voice_clone_score*100)}% confidence)"
                )
                self.raw_verdicts.append(Verdict.AI_GENERATED)
                self.raw_confidences.append(audio.voice_clone_score)

        # ── Screenshot signals ────────────────────────────────────────────────
        if ss:
            if ss.tampering_detected:
                self.is_screenshot_tampered = True
                self.signals.append(f"News screenshot tampering: {ss.tampering_details}")
                self.raw_verdicts.append(Verdict.MANIPULATED)
                self.raw_confidences.append(0.75)

            if ss.detected_channel and ss.chyron_verified:
                self.signals.append(
                    f"Screenshot verified from {ss.detected_channel}"
                )

        # ── No evidence case ──────────────────────────────────────────────────
        if not self.signals:
            self.no_evidence = True
            self.signals.append("No specific manipulation or misinformation signals detected")


def aggregate_evidence(bundle: EvidenceBundle) -> AggregatedEvidence:
    """Entry point: bundle → AggregatedEvidence."""
    evidence = AggregatedEvidence(bundle)
    log.info(
        "evidence_aggregated",
        request_id=bundle.request_id,
        n_signals=len(evidence.signals),
        verdicts=evidence.raw_verdicts,
        is_ai=evidence.is_ai_generated,
        is_recycled=evidence.is_recycled,
        has_fc=evidence.has_fact_check,
    )
    return evidence