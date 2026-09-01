"""Concise evidence-grounded answer for SIMPLE_FACT fast path."""

from domain.models import Claim, Evidence, EvidenceConfidence, Source
from services.answer_confidence import compute_fast_fact_confidence
from services.evidence_confidence import CONFIDENCE_NUMERIC
from services.fact_target import AnswerTarget
from services.fact_value import StructuredFactValue


def build_fast_answer(
    target: AnswerTarget,
    evidence: Evidence,
    core_claim: Claim | None,
    sources: list[Source],
    *,
    fact_value: StructuredFactValue | None = None,
    evidence_display_id: str = "E1",
) -> "FinalReport":
    from state import FinalReport

    source_url = evidence.metadata.get("source_url", "")
    if not source_url and sources:
        for s in sources:
            if s.id == evidence.source_id:
                source_url = s.url
                break

    if core_claim:
        answer_text = f"{core_claim.text} [{evidence_display_id}]"
    else:
        answer_text = f"{evidence.exact_text} [{evidence_display_id}]"

    confidence = compute_fast_fact_confidence(
        target, evidence, sources, fact_value=fact_value
    )

    return FinalReport(
        content=answer_text,
        sources=[source_url] if source_url else [],
        confidence=confidence.answer_confidence_numeric,
        confidence_level=confidence.answer_confidence.value,
        confidence_reasoning=confidence.answer_reasoning,
        answer_confidence_level=confidence.answer_confidence.value,
        answer_confidence_reasoning=confidence.answer_reasoning,
        research_completeness_level=confidence.research_completeness.value,
        research_completeness_reasoning=confidence.completeness_reasoning,
        evidence_ids_used=[evidence_display_id],
        report_metrics={
            "writer_mode": "fast_path",
            "full_writer_skipped": True,
            "full_claim_extractor_skipped": True,
            "core_claims": 1 if core_claim else 0,
            "fact_value": fact_value.model_dump() if fact_value else None,
            "answer_confidence_level": confidence.answer_confidence.value,
            "research_completeness_level": confidence.research_completeness.value,
        },
    )
