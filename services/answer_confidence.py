"""Answer confidence vs research completeness — separate concepts."""

from dataclasses import dataclass

from domain.models import Evidence, EvidenceConfidence, Source, SourceQuality
from services.evidence_confidence import CONFIDENCE_NUMERIC
from services.query_router import QueryComplexity


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Separated confidence dimensions."""

    answer_confidence: EvidenceConfidence
    answer_confidence_numeric: float
    answer_reasoning: str
    research_completeness: EvidenceConfidence
    research_completeness_numeric: float
    completeness_reasoning: str


def _direct_answer_evidence(
    evidence_list: list[Evidence],
    query_terms: set[str],
) -> list[Evidence]:
    if not query_terms:
        return evidence_list[:1] if evidence_list else []
    return [
        ev
        for ev in evidence_list
        if sum(1 for t in query_terms if t in ev.exact_text.lower())
        >= min(2, len(query_terms))
    ]


def _extract_query_terms(query: str) -> set[str]:
    import re

    stop = frozenset({
        "the", "a", "an", "is", "was", "were", "are", "what", "who", "when",
        "where", "how", "why", "in", "of", "for", "to", "and", "or", "as",
    })
    words = re.findall(r"[a-z0-9]+", query.lower())
    return {w for w in words if len(w) > 2 and w not in stop}


def compute_confidence_assessment(
    query: str,
    evidence_list: list[Evidence],
    sources: list[Source],
    *,
    complexity: QueryComplexity,
    potential_conflicts: list[str] | None = None,
    consistency_issues: list[str] | None = None,
) -> ConfidenceAssessment:
    """
    Compute answer confidence (core answer support) separately from
    research completeness (breadth of investigation).
    """
    source_lookup = {s.id: s for s in sources if s.id is not None}
    query_terms = _extract_query_terms(query)
    direct_evidence = _direct_answer_evidence(evidence_list, query_terms)
    conflicts = potential_conflicts or []
    consistency = consistency_issues or []

    # --- Answer confidence: how well does evidence support the core answer? ---
    answer_parts: list[str] = []
    answer_score = 0.0

    if direct_evidence:
        answer_score += 0.4
        answer_parts.append(f"{len(direct_evidence)} evidence item(s) directly address the question")
    else:
        answer_parts.append("no evidence directly addresses the question")

    auth_direct = 0
    for ev in direct_evidence:
        source = source_lookup.get(ev.source_id)
        if source and source.source_quality in (
            SourceQuality.PRIMARY,
            SourceQuality.OFFICIAL,
            SourceQuality.ACADEMIC,
            SourceQuality.REPUTABLE_SECONDARY,
        ):
            auth_direct += 1

    if auth_direct >= 2:
        answer_score += 0.35
        answer_parts.append(f"{auth_direct} authoritative sources confirm answer")
    elif auth_direct >= 1:
        answer_score += 0.25
        answer_parts.append("authoritative source supports answer")

    if not conflicts and not consistency:
        answer_score += 0.15
    else:
        answer_score -= 0.2
        if conflicts:
            answer_parts.append(f"{len(conflicts)} conflict(s)")
        if consistency:
            answer_parts.append(f"{len(consistency)} consistency issue(s)")

    answer_score = max(0.0, min(1.0, answer_score))

    if answer_score >= 0.7 and auth_direct >= 1 and not conflicts:
        answer_level = EvidenceConfidence.HIGH
    elif answer_score >= 0.45:
        answer_level = EvidenceConfidence.MEDIUM
    else:
        answer_level = EvidenceConfidence.LOW

    # --- Research completeness: how thoroughly was the question investigated? ---
    unique_sources = len({ev.source_id for ev in evidence_list})
    unique_domains: set[str] = set()
    for ev in evidence_list:
        source = source_lookup.get(ev.source_id)
        if source:
            domain = source.metadata.get("domain", "")
            if domain:
                unique_domains.add(domain)

    completeness_parts: list[str] = []
    completeness_score = 0.0

    if complexity == QueryComplexity.SIMPLE:
        # For simple questions, limited research is expected and acceptable
        if direct_evidence and auth_direct >= 1:
            completeness_score = 0.5
            completeness_parts.append(
                "targeted research sufficient for simple factual question"
            )
        else:
            completeness_score = 0.3
            completeness_parts.append("limited direct evidence found")
    else:
        if len(evidence_list) >= 8:
            completeness_score += 0.3
        elif len(evidence_list) >= 3:
            completeness_score += 0.15
        completeness_parts.append(f"{len(evidence_list)} evidence items collected")

        if unique_sources >= 4:
            completeness_score += 0.25
        elif unique_sources >= 2:
            completeness_score += 0.12
        completeness_parts.append(f"{unique_sources} sources consulted")

        if len(unique_domains) >= 3:
            completeness_score += 0.25
        elif len(unique_domains) >= 2:
            completeness_score += 0.12
        completeness_parts.append(f"{len(unique_domains)} independent domains")

        completeness_score = min(1.0, completeness_score)

    if completeness_score >= 0.7:
        completeness_level = EvidenceConfidence.HIGH
    elif completeness_score >= 0.4:
        completeness_level = EvidenceConfidence.MEDIUM
    else:
        completeness_level = EvidenceConfidence.LOW

    return ConfidenceAssessment(
        answer_confidence=answer_level,
        answer_confidence_numeric=CONFIDENCE_NUMERIC[answer_level],
        answer_reasoning="; ".join(answer_parts),
        research_completeness=completeness_level,
        research_completeness_numeric=CONFIDENCE_NUMERIC[completeness_level],
        completeness_reasoning="; ".join(completeness_parts),
    )


def compute_fast_fact_confidence(
    target: "AnswerTarget",
    evidence: Evidence,
    sources: list[Source],
    *,
    fact_value: "StructuredFactValue | None" = None,
) -> ConfidenceAssessment:
    """
    HIGH answer confidence for validated fast-path facts with adequate authority.

    Research completeness is intentionally LOW — targeted research is sufficient.
    """
    from services.fact_target import AnswerTarget
    from services.fact_value import StructuredFactValue
    from services.source_authority import is_source_adequate_for_domain

    source = next((s for s in sources if s.id == evidence.source_id), None)
    if source is None and sources:
        source = sources[0]

    answer_parts: list[str] = []
    has_value = fact_value is not None
    adequate_source = source is not None and is_source_adequate_for_domain(
        source, target.domain
    )

    if has_value:
        answer_parts.append("structured target value extracted and validated")
    if adequate_source:
        answer_parts.append(f"adequate source authority for {target.domain.value}")
    if target.temporal_scope:
        answer_parts.append(f"temporal scope {target.temporal_scope} preserved")

    if has_value and adequate_source:
        answer_level = EvidenceConfidence.HIGH
    elif has_value or adequate_source:
        answer_level = EvidenceConfidence.MEDIUM
    else:
        answer_level = EvidenceConfidence.LOW

    completeness_level = EvidenceConfidence.LOW
    completeness_parts = [
        "targeted fast-path research; exhaustive investigation not required"
    ]

    return ConfidenceAssessment(
        answer_confidence=answer_level,
        answer_confidence_numeric=CONFIDENCE_NUMERIC[answer_level],
        answer_reasoning="; ".join(answer_parts) or "fast fact path",
        research_completeness=completeness_level,
        research_completeness_numeric=CONFIDENCE_NUMERIC[completeness_level],
        completeness_reasoning="; ".join(completeness_parts),
    )
