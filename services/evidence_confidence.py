"""Deterministic evidence confidence calculation.

Confidence reflects evidence quality and coverage — NOT probability of truth.
"""

from domain.models import Evidence, EvidenceConfidence, Source, SourceQuality


# Numeric mapping for API/database compatibility (not LLM-generated precision)
CONFIDENCE_NUMERIC: dict[EvidenceConfidence, float] = {
    EvidenceConfidence.HIGH: 0.85,
    EvidenceConfidence.MEDIUM: 0.65,
    EvidenceConfidence.LOW: 0.40,
}

_QUALITY_RANK: dict[SourceQuality, int] = {
    SourceQuality.PRIMARY: 5,
    SourceQuality.OFFICIAL: 5,
    SourceQuality.ACADEMIC: 4,
    SourceQuality.REPUTABLE_SECONDARY: 3,
    SourceQuality.GENERAL_SECONDARY: 2,
    SourceQuality.USER_GENERATED: 1,
    SourceQuality.UNKNOWN: 1,
}


def compute_evidence_confidence(
    evidence_list: list[Evidence],
    sources: list[Source],
    *,
    critique_quality_score: float | None = None,
    potential_conflicts: list[str] | None = None,
    unsupported_areas: list[str] | None = None,
    consistency_issues: list[str] | None = None,
) -> tuple[EvidenceConfidence, float, str]:
    """
    Compute conservative evidence confidence from validated evidence and critique.

    Returns:
        Tuple of (confidence_level, numeric_value, reasoning)
    """
    if not evidence_list:
        return (
            EvidenceConfidence.LOW,
            CONFIDENCE_NUMERIC[EvidenceConfidence.LOW],
            "No validated evidence available.",
        )

    source_lookup = {s.id: s for s in sources if s.id is not None}
    unique_source_ids = {ev.source_id for ev in evidence_list}
    unique_domains: set[str] = set()
    quality_scores: list[int] = []

    for sid in unique_source_ids:
        source = source_lookup.get(sid)
        if source:
            domain = source.metadata.get("domain", "")
            if domain:
                unique_domains.add(domain)
            quality_scores.append(_QUALITY_RANK.get(source.source_quality, 1))

    evidence_count = len(evidence_list)
    source_count = len(unique_source_ids)
    domain_count = len(unique_domains)
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 1

    conflicts = potential_conflicts or []
    gaps = unsupported_areas or []
    consistency = consistency_issues or []
    critique_score = critique_quality_score if critique_quality_score is not None else 0.7

    reasoning_parts: list[str] = []
    score = 0.0  # internal 0-1 scale for level assignment

    # Evidence volume (max 0.25)
    if evidence_count >= 5:
        score += 0.25
        reasoning_parts.append(f"{evidence_count} validated evidence items")
    elif evidence_count >= 2:
        score += 0.15
        reasoning_parts.append(f"{evidence_count} validated evidence items (limited)")
    else:
        score += 0.05
        reasoning_parts.append(f"only {evidence_count} validated evidence item(s)")

    # Source diversity (max 0.20)
    if domain_count >= 3:
        score += 0.20
        reasoning_parts.append(f"{domain_count} independent source domains")
    elif domain_count >= 2:
        score += 0.12
        reasoning_parts.append(f"{domain_count} source domains")
    else:
        score += 0.05
        reasoning_parts.append("single source family")

    # Source quality (max 0.25)
    if avg_quality >= 4:
        score += 0.25
        reasoning_parts.append("high-quality sources")
    elif avg_quality >= 3:
        score += 0.15
        reasoning_parts.append("moderate source quality")
    else:
        score += 0.05
        reasoning_parts.append("lower source quality")

    # Critique alignment (max 0.20)
    score += min(critique_score, 1.0) * 0.20

    # Penalties
    if conflicts:
        score -= 0.15 * min(len(conflicts), 3)
        reasoning_parts.append(f"{len(conflicts)} potential conflict(s)")
    if gaps:
        score -= 0.10 * min(len(gaps), 3)
        reasoning_parts.append(f"{len(gaps)} unsupported area(s)")
    if consistency:
        score -= 0.20 * min(len(consistency), 3)
        reasoning_parts.append(f"{len(consistency)} consistency issue(s)")

    score = max(0.0, min(1.0, score))

    if score >= 0.75 and not conflicts and not consistency:
        level = EvidenceConfidence.HIGH
    elif score >= 0.50:
        level = EvidenceConfidence.MEDIUM
    else:
        level = EvidenceConfidence.LOW

    # Never assign HIGH with serious issues
    if (conflicts or consistency) and level == EvidenceConfidence.HIGH:
        level = EvidenceConfidence.MEDIUM

    if evidence_count < 2 and level != EvidenceConfidence.LOW:
        level = EvidenceConfidence.MEDIUM

    numeric = CONFIDENCE_NUMERIC[level]
    reasoning = "; ".join(reasoning_parts)
    return level, numeric, reasoning
