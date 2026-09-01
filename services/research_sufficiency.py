"""Research sufficiency checks and authoritative-source short-circuit."""

import re
from dataclasses import dataclass

from domain.models import Evidence, Source, SourceQuality
from services.query_router import QueryComplexity, ResearchBudget

_AUTHORITATIVE_QUALITIES = frozenset({
    SourceQuality.PRIMARY,
    SourceQuality.OFFICIAL,
    SourceQuality.ACADEMIC,
})


@dataclass(frozen=True)
class SufficiencyResult:
    """Whether research can stop early."""

    is_sufficient: bool
    reason: str
    authoritative_source_count: int = 0
    direct_answer_evidence_count: int = 0


def _extract_query_terms(query: str) -> set[str]:
    stop = frozenset({
        "the", "a", "an", "is", "was", "were", "are", "what", "who", "when",
        "where", "how", "why", "in", "of", "for", "to", "and", "or", "as",
    })
    words = re.findall(r"[a-z0-9]+", query.lower())
    return {w for w in words if len(w) > 2 and w not in stop}


def _evidence_matches_query(evidence_text: str, query_terms: set[str]) -> bool:
    if not query_terms:
        return False
    text_lower = evidence_text.lower()
    matches = sum(1 for t in query_terms if t in text_lower)
    return matches >= min(2, len(query_terms))


def check_research_sufficiency(
    query: str,
    evidence_list: list[Evidence],
    sources: list[Source],
    *,
    complexity: QueryComplexity,
    budget: ResearchBudget,
    potential_conflicts: list[str] | None = None,
) -> SufficiencyResult:
    """
    Determine if authoritative evidence is sufficient to stop researching.

    For SIMPLE questions: stop when high-quality sources directly establish the answer.
    """
    if not budget.enable_short_circuit:
        return SufficiencyResult(False, "Short-circuit disabled for this complexity")

    if not evidence_list:
        return SufficiencyResult(False, "No validated evidence yet")

    if potential_conflicts:
        return SufficiencyResult(
            False,
            f"Conflicts detected: {potential_conflicts[0][:80]}",
        )

    source_lookup = {s.id: s for s in sources if s.id is not None}
    query_terms = _extract_query_terms(query)

    authoritative_ids: set[int] = set()
    direct_answer_count = 0

    for ev in evidence_list:
        source = source_lookup.get(ev.source_id)
        if source and source.source_quality in _AUTHORITATIVE_QUALITIES:
            authoritative_ids.add(ev.source_id)
        if _evidence_matches_query(ev.exact_text, query_terms):
            direct_answer_count += 1

    auth_count = len(authoritative_ids)

    if complexity == QueryComplexity.SIMPLE:
        if auth_count >= 1 and direct_answer_count >= 1:
            return SufficiencyResult(
                True,
                "Authoritative source with direct answer evidence",
                authoritative_source_count=auth_count,
                direct_answer_evidence_count=direct_answer_count,
            )
        if auth_count >= 2 and direct_answer_count >= 1:
            return SufficiencyResult(
                True,
                "Multiple authoritative sources confirm answer",
                authoritative_source_count=auth_count,
                direct_answer_evidence_count=direct_answer_count,
            )
        return SufficiencyResult(
            False,
            f"Insufficient authoritative evidence ({auth_count} auth sources, "
            f"{direct_answer_count} direct matches)",
            authoritative_source_count=auth_count,
            direct_answer_evidence_count=direct_answer_count,
        )

    if complexity == QueryComplexity.STANDARD:
        if auth_count >= 2 and direct_answer_count >= 2:
            return SufficiencyResult(
                True,
                "Multiple authoritative sources with direct evidence",
                authoritative_source_count=auth_count,
                direct_answer_evidence_count=direct_answer_count,
            )

    return SufficiencyResult(
        False,
        "Standard/deep question requires broader investigation",
        authoritative_source_count=auth_count,
        direct_answer_evidence_count=direct_answer_count,
    )


def prioritize_sources(sources: list[Source], *, authoritative_first: bool) -> list[Source]:
    """Sort sources by quality when authoritative prioritization is enabled."""
    if not authoritative_first:
        return sources

    def sort_key(s: Source) -> int:
        quality_rank = {
            SourceQuality.PRIMARY: 5,
            SourceQuality.OFFICIAL: 5,
            SourceQuality.ACADEMIC: 4,
            SourceQuality.REPUTABLE_SECONDARY: 3,
            SourceQuality.GENERAL_SECONDARY: 2,
            SourceQuality.USER_GENERATED: 1,
            SourceQuality.UNKNOWN: 0,
        }
        return -quality_rank.get(s.source_quality, 0)

    return sorted(sources, key=sort_key)
