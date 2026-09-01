"""Claim relevance/materiality filtering — distinct from support validation."""

import re
from enum import Enum

from services.claim_schemas import CandidateClaimItem


class ClaimRelevance(str, Enum):
    CRITICAL = "critical"
    SUPPORTING = "supporting"
    CONTEXTUAL = "contextual"
    IRRELEVANT = "irrelevant"


# Tangential patterns for narrow factual questions
_TANGENTIAL_PATTERNS = re.compile(
    r"\b(grid position|finished (first|second|third)|nationality|NED|MEX|"
    r"started (ninth|tenth|eleventh)|climbed through|overtook|team mate|"
    r"pole position|qualifying|practice session|free practice)\b",
    re.IGNORECASE,
)

_ANSWER_PATTERNS = {
    "championship": re.compile(
        r"\b(won|champion|title|world championship|drivers?.? title)\b", re.I
    ),
    "capital": re.compile(r"\b(capital|capital city)\b", re.I),
    "revenue": re.compile(r"\b(revenue|sales|earnings|billion|million)\b", re.I),
    "president": re.compile(r"\b(president|elected|inaugurated)\b", re.I),
}


def _extract_query_terms(query: str) -> set[str]:
    stop = frozenset({
        "the", "a", "an", "is", "was", "were", "are", "what", "who", "when",
        "where", "how", "why", "in", "of", "for", "to", "and", "or", "as",
    })
    words = re.findall(r"[a-z0-9]+", query.lower())
    return {w for w in words if len(w) > 2 and w not in stop}


def _question_topic(query: str) -> str | None:
    q = query.lower()
    if any(k in q for k in ("championship", "won the", "winner")):
        return "championship"
    if "capital" in q:
        return "capital"
    if "revenue" in q or "fiscal" in q:
        return "revenue"
    if "president" in q:
        return "president"
    return None


def assess_claim_relevance(
    candidate: CandidateClaimItem,
    research_question: str,
    *,
    claim_depth: str = "moderate",
) -> ClaimRelevance:
    """
    Assess materiality of a claim to the research question.

    Relevance is distinct from support — a claim can be supported but irrelevant.
    """
    claim_text = candidate.claim_text
    claim_lower = claim_text.lower()
    query_terms = _extract_query_terms(research_question)
    importance = candidate.importance.lower().strip()

    # Respect extractor-assigned HIGH importance as critical
    if importance == "high":
        topic = _question_topic(research_question)
        if topic and topic in _ANSWER_PATTERNS:
            if _ANSWER_PATTERNS[topic].search(claim_lower):
                return ClaimRelevance.CRITICAL
        overlap = sum(1 for t in query_terms if t in claim_lower)
        if overlap >= min(2, len(query_terms)):
            return ClaimRelevance.CRITICAL

    # Tangential race/event details for championship questions
    topic = _question_topic(research_question)
    if topic == "championship" and _TANGENTIAL_PATTERNS.search(claim_lower):
        if not _ANSWER_PATTERNS["championship"].search(claim_lower):
            return ClaimRelevance.IRRELEVANT

    # Keyword overlap scoring
    overlap = sum(1 for t in query_terms if t in claim_lower)
    overlap_ratio = overlap / max(len(query_terms), 1)

    if claim_depth == "minimal":
        if overlap_ratio >= 0.4 and importance in ("high", "medium"):
            return ClaimRelevance.CRITICAL
        if overlap_ratio >= 0.25:
            return ClaimRelevance.SUPPORTING
        if overlap_ratio >= 0.1:
            return ClaimRelevance.CONTEXTUAL
        return ClaimRelevance.IRRELEVANT

    if overlap_ratio >= 0.35 or importance == "high":
        return ClaimRelevance.CRITICAL
    if overlap_ratio >= 0.2 or importance == "medium":
        return ClaimRelevance.SUPPORTING
    if overlap_ratio >= 0.1:
        return ClaimRelevance.CONTEXTUAL
    return ClaimRelevance.IRRELEVANT


def should_validate_expensively(relevance: ClaimRelevance, claim_depth: str) -> bool:
    """Whether a claim warrants expensive LLM entailment validation."""
    if relevance == ClaimRelevance.IRRELEVANT:
        return False
    if claim_depth == "minimal" and relevance == ClaimRelevance.CONTEXTUAL:
        return False
    return True


def is_material_claim(relevance: ClaimRelevance) -> bool:
    """Whether a claim belongs in the material set for verification."""
    return relevance in (ClaimRelevance.CRITICAL, ClaimRelevance.SUPPORTING)
