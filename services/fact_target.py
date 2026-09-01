"""Deterministic answer-target extraction for SIMPLE_FACT questions."""

import re
from enum import Enum

from pydantic import BaseModel, Field


class FactFreshness(str, Enum):
    """Freshness classification for future cache/TTL rules."""

    HISTORICAL_STABLE = "historical_stable"
    STRUCTURALLY_STABLE = "structurally_stable"
    TIME_SENSITIVE = "time_sensitive"


class AnswerType(str, Enum):
    PLACE = "place"
    PERSON = "person"
    CURRENCY_VALUE = "currency_value"
    DATE = "date"
    TEXT = "text"
    OTHER = "other"


class FactDomain(str, Enum):
    GEOGRAPHIC = "geographic"
    FINANCIAL = "financial"
    SPORTS = "sports"
    CORPORATE = "corporate"
    TECHNICAL = "technical"
    GENERAL = "general"


class AnswerTarget(BaseModel):
    """Structured target for what a factual question seeks."""

    entity: str = Field(..., description="Primary subject of the question")
    attribute: str = Field(..., description="What property is being asked about")
    temporal_scope: str | None = Field(None, description="Time period if relevant")
    expected_answer_type: AnswerType = AnswerType.OTHER
    domain: FactDomain = FactDomain.GENERAL
    category: str | None = Field(
        None, description="Sub-type e.g. drivers_championship, constructors_championship"
    )
    freshness: FactFreshness = FactFreshness.HISTORICAL_STABLE
    original_question: str = ""


_CAPITAL_RE = re.compile(
    r"what\s+is\s+the\s+capital\s+of\s+(.+?)\??$", re.IGNORECASE
)
_REVENUE_RE = re.compile(
    r"what\s+was\s+(.+?)(?:'s|s)?\s+(revenue|net income|eps)\s+(?:in\s+)?(.+?)\??$",
    re.IGNORECASE,
)
_WINNER_RE = re.compile(
    r"who\s+won\s+(?:the\s+)?(.+?)\??$", re.IGNORECASE
)
_CEO_RE = re.compile(
    r"who\s+is\s+(?:the\s+)?(?:current\s+)?ceo\s+of\s+(.+?)\??$", re.IGNORECASE
)
_WHEN_RE = re.compile(
    r"when\s+was\s+(.+?)\??$", re.IGNORECASE
)
_WHO_IS_RE = re.compile(
    r"who\s+is\s+(?:the\s+)?(.+?)\??$", re.IGNORECASE
)


def _resolve_f1_championship(competition: str, question: str) -> tuple[str, str | None]:
    """Resolve F1 championship type; default to drivers when ambiguous."""
    comp_lower = competition.lower()

    if "constructors" in comp_lower or "constructor" in comp_lower:
        return competition, "constructors_championship"

    if "drivers" in comp_lower or "driver" in comp_lower:
        return competition, "drivers_championship"

    if any(k in comp_lower for k in ("f1", "formula 1", "formula one")):
        year_m = re.search(r"\b((?:19|20)\d{2})\b", competition)
        year = year_m.group(1) if year_m else ""
        entity = f"{year} Formula One World Drivers' Championship".strip()
        return entity, "drivers_championship"

    return competition, None


def extract_fact_target(query: str) -> AnswerTarget | None:
    """Derive the answer target from a narrow factual question."""
    q = query.strip()
    q_with_q = q.rstrip("?").strip() + "?"

    m = _CAPITAL_RE.search(q_with_q)
    if m:
        entity = m.group(1).strip().rstrip("?")
        return AnswerTarget(
            entity=entity,
            attribute="capital",
            expected_answer_type=AnswerType.PLACE,
            domain=FactDomain.GEOGRAPHIC,
            freshness=FactFreshness.STRUCTURALLY_STABLE,
            original_question=query,
        )

    m = _REVENUE_RE.search(q_with_q)
    if m:
        entity = m.group(1).strip()
        metric = m.group(2).strip().lower()
        temporal = m.group(3).strip()
        return AnswerTarget(
            entity=entity,
            attribute=metric if metric in ("eps",) else "revenue",
            temporal_scope=temporal,
            expected_answer_type=AnswerType.CURRENCY_VALUE,
            domain=FactDomain.FINANCIAL,
            freshness=FactFreshness.HISTORICAL_STABLE,
            original_question=query,
        )

    m = _WINNER_RE.search(q_with_q)
    if m:
        competition = m.group(1).strip()
        temporal = None
        year_match = re.search(r"\b(19|20)\d{2}\b", competition)
        if year_match:
            temporal = year_match.group(0)
        domain = FactDomain.SPORTS if any(
            k in competition.lower()
            for k in ("f1", "formula", "championship", "world cup", "olympics", "super bowl")
        ) else FactDomain.GENERAL
        entity, category = (
            _resolve_f1_championship(competition, query)
            if domain == FactDomain.SPORTS
            else (competition, None)
        )
        return AnswerTarget(
            entity=entity,
            attribute="winner",
            temporal_scope=temporal,
            expected_answer_type=AnswerType.PERSON,
            domain=domain,
            category=category,
            freshness=FactFreshness.HISTORICAL_STABLE,
            original_question=query,
        )

    m = _CEO_RE.search(q_with_q)
    if m:
        return AnswerTarget(
            entity=m.group(1).strip(),
            attribute="ceo",
            expected_answer_type=AnswerType.PERSON,
            domain=FactDomain.CORPORATE,
            freshness=FactFreshness.TIME_SENSITIVE,
            original_question=query,
        )

    m = _WHEN_RE.search(q_with_q)
    if m:
        subject = m.group(1).strip()
        return AnswerTarget(
            entity=subject,
            attribute="date",
            expected_answer_type=AnswerType.DATE,
            domain=FactDomain.TECHNICAL
            if "released" in subject.lower() or "python" in subject.lower()
            else FactDomain.GENERAL,
            freshness=FactFreshness.HISTORICAL_STABLE,
            original_question=query,
        )

    m = _WHO_IS_RE.search(q_with_q)
    if m and "ceo" not in q.lower():
        return AnswerTarget(
            entity=m.group(1).strip(),
            attribute="identity",
            expected_answer_type=AnswerType.PERSON,
            domain=FactDomain.GENERAL,
            freshness=FactFreshness.TIME_SENSITIVE,
            original_question=query,
        )

    return None


def is_causal_or_analytical(query: str) -> bool:
    q = query.lower()
    causal = (
        "why ", "how did ", "what caused", "what factors", "what drove",
        "reason for", "explain why", "what led to",
    )
    return any(p in q for p in causal)


def entity_match_tokens(entity: str, text: str) -> bool:
    """Improved entity matching including short tokens like f1."""
    text_lower = text.lower()
    entity_lower = entity.lower()

    if entity_lower in text_lower:
        return True

    tokens = re.findall(r"[a-z0-9]+", entity_lower)
    significant = [t for t in tokens if len(t) > 2 or t in ("f1",)]
    if not significant:
        return bool(re.search(r"\b(19|20)\d{2}\b", entity_lower) and re.search(r"\b(19|20)\d{2}\b", text_lower))

    matches = sum(1 for t in significant if t in text_lower)
    required = min(2, len(significant)) if len(significant) > 1 else 1
    return matches >= required


_DOMAIN_OFFICIAL_SEARCH: dict[FactDomain, tuple[str, ...]] = {
    FactDomain.SPORTS: ("formula1.com", "fia.com"),
    FactDomain.FINANCIAL: ("sec.gov", "investor.apple.com", "apple.com"),
    FactDomain.CORPORATE: ("sec.gov",),
}


def build_targeted_search_query(target: AnswerTarget) -> str:
    """Augment search query with target-specific terms."""
    if target.attribute == "revenue":
        return f"{target.entity} {target.temporal_scope or ''} revenue SEC filing".strip()
    if target.attribute == "winner" and target.domain.value == "sports":
        if any(k in target.entity.lower() for k in ("f1", "formula")):
            return (
                f"{target.temporal_scope or ''} Formula One drivers championship "
                "standings winner"
            ).strip()

    parts = [target.original_question.rstrip("?")]
    if target.temporal_scope and target.temporal_scope.lower() not in parts[0].lower():
        parts.append(target.temporal_scope)
    if target.attribute == "winner" and "winner" not in parts[0].lower():
        parts.append("winner")
    return " ".join(parts)


def official_domains_for_target(target: AnswerTarget) -> tuple[str, ...]:
    """Preferred official domains for targeted fallback search."""
    return _DOMAIN_OFFICIAL_SEARCH.get(target.domain, ())
