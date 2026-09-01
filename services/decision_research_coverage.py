"""Decision-aware research coverage helpers (no scoring, no new retrieval layer)."""

from __future__ import annotations

from services.decision_framing_schemas import DecisionFrame
from state import ResearchPlan


def _normalize_query(text: str) -> str:
    return " ".join(text.strip().lower().split())


def build_coverage_subqueries(frame: DecisionFrame) -> list[str]:
    """
    Deterministic option×primary-criterion research coverage queries.

    Ensures comparable evidence collection is attempted for each explicit pair.
    """
    explicit_options = [o for o in frame.options if o.origin == "explicit"]
    primary_criteria = [
        c for c in frame.criteria if c.origin == "explicit" and c.priority == "primary"
    ]
    if not explicit_options or not primary_criteria:
        return []

    queries: list[str] = []
    for option in explicit_options:
        for criterion in primary_criteria:
            queries.append(
                f"{option.label} enterprise {criterion.label} official API pricing documentation"
            )
    return queries


def merge_decision_coverage_into_plan(
    plan: ResearchPlan,
    frame: DecisionFrame,
    *,
    max_queries: int,
) -> ResearchPlan:
    """Prepend decision coverage queries so they survive budget trimming."""
    coverage = build_coverage_subqueries(frame)
    if not coverage:
        return plan

    seen = {_normalize_query(q) for q in plan.sub_queries}
    prepended: list[str] = []
    for query in coverage:
        norm = _normalize_query(query)
        if norm not in seen:
            prepended.append(query)
            seen.add(norm)

    merged_sub_queries = prepended + list(plan.sub_queries)
    merged_sub_queries = list(dict.fromkeys(merged_sub_queries))[:max_queries]

    merged_search_terms = list(dict.fromkeys(prepended + list(plan.search_terms)))

    return plan.model_copy(
        update={
            "sub_queries": merged_sub_queries,
            "search_terms": merged_search_terms,
        }
    )
