"""Query complexity classification and research budget assignment."""

import re
from enum import Enum

from pydantic import BaseModel, Field

from services.fact_target import AnswerTarget, extract_fact_target, is_causal_or_analytical


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    STANDARD = "standard"
    DEEP = "deep"


class ExecutionRoute(str, Enum):
    SIMPLE_FACT = "simple_fact"
    STANDARD = "standard"
    DEEP = "deep"


class ClaimDepth(str, Enum):
    MINIMAL = "minimal"
    MODERATE = "moderate"
    BROAD = "broad"


class ResearchBudget(BaseModel):
    """Configurable research limits per complexity class."""

    max_search_queries: int = Field(..., ge=1)
    max_results_per_search: int = Field(..., ge=1)
    target_sources: int = Field(..., ge=1)
    max_iterations: int = Field(..., ge=0)
    claim_depth: ClaimDepth = ClaimDepth.MODERATE
    max_evidence_items: int | None = Field(
        None, description="Cap evidence extraction; None = no cap"
    )
    enable_short_circuit: bool = True
    prioritize_authoritative: bool = False


class QueryClassification(BaseModel):
    """Routing decision for adaptive research."""

    complexity: QueryComplexity
    route: ExecutionRoute
    direct_answer_expected: bool = False
    reason: str
    research_budget: ResearchBudget
    fact_target: AnswerTarget | None = None


# Fast path budget — maxima, not targets
SIMPLE_FACT_BUDGET = ResearchBudget(
    max_search_queries=1,
    max_results_per_search=3,
    target_sources=3,
    max_iterations=0,
    claim_depth=ClaimDepth.MINIMAL,
    max_evidence_items=2,
    enable_short_circuit=True,
    prioritize_authoritative=True,
)

BUDGETS: dict[QueryComplexity, ResearchBudget] = {
    QueryComplexity.SIMPLE: SIMPLE_FACT_BUDGET.model_copy(),
    QueryComplexity.STANDARD: ResearchBudget(
        max_search_queries=3,
        max_results_per_search=5,
        target_sources=8,
        max_iterations=1,
        claim_depth=ClaimDepth.MODERATE,
        max_evidence_items=15,
        enable_short_circuit=True,
        prioritize_authoritative=False,
    ),
    QueryComplexity.DEEP: ResearchBudget(
        max_search_queries=5,
        max_results_per_search=5,
        target_sources=15,
        max_iterations=3,
        claim_depth=ClaimDepth.BROAD,
        max_evidence_items=None,
        enable_short_circuit=False,
        prioritize_authoritative=False,
    ),
}

_DEEP_PATTERNS = re.compile(
    r"\b("
    r"should\s+(we|a|the|\w+).*?(acquire|enter|invest|expand)|"
    r"compare|versus|vs\.?|strengths?\s+and\s+weaknesses|"
    r"strategic\s+risks?|decision|recommend|evaluate\s+whether|"
    r"pros?\s+and\s+cons?|trade-?offs?|implications?\s+for|"
    r"over\s+(the\s+)?next\s+\d+\s+years?"
    r")\b",
    re.IGNORECASE,
)

_SIMPLE_PATTERNS = re.compile(
    r"^(who|what|when|where)\s+(is|was|were|are)\s+",
    re.IGNORECASE,
)

_SIMPLE_FACTUAL = re.compile(
    r"\b(capital of|won the|revenue (in|for|was)|population of|"
    r"ceo of|founded in|born in|located in|first released)\b",
    re.IGNORECASE,
)


def classify_query(query: str) -> QueryClassification:
    """
    Classify query complexity, route, and assign research budget.

    SIMPLE_FACT route for narrow factual questions with extractable targets.
    """
    q = query.strip()
    q_lower = q.lower()

    if _DEEP_PATTERNS.search(q):
        return QueryClassification(
            complexity=QueryComplexity.DEEP,
            route=ExecutionRoute.DEEP,
            direct_answer_expected=False,
            reason="Question requires synthesis, comparison, or decision support",
            research_budget=BUDGETS[QueryComplexity.DEEP].model_copy(),
        )

    if is_causal_or_analytical(q):
        return QueryClassification(
            complexity=QueryComplexity.STANDARD,
            route=ExecutionRoute.STANDARD,
            direct_answer_expected=False,
            reason="Causal or analytical question — not a direct fact",
            research_budget=BUDGETS[QueryComplexity.STANDARD].model_copy(),
        )

    fact_target = extract_fact_target(q)
    is_simple_shape = (
        _SIMPLE_PATTERNS.match(q) or _SIMPLE_FACTUAL.search(q)
    ) and len(q.split()) <= 15

    if fact_target and is_simple_shape:
        return QueryClassification(
            complexity=QueryComplexity.SIMPLE,
            route=ExecutionRoute.SIMPLE_FACT,
            direct_answer_expected=True,
            reason="Narrow factual question with identifiable answer target",
            research_budget=SIMPLE_FACT_BUDGET.model_copy(),
            fact_target=fact_target,
        )

    if is_simple_shape:
        return QueryClassification(
            complexity=QueryComplexity.SIMPLE,
            route=ExecutionRoute.STANDARD,
            direct_answer_expected=True,
            reason="Simple question without extractable target — standard pipeline",
            research_budget=BUDGETS[QueryComplexity.SIMPLE].model_copy(),
        )

    if any(
        kw in q_lower
        for kw in ("compare", "versus", " vs ", "drivers of", "factors", "analysis")
    ):
        return QueryClassification(
            complexity=QueryComplexity.STANDARD,
            route=ExecutionRoute.STANDARD,
            direct_answer_expected=False,
            reason="Moderate synthesis or comparison required",
            research_budget=BUDGETS[QueryComplexity.STANDARD].model_copy(),
        )

    return QueryClassification(
        complexity=QueryComplexity.STANDARD,
        route=ExecutionRoute.STANDARD,
        direct_answer_expected=False,
        reason="Default standard research depth",
        research_budget=BUDGETS[QueryComplexity.STANDARD].model_copy(),
    )
