"""Evidence extractor node: extract and validate evidence from sources."""

import logging

from services.evidence_pipeline import process_sources_for_evidence
from services.research_sufficiency import prioritize_sources
from state import AgentState

logger = logging.getLogger(__name__)


async def evidence_extractor_node(state: AgentState) -> AgentState:
    """
    Extract, validate, and persist evidence from normalized sources.

    Respects evidence budget caps for simple questions.
    """
    sources = state.get("normalized_sources") or []
    run_id = state.get("research_run_id")
    query = state.get("user_query", "")

    classification = state.get("query_classification") or {}
    budget = classification.get("research_budget", {})
    max_evidence = budget.get("max_evidence_items")
    prioritize_auth = budget.get("prioritize_authoritative", False)

    if run_id is None:
        logger.warning("No research_run_id — skipping evidence extraction")
        state["validated_evidence"] = []
        state["evidence_metrics"] = None
        state["current_node"] = "claim_extractor"
        return state

    if not sources:
        logger.info("No sources available for evidence extraction")
        state["validated_evidence"] = []
        state["evidence_metrics"] = {"sources_processed": 0, "validated_count": 0}
        state["current_node"] = "claim_extractor"
        return state

    if prioritize_auth:
        sources = prioritize_sources(sources, authoritative_first=True)

    validated, metrics = await process_sources_for_evidence(
        sources=sources,
        research_question=query,
        research_run_id=run_id,
        is_persisted=state.get("is_run_persisted", False),
    )

    from services.evidence_context import assign_evidence_ids

    validated = assign_evidence_ids(validated)

    # Cap evidence items per budget
    if max_evidence and len(validated) > max_evidence:
        validated = validated[:max_evidence]
        metrics.validated_count = len(validated)

    state["validated_evidence"] = validated
    state["evidence_metrics"] = metrics.to_dict()
    state["current_node"] = "claim_extractor"

    cost = state.get("cost_metrics") or {}
    cost["evidence_items"] = len(validated)
    state["cost_metrics"] = cost

    logger.info(
        "Evidence extractor: %d validated from %d sources (%d rejected)",
        metrics.validated_count,
        metrics.sources_processed,
        metrics.rejected_count,
    )

    return state
