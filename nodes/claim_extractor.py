"""Claim extractor node: extract atomic claims from validated evidence."""

import logging

from langchain_anthropic import ChatAnthropic

from config import settings
from services.claim_pipeline import process_evidence_for_claims
from state import AgentState

logger = logging.getLogger(__name__)


def _get_llm():
    return ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0.0,
    )


async def claim_extractor_node(state: AgentState) -> AgentState:
    """
    Extract, validate, deduplicate, and persist claims from validated evidence.

    Uses complexity-based claim depth and batch validation.
    """
    evidence = state.get("validated_evidence") or []
    run_id = state.get("research_run_id")
    query = state.get("user_query", "")
    sources = state.get("normalized_sources") or []

    classification = state.get("query_classification") or {}
    budget = classification.get("research_budget", {})
    claim_depth = budget.get("claim_depth", "moderate")

    if run_id is None:
        logger.warning("No research_run_id — skipping claim extraction")
        state["validated_claims"] = []
        state["material_claims"] = []
        state["claim_evidence_relations"] = []
        state["claim_metrics"] = None
        state["current_node"] = "claim_verifier"
        return state

    if not evidence:
        logger.info("No validated evidence available for claim extraction")
        state["validated_claims"] = []
        state["material_claims"] = []
        state["claim_evidence_relations"] = []
        state["claim_metrics"] = {
            "evidence_items_processed": 0,
            "claims_accepted": 0,
        }
        state["current_node"] = "claim_verifier"
        return state

    claims, material_claims, relations, metrics = await process_evidence_for_claims(
        validated_evidence=evidence,
        research_question=query,
        research_run_id=run_id,
        is_persisted=state.get("is_run_persisted", False),
        sources=sources,
        llm=_get_llm(),
        model_name=settings.model_name,
        use_llm_validation=True,
        claim_depth=claim_depth,
        use_batch_validation=True,
    )

    state["validated_claims"] = claims
    state["material_claims"] = material_claims
    state["claim_evidence_relations"] = relations
    state["claim_metrics"] = metrics.to_dict()

    cost = state.get("cost_metrics") or {}
    cost.update({
        "candidate_claims": metrics.candidate_claims_generated,
        "material_claims": metrics.material_claims_count,
        "deterministic_rejects": metrics.claims_rejected_deterministic,
        "relevance_rejects": metrics.claims_rejected_relevance,
        "llm_validation_calls": metrics.llm_validation_calls,
        "validation_batches": metrics.validation_batches,
    })
    state["cost_metrics"] = cost

    state["current_node"] = "claim_verifier"

    logger.info(
        "Claim extractor: %d unique (%d material) from %d evidence "
        "(%d rejected, %d LLM calls, %d batches)",
        metrics.unique_claims_persisted,
        metrics.material_claims_count,
        metrics.evidence_items_processed,
        metrics.claims_rejected,
        metrics.llm_validation_calls,
        metrics.validation_batches,
    )

    return state
