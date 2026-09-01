"""Knowledge state derivation node (Phase 2D)."""

import logging

from services.knowledge_state import derive_knowledge_state, knowledge_state_snapshot
from state import AgentState, Critique

logger = logging.getLogger(__name__)


async def knowledge_state_node(state: AgentState) -> AgentState:
    """
    Derive final knowledge state after Critic exits to Writer.

    Applies only to full-pipeline runs with verification results.
    Fast-path runs bypass this node.
    """
    material_claims = state.get("material_claims") or []
    verification_results = state.get("verification_results") or []
    relations = state.get("claim_evidence_relations") or []
    critique = state.get("critique")

    if not verification_results:
        state["knowledge_state"] = None
        state["current_node"] = "writer"
        return state

    critique_model = critique if isinstance(critique, Critique) else None
    if critique is not None and not isinstance(critique, Critique):
        critique_model = Critique(**critique)

    knowledge_state, updated_verifications = derive_knowledge_state(
        material_claims=material_claims,
        verification_results=verification_results,
        claim_evidence_relations=relations,
        critique=critique_model,
    )

    if state.get("is_run_persisted", False):
        from db.evidence_repositories import VerificationRepository, is_persistence_enabled

        if is_persistence_enabled():
            try:
                vrepo = VerificationRepository()
                updated_verifications = await vrepo.save_verifications(updated_verifications)
            except Exception as exc:
                logger.warning("Failed to persist knowledge categories: %s", exc)
                metrics = dict(knowledge_state.metrics)
                metrics["persistence_error"] = str(exc)
                knowledge_state = knowledge_state.model_copy(update={"metrics": metrics})

    state["verification_results"] = updated_verifications
    state["knowledge_state"] = knowledge_state_snapshot(knowledge_state)

    cost = state.get("cost_metrics") or {}
    cost.update({
        "knowledge_known": knowledge_state.metrics.get("known_count", 0),
        "knowledge_likely": knowledge_state.metrics.get("likely_count", 0),
        "knowledge_disputed": knowledge_state.metrics.get("disputed_count", 0),
        "knowledge_unknown": knowledge_state.metrics.get("unknown_count", 0),
        "knowledge_contradicted": knowledge_state.metrics.get("contradicted_count", 0),
        "knowledge_unverifiable": knowledge_state.metrics.get("unverifiable_count", 0),
        "knowledge_gaps": knowledge_state.metrics.get("information_gap_count", 0),
        "knowledge_orphan_claims": knowledge_state.metrics.get("orphan_material_claims", 0),
    })
    state["cost_metrics"] = cost
    state["current_node"] = "writer"

    logger.info(
        "Knowledge state: known=%d likely=%d disputed=%d unknown=%d "
        "contradicted=%d unverifiable=%d gaps=%d orphans=%d (%.0fms)",
        knowledge_state.metrics.get("known_count", 0),
        knowledge_state.metrics.get("likely_count", 0),
        knowledge_state.metrics.get("disputed_count", 0),
        knowledge_state.metrics.get("unknown_count", 0),
        knowledge_state.metrics.get("contradicted_count", 0),
        knowledge_state.metrics.get("unverifiable_count", 0),
        knowledge_state.metrics.get("information_gap_count", 0),
        knowledge_state.metrics.get("orphan_material_claims", 0),
        knowledge_state.metrics.get("derivation_time_ms", 0),
    )

    return state
