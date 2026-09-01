"""Cross-source claim verification node."""

import logging

from langchain_anthropic import ChatAnthropic

from config import settings
from services.claim_verification import verify_material_claims
from state import AgentState

logger = logging.getLogger(__name__)


def _get_llm():
    return ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0.0,
    )


async def claim_verifier_node(state: AgentState) -> AgentState:
    """
    Verify material claims against cross-source validated evidence.

    Preserves origin SUPPORTS from claim extraction; adds cross-source links.
    """
    material_claims = state.get("material_claims") or []
    evidence = state.get("validated_evidence") or []
    sources = state.get("normalized_sources") or []
    origin_relations = state.get("claim_evidence_relations") or []
    run_id = state.get("research_run_id")

    if run_id is None or not material_claims:
        state["verification_results"] = []
        state["verification_metrics"] = {
            "material_claims_processed": 0,
            "skipped": True,
        }
        state["current_node"] = "critic"
        return state

    verifications, new_relations, metrics = await verify_material_claims(
        material_claims,
        evidence,
        sources,
        origin_relations,
        run_id,
        llm=_get_llm(),
        use_llm=True,
    )

    all_relations = list(origin_relations) + new_relations

    if state.get("is_run_persisted", False):
        from db.evidence_repositories import (
            VerificationRepository,
            get_claim_repo,
            is_persistence_enabled,
        )

        if is_persistence_enabled():
            try:
                if new_relations:
                    claim_repo = get_claim_repo()
                    saved_rels = await claim_repo.save_claim_evidence(new_relations)
                    all_relations = list(origin_relations) + saved_rels
                vrepo = VerificationRepository()
                verifications = await vrepo.save_verifications(verifications)
            except Exception as e:
                logger.warning("Failed to persist verification results: %s", e)
                metrics.failures.append({
                    "failure_type": "persistence_error",
                    "error": str(e),
                })

    state["verification_results"] = verifications
    state["claim_evidence_relations"] = all_relations
    state["verification_metrics"] = metrics.to_dict()

    cost = state.get("cost_metrics") or {}
    cost.update({
        "verification_batches": metrics.llm_batches,
        "cross_source_relations": metrics.cross_source_relations_added,
        "claims_supported": metrics.supported,
        "claims_partially_supported": metrics.partially_supported,
        "claims_contradicted": metrics.contradicted,
        "claims_uncertain": metrics.uncertain,
    })
    state["cost_metrics"] = cost
    state["current_node"] = "critic"

    logger.info(
        "Claim verifier: %d material claims, %d cross-source relations, "
        "%d supported, %d partial, %d uncertain (%.0fms)",
        metrics.material_claims_processed,
        metrics.cross_source_relations_added,
        metrics.supported,
        metrics.partially_supported,
        metrics.uncertain,
        metrics.processing_time_ms,
    )

    return state
