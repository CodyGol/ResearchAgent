"""Pipeline initialization helpers."""

from domain.models import ResearchRunStatus
from services.cost_metrics import CostMetrics
from services.query_router import classify_query
from services.research_run_service import RunContext, finalize_research_run, start_research_run
from state import AgentState


async def create_initial_state(query: str) -> tuple[AgentState, RunContext]:
    """
    Create initial agent state with a research run.

    Args:
        query: User research question

    Returns:
        Tuple of (initial AgentState, RunContext for finalization)
    """
    ctx = await start_research_run(query)
    classification = classify_query(query)
    cost = CostMetrics(complexity_class=classification.route.value)
    state: AgentState = {
        "user_query": query,
        "query_classification": classification.model_dump(mode="json"),
        "fast_path_metrics": None,
        "escalate_to_standard": False,
        "escalated_from_fast_path": False,
        "escalation_reason": None,
        "research_run_id": ctx.run.id,
        "is_run_persisted": ctx.is_persisted,
        "normalized_sources": None,
        "validated_evidence": None,
        "evidence_metrics": None,
        "validated_claims": None,
        "material_claims": None,
        "claim_evidence_relations": None,
        "verification_results": None,
        "verification_metrics": None,
        "knowledge_state": None,
        "decision_frame": None,
        "decision_frame_metrics": None,
        "claim_metrics": None,
        "source_dedup_metrics": None,
        "report_metrics": None,
        "cost_metrics": cost.to_dict(),
        "research_sufficient": False,
        "research_plan": None,
        "research_results": None,
        "critique": None,
        "final_report": None,
        "current_node": "router",
        "iteration_count": 0,
        "error": None,
    }
    return state, ctx


async def finalize_from_state(state: AgentState, ctx: RunContext) -> None:
    """Finalize research run based on terminal agent state."""
    if state.get("research_run_id") is None:
        return

    ctx.sources = state.get("normalized_sources") or []
    evidence_metrics = state.get("evidence_metrics") or {}
    claim_metrics = state.get("claim_metrics") or {}
    cost_metrics = state.get("cost_metrics") or {}
    classification = state.get("query_classification") or {}
    status = (
        ResearchRunStatus.FAILED if state.get("error") else ResearchRunStatus.COMPLETED
    )
    await finalize_research_run(
        ctx,
        status=status,
        iteration_count=state.get("iteration_count", 0),
        evidence_count=evidence_metrics.get("validated_count", 0),
        claims_count=claim_metrics.get("unique_claims_persisted", 0),
        failed_validations=evidence_metrics.get("validation_failures", 0),
        metadata={
            "source_count": len(ctx.sources),
            "has_report": state.get("final_report") is not None,
            "evidence_metrics": evidence_metrics,
            "claim_metrics": claim_metrics,
            "cost_metrics": cost_metrics,
            "query_classification": classification,
            "fast_path_metrics": state.get("fast_path_metrics"),
            "escalated_from_fast_path": state.get("escalated_from_fast_path", False),
            "escalation_reason": state.get("escalation_reason"),
            "material_claims_count": claim_metrics.get("material_claims_count", 0),
            "verification_metrics": state.get("verification_metrics"),
            "knowledge_state": state.get("knowledge_state"),
            "knowledge_state_metrics": (
                (state.get("knowledge_state") or {}).get("metrics")
            ),
            "decision_frame": state.get("decision_frame"),
            "decision_frame_metrics": state.get("decision_frame_metrics"),
        },
        error=state.get("error"),
    )
