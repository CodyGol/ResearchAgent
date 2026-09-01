"""SIMPLE_FACT fast path node — bypasses full claim/critic/writer pipeline."""

import logging

from langchain_anthropic import ChatAnthropic

from config import settings
from services.fast_path import run_fast_path
from services.fast_writer import build_fast_answer
from services.fact_target import AnswerTarget
from state import AgentState

logger = logging.getLogger(__name__)


def _get_llm():
    return ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0.0,
    )


async def fast_path_node(state: AgentState) -> AgentState:
    """
    Execute SIMPLE_FACT fast path.

    On success: sets final_report and routes to END.
    On failure: sets escalate_to_standard and routes to full pipeline.
    """
    query = state["user_query"]
    run_id = state.get("research_run_id")
    classification = state.get("query_classification") or {}
    target_data = classification.get("fact_target")

    if not target_data or run_id is None:
        state["escalate_to_standard"] = True
        state["escalation_reason"] = "Missing fact target or run ID"
        state["current_node"] = "planner"
        return state

    target = AnswerTarget(**target_data)

    result = await run_fast_path(
        query,
        target,
        run_id,
        is_persisted=state.get("is_run_persisted", False),
        llm=_get_llm(),
    )

    cost = state.get("cost_metrics") or {}
    cost.update(result.metrics.to_dict())
    cost["route"] = "simple_fact"
    cost["fast_path_entered"] = True
    state["cost_metrics"] = cost
    state["fast_path_metrics"] = result.metrics.to_dict()

    if result.escalate or not result.success:
        logger.info(
            "Fast path escalating to STANDARD: %s",
            result.escalation_reason,
        )
        state["escalate_to_standard"] = True
        state["escalated_from_fast_path"] = True
        state["escalation_reason"] = result.escalation_reason
        # Preserve any evidence gathered
        if result.evidence:
            state["validated_evidence"] = result.evidence
        state["current_node"] = "planner"
        return state

    # Success — build concise answer
    evidence = result.evidence
    core_claim = result.core_claim

    display_id = evidence[0].metadata.get("display_id", "E1") if evidence else "E1"
    report = build_fast_answer(
        target,
        evidence[0],
        core_claim,
        result.sources or state.get("normalized_sources") or [],
        fact_value=result.fact_value,
        evidence_display_id=display_id,
    )

    state["validated_evidence"] = evidence
    state["normalized_sources"] = result.sources or state.get("normalized_sources")
    state["validated_claims"] = [core_claim] if core_claim else []
    state["material_claims"] = [core_claim] if core_claim else []
    state["claim_metrics"] = {
        "claims_accepted": 1 if core_claim else 0,
        "unique_claims_persisted": 1 if core_claim else 0,
        "material_claims_count": 1 if core_claim else 0,
        "candidate_claims_generated": 1,
        "full_claim_extractor_skipped": True,
        "llm_validation_calls": 0,
        "validation_batches": 0,
    }
    state["evidence_metrics"] = {
        "validated_count": len(evidence),
        "sources_processed": result.metrics.sources_processed,
        "fast_path": True,
    }
    state["research_sufficient"] = True
    state["final_report"] = report
    state["report_metrics"] = report.report_metrics
    state["current_node"] = "end"

    logger.info(
        "Fast path complete: %d evidence, %d core claim, %.0fms",
        len(evidence),
        1 if core_claim else 0,
        result.metrics.processing_time_ms,
    )

    return state
