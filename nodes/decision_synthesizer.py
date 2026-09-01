"""Decision synthesis node (Phase 3C)."""

import logging

from langchain_anthropic import ChatAnthropic

from config import settings
from services.decision_framing_schemas import DecisionFrame
from services.decision_synthesis import skip_metrics, synthesize_decision
from services.knowledge_state_schemas import KnowledgeState
from services.option_evaluation_schemas import OptionEvaluation
from state import AgentState

logger = logging.getLogger(__name__)


def _get_llm():
    return ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0.0,
    )


async def decision_synthesizer_node(state: AgentState) -> AgentState:
    """
    Evidence-grounded decision synthesis after Option Evaluation.

    Skips when no option evaluation or evaluation failed/skipped.
    Fails open on synthesis errors.
    """
    frame_data = state.get("decision_frame")
    oe_data = state.get("option_evaluation")
    oe_metrics = state.get("option_evaluation_metrics") or {}
    ks_data = state.get("knowledge_state")

    if not frame_data or not oe_data:
        state["decision_synthesis"] = None
        state["decision_synthesis_metrics"] = skip_metrics("no_option_evaluation").to_dict()
        state["current_node"] = "writer"
        return state

    if oe_metrics.get("evaluation_failed") or oe_metrics.get("evaluation_skipped"):
        reason = oe_metrics.get("evaluation_skipped_reason") or oe_metrics.get("failure_reason") or "option_evaluation_unavailable"
        state["decision_synthesis"] = None
        state["decision_synthesis_metrics"] = skip_metrics(reason).to_dict()
        state["current_node"] = "writer"
        return state

    frame = DecisionFrame(**frame_data)
    option_evaluation = OptionEvaluation(**oe_data)
    knowledge_state = KnowledgeState(**ks_data) if ks_data else KnowledgeState()
    material_claims = state.get("material_claims") or []

    synthesis, metrics = await synthesize_decision(
        frame,
        option_evaluation,
        knowledge_state,
        material_claims,
        llm=_get_llm(),
    )

    state["decision_synthesis"] = (
        synthesis.model_dump(mode="json") if synthesis else None
    )
    state["decision_synthesis_metrics"] = metrics.to_dict()

    cost = state.get("cost_metrics") or {}
    cost.update({
        "synthesis_recommendation_status": metrics.recommendation_status,
        "synthesis_recommendation_present": metrics.recommendation_present,
        "synthesis_failed": metrics.synthesis_failed,
        "synthesis_skipped": metrics.synthesis_skipped,
        "synthesis_time_ms": metrics.synthesis_time_ms,
    })
    state["cost_metrics"] = cost
    state["current_node"] = "writer"

    logger.info(
        "Decision synthesizer: status=%s option=%s failed=%s skipped=%s (%.0fms)",
        metrics.recommendation_status,
        metrics.recommendation_present,
        metrics.synthesis_failed,
        metrics.synthesis_skipped,
        metrics.synthesis_time_ms,
    )

    return state
