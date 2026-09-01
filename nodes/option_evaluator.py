"""Option evaluation node (Phase 3B)."""

import logging

from langchain_anthropic import ChatAnthropic

from config import settings
from services.decision_framing_schemas import DecisionFrame
from services.knowledge_state_schemas import KnowledgeState
from services.option_evaluation import evaluate_options
from state import AgentState

logger = logging.getLogger(__name__)


def _get_llm():
    return ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0.0,
    )


async def option_evaluator_node(state: AgentState) -> AgentState:
    """
    Evidence-grounded option evaluation after Knowledge State.

    Skips cleanly when no decision frame, knowledge state, or concrete options.
    Fails open on evaluation errors.
    """
    frame_data = state.get("decision_frame")
    ks_data = state.get("knowledge_state")

    if not frame_data or not ks_data:
        state["option_evaluation"] = None
        state["option_evaluation_metrics"] = None
        state["current_node"] = "writer"
        return state

    frame = DecisionFrame(**frame_data)
    knowledge_state = KnowledgeState(**ks_data)
    material_claims = state.get("material_claims") or []

    evaluation, metrics = await evaluate_options(
        frame,
        knowledge_state,
        material_claims,
        llm=_get_llm(),
    )

    state["option_evaluation"] = (
        evaluation.model_dump(mode="json") if evaluation else None
    )
    state["option_evaluation_metrics"] = metrics.to_dict()

    cost = state.get("cost_metrics") or {}
    cost.update({
        "option_eval_generated": metrics.evaluations_generated,
        "option_eval_grounded": metrics.grounded_evaluation_count,
        "option_eval_skipped": metrics.evaluation_skipped,
        "option_eval_failed": metrics.evaluation_failed,
        "evaluation_time_ms": metrics.evaluation_time_ms,
    })
    state["cost_metrics"] = cost
    state["current_node"] = "writer"

    logger.info(
        "Option evaluator: options=%d criteria=%d evals=%d skipped=%s failed=%s (%.0fms)",
        metrics.option_count,
        metrics.criterion_count,
        metrics.evaluations_generated,
        metrics.evaluation_skipped,
        metrics.evaluation_failed,
        metrics.evaluation_time_ms,
    )

    return state
