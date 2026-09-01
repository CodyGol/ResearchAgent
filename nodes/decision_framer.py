"""Decision framing node (Phase 3A)."""

import logging

from langchain_anthropic import ChatAnthropic

from config import settings
from services.decision_framing import frame_decision_query
from state import AgentState

logger = logging.getLogger(__name__)


def _get_llm():
    return ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0.0,
    )


async def decision_framer_node(state: AgentState) -> AgentState:
    """
    Detect decision orientation and extract DecisionFrame before planning.

    Fail-open: research continues with decision_frame=None on any failure.
    """
    query = state["user_query"]
    frame, metrics = await frame_decision_query(query, llm=_get_llm())

    state["decision_frame"] = frame.model_dump(mode="json") if frame else None
    state["decision_frame_metrics"] = metrics.to_dict()

    cost = state.get("cost_metrics") or {}
    cost.update({
        "decision_detected": metrics.decision_detected,
        "decision_type": metrics.decision_type,
        "framing_llm_calls": metrics.framing_llm_calls,
        "framing_time_ms": metrics.framing_time_ms,
        "framing_failed": metrics.framing_failed,
    })
    state["cost_metrics"] = cost
    state["current_node"] = "planner"

    logger.info(
        "Decision framer: detected=%s type=%s options=%d criteria=%d failed=%s (%.0fms)",
        metrics.decision_detected,
        metrics.decision_type,
        metrics.option_count,
        metrics.criteria_count,
        metrics.framing_failed,
        metrics.framing_time_ms,
    )

    return state
