"""Query router node: classify complexity and assign research budget."""

import logging

from services.query_router import classify_query
from state import AgentState

logger = logging.getLogger(__name__)


async def router_node(state: AgentState) -> AgentState:
    """
    Classify query complexity and assign research budget before planning.

    Lightweight deterministic classification — no LLM call.
    """
    query = state["user_query"]
    classification = classify_query(query)

    state["query_classification"] = classification.model_dump(mode="json")
    route = classification.route.value
    state["current_node"] = "fast_path" if route == "simple_fact" else "planner"

    logger.info(
        "Query classified as %s / route=%s: %s",
        classification.complexity.value,
        route,
        classification.reason,
    )

    return state
