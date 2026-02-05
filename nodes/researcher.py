"""Researcher node: Executes searches and aggregates results."""

from state import AgentState, ResearchResults
from tools.search import search_tavily_with_retry


async def researcher_node(state: AgentState) -> AgentState:
    """
    Researcher node: Executes searches based on research plan.

    Uses Tavily API with retry logic to execute web searches for each sub-query
    in the research plan.

    Args:
        state: Current agent state

    Returns:
        Updated state with research_results populated
    """
    plan = state.get("research_plan")
    if not plan:
        state["error"] = "Research plan not found"
        state["current_node"] = "end"
        return state

    # Execute searches for each sub-query
    # Combine domain filters: required_domains (academic) + domains (technical)
    # This ensures we prioritize primary sources
    domains = []
    if hasattr(plan, 'required_domains') and plan.required_domains:
        domains.extend(plan.required_domains)
    if hasattr(plan, 'domains') and plan.domains:
        domains.extend(plan.domains)
    # Remove duplicates while preserving order
    domains = list(dict.fromkeys(domains)) if domains else None
    
    all_results = []
    for sub_query in plan.sub_queries:
        try:
            results = await search_tavily_with_retry(
                query=sub_query,
                max_results=5,
                domains=domains,
            )
            all_results.extend(results)
        except Exception as e:
            # Fail loudly (Rule 3.B)
            state["error"] = f"Search failed for '{sub_query}': {str(e)}"
            state["current_node"] = "end"
            return state

    state["research_results"] = ResearchResults(
        results=all_results,
        total_count=len(all_results),
    )
    state["current_node"] = "critic"
    return state
