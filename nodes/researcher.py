"""Researcher node: Executes searches and aggregates results."""

from services.research_run_service import persist_sources_for_run
from services.research_sufficiency import prioritize_sources
from services.source_normalizer import normalize_search_results_with_metrics
from state import AgentState, ResearchResults
from tools.search import search_tavily_with_retry


async def researcher_node(state: AgentState) -> AgentState:
    """
    Researcher node: Executes searches based on research plan and budget.

    Respects complexity-based search limits and authoritative source prioritization.
    """
    plan = state.get("research_plan")
    if not plan:
        state["error"] = "Research plan not found"
        state["current_node"] = "end"
        return state

    classification = state.get("query_classification") or {}
    budget = classification.get("research_budget", {})
    max_queries = budget.get("max_search_queries", len(plan.sub_queries))
    max_results = budget.get("max_results_per_search", 5)
    target_sources = budget.get("target_sources", 15)
    prioritize_auth = budget.get("prioritize_authoritative", False)

    # Skip further searches if research already sufficient
    if state.get("research_sufficient") and state.get("iteration_count", 0) > 0:
        state["current_node"] = "evidence_extractor"
        return state

    domains = []
    if hasattr(plan, "required_domains") and plan.required_domains:
        domains.extend(plan.required_domains)
    if hasattr(plan, "domains") and plan.domains:
        domains.extend(plan.domains)
    domains = list(dict.fromkeys(domains)) if domains else None

    sub_queries = plan.sub_queries[:max_queries]

    all_results = []
    for sub_query in sub_queries:
        try:
            results = await search_tavily_with_retry(
                query=sub_query,
                max_results=max_results,
                domains=domains,
            )
            all_results.extend(results)
        except Exception as e:
            state["error"] = f"Search failed for '{sub_query}': {str(e)}"
            state["current_node"] = "end"
            return state

    state["research_results"] = ResearchResults(
        results=all_results,
        total_count=len(all_results),
    )

    run_id = state.get("research_run_id")
    if run_id is not None:
        sources, dedup_metrics = normalize_search_results_with_metrics(
            all_results, run_id
        )
        sources = prioritize_sources(sources, authoritative_first=prioritize_auth)
        # Cap sources to budget target
        if target_sources and len(sources) > target_sources:
            sources = sources[:target_sources]

        state["source_dedup_metrics"] = dedup_metrics.to_dict()
        saved = await persist_sources_for_run(
            run_id,
            plan.query,
            sources,
            is_persisted=state.get("is_run_persisted", False),
        )
        state["normalized_sources"] = saved

        # Update cost metrics
        cost = state.get("cost_metrics") or {}
        cost["search_queries_executed"] = cost.get("search_queries_executed", 0) + len(sub_queries)
        cost["raw_sources"] = dedup_metrics.raw_sources_found
        cost["canonical_sources"] = len(saved)
        state["cost_metrics"] = cost
    else:
        state["normalized_sources"] = None
        state["source_dedup_metrics"] = None

    state["current_node"] = "evidence_extractor"
    return state
