"""Researcher node: Executes searches and aggregates results."""

from services.decision_framing_schemas import DecisionFrame
from services.decision_research_coverage import (
    DecisionCoverageMetrics,
    build_coverage_pair_specs,
    discover_pricing_domains_from_results,
    is_first_party_source,
    pin_coverage_sources,
    result_has_authoritative_hit,
)
from services.research_run_service import persist_sources_for_run
from services.research_sufficiency import prioritize_sources
from services.source_normalizer import normalize_search_results_with_metrics
from state import AgentState, ResearchResults
from tools.search import search_tavily_with_retry


async def _execute_coverage_search(
    spec,
    *,
    max_results: int,
    planner_domains: list[str] | None,
    metrics: DecisionCoverageMetrics,
) -> list:
    """Run authority-seeking search with at most one authoritative retry."""
    metrics.authoritative_search_attempts += 1
    coverage_max = max(max_results, 10) if spec.vendor_controlled else max_results
    pair_detail = {
        "option": spec.option_label,
        "criterion": spec.criterion_label,
        "primary_query": spec.primary_query,
        "retry_query": None,
        "retry_domains": None,
        "authoritative_hit_primary": False,
        "authoritative_hit_retry": False,
    }

    results = await search_tavily_with_retry(
        spec.primary_query,
        max_results=coverage_max,
        domains=planner_domains,
    )
    discovered_primary = discover_pricing_domains_from_results(results, spec.option_label)
    if discovered_primary:
        spec.official_domain_candidates = list(
            dict.fromkeys(spec.official_domain_candidates + discovered_primary)
        )

    primary_hit = result_has_authoritative_hit(results, spec)
    pair_detail["authoritative_hit_primary"] = primary_hit
    if primary_hit:
        metrics.authoritative_results_found += 1

    if (
        spec.vendor_controlled
        and not primary_hit
        and spec.retry_query
    ):
        metrics.authoritative_retries += 1
        pair_detail["retry_query"] = spec.retry_query
        retry_domains = list(
            dict.fromkeys(spec.official_domain_candidates + discovered_primary)
        ) or None
        # Unrestricted retry when inferred domains miss pricing (e.g. Anthropic -> claude.com)
        if retry_domains and not discovered_primary:
            retry_domains = None
        pair_detail["retry_domains"] = retry_domains
        retry_results = await search_tavily_with_retry(
            spec.retry_query,
            max_results=coverage_max,
            domains=retry_domains,
        )
        discovered_retry = discover_pricing_domains_from_results(
            retry_results, spec.option_label
        )
        if discovered_retry:
            spec.official_domain_candidates = list(
                dict.fromkeys(spec.official_domain_candidates + discovered_retry)
            )
        retry_hit = result_has_authoritative_hit(retry_results, spec)
        pair_detail["authoritative_hit_retry"] = retry_hit
        if retry_hit:
            metrics.authoritative_results_found += 1
        results.extend(retry_results)

    if not (pair_detail["authoritative_hit_primary"] or pair_detail["authoritative_hit_retry"]):
        metrics.decision_coverage_pairs_without_evidence += 1

    metrics.pair_details.append(pair_detail)
    return results


def _merge_coverage_metrics(
    prior: dict | None,
    current: DecisionCoverageMetrics,
) -> dict:
    """Accumulate coverage metrics across research iterations."""
    if not prior:
        return current.to_dict()
    merged = current.to_dict()
    for key in (
        "decision_coverage_pairs",
        "authoritative_search_attempts",
        "authoritative_retries",
        "authoritative_results_found",
        "authoritative_evidence_accepted",
        "decision_coverage_pairs_without_evidence",
    ):
        merged[key] = prior.get(key, 0) + merged.get(key, 0)
    merged["pair_details"] = (prior.get("pair_details") or []) + merged["pair_details"]
    return merged


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
    if state.get("decision_frame"):
        prioritize_auth = True

    # Skip further searches if research already sufficient
    if state.get("research_sufficient") and state.get("iteration_count", 0) > 0:
        state["current_node"] = "evidence_extractor"
        return state

    domains = []
    if hasattr(plan, "required_domains") and plan.required_domains:
        domains.extend(plan.required_domains)
    if hasattr(plan, "domains") and plan.domains:
        domains.extend(plan.domains)
    planner_domains = list(dict.fromkeys(domains)) if domains else None

    coverage_metrics = DecisionCoverageMetrics()
    prior_coverage_metrics = state.get("decision_coverage_metrics")
    coverage_specs = []
    frame_data = state.get("decision_frame")
    if frame_data:
        try:
            frame = DecisionFrame(**frame_data)
            coverage_specs = build_coverage_pair_specs(frame)
            coverage_metrics.decision_coverage_pairs = len(coverage_specs)
        except Exception:
            coverage_specs = []

    prior_results: list = []
    existing_research = state.get("research_results")
    if existing_research and state.get("iteration_count", 0) > 0:
        prior_results = list(existing_research.results)

    all_results = list(prior_results)
    coverage_query_norms = set()

    for spec in coverage_specs:
        coverage_query_norms.add(spec.primary_query.strip().lower())
        try:
            results = await _execute_coverage_search(
                spec,
                max_results=max_results,
                planner_domains=planner_domains,
                metrics=coverage_metrics,
            )
            all_results.extend(results)
        except Exception as e:
            state["error"] = f"Coverage search failed for '{spec.primary_query}': {str(e)}"
            state["current_node"] = "end"
            return state

    sub_queries = [
        q for q in plan.sub_queries[:max_queries]
        if q.strip().lower() not in coverage_query_norms
    ]

    for sub_query in sub_queries:
        try:
            results = await search_tavily_with_retry(
                query=sub_query,
                max_results=max_results,
                domains=planner_domains,
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
        sources = pin_coverage_sources(sources, coverage_specs)
        if coverage_specs:
            coverage_metrics.authoritative_evidence_accepted = sum(
                1
                for spec in coverage_specs
                for source in sources
                if is_first_party_source(
                    source,
                    spec.option_label,
                    extra_domains=spec.official_domain_candidates,
                )
            )

        # Cap sources to budget target after pinning authoritative coverage sources
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

        merged_coverage = _merge_coverage_metrics(prior_coverage_metrics, coverage_metrics)
        cost = state.get("cost_metrics") or {}
        cost["search_queries_executed"] = cost.get("search_queries_executed", 0) + len(coverage_specs) + len(sub_queries) + coverage_metrics.authoritative_retries
        cost["raw_sources"] = dedup_metrics.raw_sources_found
        cost["canonical_sources"] = len(saved)
        cost["decision_coverage_metrics"] = merged_coverage
        state["cost_metrics"] = cost
        state["decision_coverage_metrics"] = merged_coverage
    else:
        state["normalized_sources"] = None
        state["source_dedup_metrics"] = None

    state["current_node"] = "evidence_extractor"
    return state
