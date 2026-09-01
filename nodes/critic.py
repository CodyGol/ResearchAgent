"""Critic node: Evidence-grounded research quality evaluation."""

from langchain_anthropic import ChatAnthropic

from config import settings
from services.evidence_context import assign_evidence_ids, format_evidence_for_prompt
from services.query_router import BUDGETS, QueryComplexity, ResearchBudget
from services.research_sufficiency import check_research_sufficiency
from state import AgentState, Critique
from utils.observability import trace_llm_call
from utils.runtime_date import temporal_context_block


async def critic_node(state: AgentState) -> AgentState:
    """
    Evidence-grounded Critic: evaluates validated evidence quality.

    Primary input is validated evidence — NOT raw search result count.
  """
    plan = state.get("research_plan")
    iteration = state.get("iteration_count", 0)
    evidence_list = state.get("validated_evidence") or []
    sources = state.get("normalized_sources") or []
    evidence_metrics = state.get("evidence_metrics") or {}
    classification = state.get("query_classification") or {}
    budget = classification.get("research_budget", {})
    complexity_str = classification.get("complexity", "standard")
    try:
        complexity = QueryComplexity(complexity_str)
    except ValueError:
        complexity = QueryComplexity.STANDARD
    max_iterations = budget.get("max_iterations", settings.max_research_iterations)

    if not plan:
        state["error"] = "Research plan not found"
        state["current_node"] = "end"
        return state

    # Assign stable evidence IDs for downstream traceability
    evidence_list = assign_evidence_ids(evidence_list)
    state["validated_evidence"] = evidence_list

    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0.2,
    )

    evidence_block = format_evidence_for_prompt(evidence_list, sources)
    validated_count = len(evidence_list)
    raw_search_count = state.get("research_results")
    raw_count = raw_search_count.total_count if raw_search_count else 0

    with trace_llm_call("critic", "evaluate_evidence_quality") as span:
        try:
            system_prompt = """You are an evidence-grounded research quality critic.

Your job is to evaluate VALIDATED EVIDENCE — not raw search result volume.

Evaluate these dimensions:

1. **Coverage**: Does the validated evidence address the research question?
2. **Directness**: Does evidence directly answer the question, or is it tangential?
3. **Source quality**: Are key facts supported by credible sources?
4. **Source diversity**: Are we over-relying on one publisher or source family?
5. **Temporal alignment**: Does evidence match the requested time period?
6. **Redundancy**: Are multiple items merely repeating the same fact?
7. **Potential conflicts**: Do evidence items appear to disagree?
8. **Missing evidence**: What important parts remain unsupported?

CRITICAL RULES:
- 25 search results with 2 useful evidence items is NOT the same as 8 independent high-quality evidence items.
- Do NOT rate quality highly based on search result count alone.
- Be strict when evidence is thin, redundant, temporally misaligned, or conflicting.
- quality_score must reflect EVIDENCE quality, not search breadth."""

            user_prompt = f"""Research question: "{plan.query}"

{temporal_context_block()}

Sub-queries investigated:
{chr(10).join(f"- {sq}" for sq in plan.sub_queries)}

Evidence extraction metrics:
- Raw search results retrieved: {raw_count}
- Validated evidence items: {validated_count}
- Sources with evidence: {evidence_metrics.get('sources_with_evidence', 0)}
- Rejected candidates: {evidence_metrics.get('rejected_count', 0)}

VALIDATED EVIDENCE (authoritative factual substrate):
{evidence_block}

Evaluate evidence quality. Threshold for sufficiency: {settings.quality_threshold}

Provide structured assessment including potential_conflicts and unsupported_areas."""

            span.set_input({
                "query": plan.query,
                "validated_evidence_count": validated_count,
                "raw_search_count": raw_count,
                "iteration": iteration,
            })

            try:
                structured_llm = llm.with_structured_output(Critique)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                critique_result = await structured_llm.ainvoke(messages)
            except Exception:
                import json
                import re

                response = await llm.ainvoke(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
                if json_match:
                    try:
                        critique_data = json.loads(json_match.group())
                        critique_result = Critique(**critique_data)
                    except Exception:
                        critique_result = _fallback_critique(validated_count)
                else:
                    critique_result = _fallback_critique(validated_count)

            # Penalize insufficient evidence deterministically
            if validated_count == 0:
                critique_result.quality_score = min(critique_result.quality_score, 0.3)
                critique_result.is_sufficient = False
                if "No validated evidence" not in str(critique_result.issues):
                    critique_result.issues.append("No validated evidence available")

            # Research sufficiency short-circuit for simple/standard questions
            research_budget = (
                ResearchBudget(**budget) if budget else BUDGETS[complexity]
            )
            sufficiency = check_research_sufficiency(
                plan.query,
                evidence_list,
                sources,
                complexity=complexity,
                budget=research_budget,
                potential_conflicts=critique_result.potential_conflicts,
            )

            if sufficiency.is_sufficient:
                critique_result.is_sufficient = True
                critique_result.quality_score = max(
                    critique_result.quality_score, settings.quality_threshold
                )
                state["research_sufficient"] = True
                cost = state.get("cost_metrics") or {}
                cost["short_circuited"] = True
                cost["short_circuit_reason"] = sufficiency.reason
                state["cost_metrics"] = cost
            else:
                critique_result.is_sufficient = (
                    critique_result.quality_score >= settings.quality_threshold
                    and validated_count > 0
                )

            span.set_output({
                "critique": critique_result.model_dump(),
                "sufficiency": sufficiency.reason,
            })
            state["critique"] = critique_result

            if not critique_result.is_sufficient and iteration >= max_iterations:
                state["current_node"] = "writer"
                return state

            if critique_result.is_sufficient:
                state["current_node"] = "writer"
            else:
                state["iteration_count"] = iteration + 1
                state["current_node"] = "researcher"

            return state

        except Exception as e:
            span.set_error(e)
            state["error"] = f"Critic failed: {str(e)}"
            state["current_node"] = "end"
            return state


def _fallback_critique(validated_count: int) -> Critique:
    """Conservative fallback when structured output fails."""
    score = 0.6 if validated_count >= 3 else 0.4 if validated_count >= 1 else 0.2
    return Critique(
        quality_score=score,
        is_sufficient=score >= settings.quality_threshold and validated_count > 0,
        coverage="Unable to fully assess — structured critique failed",
        issues=["Structured critique output failed; using conservative fallback"],
        recommendations=[],
    )
