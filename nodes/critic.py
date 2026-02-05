"""Critic node: Evaluates research quality and decides if refinement is needed."""

from langchain_anthropic import ChatAnthropic

from config import settings
from state import AgentState, Critique
from utils.observability import trace_llm_call


async def critic_node(state: AgentState) -> AgentState:
    """
    Critic node: Evaluates research quality and determines if refinement needed.

    Uses Claude 4.5 Sonnet to evaluate research results for:
    - Freshness (staleness detection)
    - Bias detection
    - Completeness
    - Source diversity

    Args:
        state: Current agent state

    Returns:
        Updated state with critique populated and current_node set
    """
    results = state.get("research_results")
    plan = state.get("research_plan")
    iteration = state.get("iteration_count", 0)

    if not results:
        state["error"] = "Research results not found"
        state["current_node"] = "end"
        return state

    # Initialize LLM with structured output
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0.2,  # Very low temperature for consistent evaluation
    )

    # Trace LLM call
    with trace_llm_call("critic", "evaluate_research_quality") as span:
        try:
            # Prepare research results summary for critique
            results_summary = "\n\n".join(
                [
                    f"Result {i+1}:\nTitle: {r.title}\nURL: {r.url}\nContent: {r.content[:200]}..."
                    for i, r in enumerate(results.results[:10])  # Limit to first 10
                ]
            )

            system_prompt = """You are a research quality critic. Your task is to evaluate research results for:

1. **Freshness**: Are the results recent and up-to-date? Check for stale information.
2. **Bias**: Are there potential biases in the sources or content?
3. **Completeness**: Does the research cover the query comprehensively?
4. **Source Diversity**: Are sources from diverse, credible origins?

Provide a quality score (0-1) and specific issues/recommendations. Be strict but fair."""

            user_prompt = f"""Evaluate these research results for the query: "{plan.query if plan else 'Unknown query'}"

Research Results ({results.total_count} total):
{results_summary}

Provide a critique with:
- Quality score (0.0 to 1.0)
- Whether quality is sufficient (threshold: {settings.quality_threshold})
- List of specific issues found
- Recommendations for improvement if insufficient"""

            span.set_input({
                "query": plan.query if plan else "Unknown",
                "result_count": results.total_count,
                "iteration": iteration,
            })

            # Use structured output
            try:
                structured_llm = llm.with_structured_output(Critique)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                critique_result = await structured_llm.ainvoke(messages)
            except Exception:
                # Fallback: Parse from response
                response = await llm.ainvoke(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                # Parse critique from response
                import json
                import re

                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    try:
                        critique_data = json.loads(json_match.group())
                        critique_result = Critique(**critique_data)
                    except Exception:
                        # Fallback: Basic evaluation
                        critique_result = Critique(
                            quality_score=0.7,
                            is_sufficient=True,
                            issues=[],
                            recommendations=[],
                        )
                else:
                    # Fallback: Basic evaluation
                    critique_result = Critique(
                        quality_score=0.7,
                        is_sufficient=True,
                        issues=[],
                        recommendations=[],
                    )

            # Ensure is_sufficient is set based on quality_threshold
            critique_result.is_sufficient = (
                critique_result.quality_score >= settings.quality_threshold
            )

            span.set_output({
                "critique": critique_result.model_dump(),
            })

            state["critique"] = critique_result

            # Check iteration limit (Rule 2: Prevent infinite loops)
            if not critique_result.is_sufficient and iteration >= settings.max_research_iterations:
                # Force proceed to writer even if quality is low
                state["current_node"] = "writer"
                return state

            # Decision: loop back to researcher or proceed to writer
            if critique_result.is_sufficient:
                state["current_node"] = "writer"
            else:
                state["iteration_count"] = iteration + 1
                state["current_node"] = "researcher"  # Recursive loop

            return state

        except Exception as e:
            span.set_error(e)
            state["error"] = f"Critic failed: {str(e)}"
            state["current_node"] = "end"
            return state
