"""Writer node: Synthesizes final report from research results."""

from langchain_anthropic import ChatAnthropic

from config import settings
from state import AgentState, FinalReport
from utils.observability import trace_llm_call


async def writer_node(state: AgentState) -> AgentState:
    """
    Writer node: Synthesizes final report from approved research.

    Uses Claude 4.5 Sonnet to synthesize a comprehensive report from research results.

    Args:
        state: Current agent state

    Returns:
        Updated state with final_report populated
    """
    plan = state.get("research_plan")
    results = state.get("research_results")
    critique = state.get("critique")

    if not plan or not results:
        state["error"] = "Missing research plan or results"
        state["current_node"] = "end"
        return state

    # Initialize LLM
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0.7,  # Higher temperature for more creative synthesis
    )

    # Trace LLM call
    with trace_llm_call("writer", "synthesize_report") as span:
        try:
            # Prepare full research content for synthesis
            research_content = "\n\n---\n\n".join(
                [
                    f"## Source {i+1}: {r.title}\nURL: {r.url}\n\n{r.content}"
                    for i, r in enumerate(results.results)
                ]
            )

            system_prompt = """You are an expert research synthesizer. Your task is to create a comprehensive, well-structured research report from multiple sources.

The report should:
1. Be clear and well-organized with proper headings
2. Synthesize information from all sources (don't just list them)
3. Cite sources using their URLs
4. Highlight key findings and insights
5. Be objective and balanced
6. Include a confidence assessment based on source quality and coverage

Format the report in Markdown."""

            user_prompt = f"""Synthesize a comprehensive research report from these sources:

**Research Query:** {plan.query}

**Sub-queries Investigated:**
{chr(10).join(f"- {sq}" for sq in plan.sub_queries)}

**Research Sources:**
{research_content}

Create a well-structured report that synthesizes this information. Include:
- Executive summary
- Key findings
- Detailed analysis
- Source citations
- Confidence assessment (0.0 to 1.0)

Format as Markdown."""

            span.set_input({
                "query": plan.query,
                "source_count": results.total_count,
                "quality_score": critique.quality_score if critique else None,
            })

            # Use structured output
            try:
                structured_llm = llm.with_structured_output(FinalReport)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                report_result = await structured_llm.ainvoke(messages)
            except Exception:
                # Fallback: Generate content and extract
                response = await llm.ainvoke(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )

                # Extract sources from content
                sources = list(set(result.url for result in results.results))

                # Try to extract confidence from response if mentioned
                import re

                confidence_match = re.search(
                    r'confidence[:\s]+([0-9.]+)', response.content, re.IGNORECASE
                )
                confidence = (
                    float(confidence_match.group(1))
                    if confidence_match
                    else (critique.quality_score if critique else 0.8)
                )

                report_result = FinalReport(
                    content=response.content,
                    sources=sources,
                    confidence=min(max(confidence, 0.0), 1.0),  # Clamp to [0, 1]
                )

            span.set_output({
                "report_length": len(report_result.content),
                "source_count": len(report_result.sources),
                "confidence": report_result.confidence,
            })

            # Save to database (Supabase integration)
            # Note: Reports are always saved if Supabase is configured (not gated by ENABLE_CACHING)
            if settings.supabase_url and settings.supabase_key:
                try:
                    from db.repository import _get_report_repo

                    report_repo = _get_report_repo()
                    report_id = await report_repo.save_report(
                        query=plan.query,
                        report=report_result,
                        quality_score=critique.quality_score if critique else None,
                        iteration_count=state.get("iteration_count", 0),
                        metadata={
                            "sub_queries": plan.sub_queries,
                            "search_terms": plan.search_terms,
                            "total_sources": len(report_result.sources),
                        },
                    )
                    # Update output with report_id
                    existing_output = span.output_data or {}
                    span.set_output({
                        **existing_output,
                        "report_id": report_id,
                    })
                    print(f"✅ Report saved to Supabase (ID: {report_id})")
                except Exception as e:
                    # Database save failure is logged but doesn't block execution
                    import traceback
                    print(f"❌ Failed to save report to database: {e}")
                    print(f"   Error details: {traceback.format_exc()}")
            else:
                print("ℹ️  Supabase not configured - report not saved to database")

            state["final_report"] = report_result
            state["current_node"] = "end"
            return state

        except Exception as e:
            span.set_error(e)
            state["error"] = f"Writer failed: {str(e)}"
            state["current_node"] = "end"
            return state
