"""Planner node: Analyzes query and generates structured research plan."""

from langchain_anthropic import ChatAnthropic

from config import settings
from state import AgentState, ResearchPlan
from utils.observability import trace_llm_call


def _create_fallback_plan(query: str) -> ResearchPlan:
    """Fallback plan generator if structured output fails."""
    return ResearchPlan(
        query=query,
        sub_queries=[
            f"What is {query}?",
            f"Recent developments in {query}",
            f"Expert analysis of {query}",
        ],
        search_terms=query.split(),
    )


async def planner_node(state: AgentState) -> AgentState:
    """
    Planner node: Decomposes user query into structured research plan.

    Uses Claude 4.5 Sonnet with structured output (Pydantic V2) to generate
    a research plan with sub-queries and search terms.
    Checks Supabase cache first to avoid redundant LLM calls.

    Args:
        state: Current agent state

    Returns:
        Updated state with research_plan populated
    """
    query = state["user_query"]

    # Check cache first (Supabase integration)
    try:
        from db.repository import _get_plan_repo

        plan_repo = _get_plan_repo()
        cached_plan = await plan_repo.get_cached_plan(query)
        if cached_plan:
            state["research_plan"] = cached_plan
            state["current_node"] = "researcher"
            return state
    except Exception as e:
        # Cache miss or error - continue with generation
        print(f"Cache check failed (non-critical): {e}")

    # Initialize LLM with structured output (Rule 3.A: Pydantic Everywhere)
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0.3,  # Lower temperature for more deterministic planning
    )

    # Trace LLM call (Rule 3.C: Trace Everything)
    with trace_llm_call("planner", "generate_research_plan") as span:
        try:
            # Detect query type for domain prioritization
            technical_keywords = [
                "architecture", "api", "application layer", "system design", "implementation",
                "code", "framework", "library", "sdk", "protocol", "algorithm", "technical",
                "developer", "engineering", "infrastructure", "component", "pattern", "stack",
                "performance", "benchmark", "latency", "throughput", "scalability"
            ]
            academic_keywords = [
                "academic", "research", "paper", "study", "scholar", "university", "journal",
                "peer-reviewed", "publication", "thesis", "dissertation", "conference"
            ]
            
            is_technical = any(keyword in query.lower() for keyword in technical_keywords)
            is_academic = any(keyword in query.lower() for keyword in academic_keywords)
            
            # System prompt defines the Persona (Rule 3.A: No "Chat")
            # You are a SENIOR TECHNICAL RESEARCHER, not a content writer
            system_prompt = """You are a Senior Technical Researcher with a deep aversion to SEO blogs and marketing content. 
You prioritize PRIMARY SOURCES: official documentation, academic papers, benchmarks, and technical specifications.

Your mission:
1. Generate SPECIFIC sub-queries that target PRIMARY SOURCES:
   - Technical queries: "AI Application Layer architecture patterns"
   - Performance queries: "LLM inference latency benchmarks"
   - Academic queries: "neural architecture search algorithms"
   - Be precise: "distributed system consistency protocols" not "distributed systems"

2. Extract enhanced search terms (for reference/documentation):
   - Include site: and filetype: modifiers to show intent: ["AI architecture site:arxiv.org filetype:pdf", "system design patterns site:github.com"]
   - These help document the research strategy even though the search API uses domain filters

3. Domain prioritization (CRITICAL):
   - Technical queries: Set "domains" to ["arxiv.org", "github.com", "docs.aws.amazon.com", "stackoverflow.com", "developer.mozilla.org"]
   - Academic queries: Set "required_domains" to ["arxiv.org", "scholar.google.com"]
   - Always prefer: arxiv.org, github.com, official docs, .edu domains
   - NEVER include: medium.com, linkedin.com, pinterest.com (these are blacklisted)

4. Query specificity:
   - Include technical terms: "latency", "throughput", "scalability", "consistency"
   - Include file types when relevant: "PDF", "whitepaper", "technical specification"
   - Include measurement terms: "benchmarks", "performance metrics", "evaluation results"

You DESPISE SEO blogs. You CRAVE DATA, BENCHMARKS, and OFFICIAL DOCUMENTATION.
The "domains" and "required_domains" fields are your PRIMARY weapons against content spam."""

            user_prompt = f"""Analyze this research query and create a structured plan targeting PRIMARY SOURCES:

Query: {query}

Generate:
1. Sub-queries with site: and filetype: filters (e.g., "AI architecture site:arxiv.org filetype:pdf")
2. Enhanced search terms with modifiers (not just keywords)
3. Domain filters for technical queries
4. Required domains for academic queries (if applicable)

Remember: You are a Senior Technical Researcher. Prioritize official docs, academic papers, and benchmarks over blogs."""

            span.set_input({
                "query": query,
                "model": settings.model_name,
            })

            # Use structured output with Pydantic model (Rule 3.A: JSON is King)
            try:
                # Attempt structured output (LangChain supports this with Pydantic V2)
                structured_llm = llm.with_structured_output(ResearchPlan)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                plan_result = await structured_llm.ainvoke(messages)
            except Exception as e:
                # Fallback: Use regular invoke and parse JSON from response
                response = await llm.ainvoke(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                )
                response_content = response.content

                # Try to extract JSON from response
                import json
                import re

                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    try:
                        plan_data = json.loads(json_match.group())
                        plan_result = ResearchPlan(**plan_data)
                    except Exception:
                        # JSON parse failed, use fallback
                        plan_result = _create_fallback_plan(query)
                else:
                    # No JSON found, use fallback
                    plan_result = _create_fallback_plan(query)

            span.set_output({
                "plan": plan_result.model_dump(),
            })

            # Save to cache (Supabase integration)
            try:
                from db.repository import _get_plan_repo

                plan_repo = _get_plan_repo()
                await plan_repo.save_plan(query, plan_result)
            except Exception as e:
                # Cache save failure is non-critical
                print(f"Cache save failed (non-critical): {e}")

            state["research_plan"] = plan_result
            state["current_node"] = "researcher"
            return state

        except Exception as e:
            span.set_error(e)
            state["error"] = f"Planner failed: {str(e)}"
            state["current_node"] = "end"
            return state
