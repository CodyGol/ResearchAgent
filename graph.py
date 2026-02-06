"""LangGraph StateGraph entry point for The Oracle research agent."""

import os
import warnings

# Suppress Pydantic and LangChain warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_core")
warnings.filterwarnings("ignore", message=".*Pydantic.*", category=UserWarning)

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START

from config import settings
from nodes.critic import critic_node
from nodes.planner import planner_node
from nodes.researcher import researcher_node
from nodes.writer import writer_node
from state import AgentState


def create_graph() -> StateGraph:
    """
    Create and configure the LangGraph StateGraph for The Oracle.

    Graph structure:
        START -> planner -> researcher -> critic -> (writer | researcher) -> END

    The critic node has a conditional edge that loops back to researcher
    if quality is insufficient (recursive refinement).

    Returns:
        Configured StateGraph instance
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", writer_node)

    # Define edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "critic")

    # Conditional edge from critic: loop back to researcher or proceed to writer
    def should_continue(state: AgentState) -> str:
        """Route based on critique result."""
        critique = state.get("critique")
        if not critique:
            return "writer"  # Fallback if critique missing

        if critique.is_sufficient:
            return "writer"
        else:
            # Check iteration limit
            iteration = state.get("iteration_count", 0)
            if iteration >= settings.max_research_iterations:
                return "writer"  # Force proceed
            return "researcher"  # Loop back

    graph.add_conditional_edges(
        "critic",
        should_continue,
        {
            "researcher": "researcher",
            "writer": "writer",
        },
    )

    graph.add_edge("writer", END)

    return graph


def get_langsmith_trace_url() -> str | None:
    """
    Generate LangSmith trace URL for the current run.
    
    Returns:
        URL string if LangSmith is configured, None otherwise
    """
    # Check if LangSmith tracing is enabled
    if not os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        return None
    
    try:
        from langsmith import Client
        
        client = Client()
        project_name = settings.langchain_project
        
        # Get organization ID from API key or environment
        org_id = os.getenv("LANGCHAIN_ORG_ID") or "default"
        
        # Construct trace URL (LangSmith project URL format)
        base_url = os.getenv("LANGCHAIN_ENDPOINT", "https://smith.langchain.com")
        trace_url = f"{base_url}/o/{org_id}/projects/p/{project_name}"
        
        return trace_url
    except Exception:
        # Fallback if LangSmith client not available
        project_name = settings.langchain_project
        return f"https://smith.langchain.com/o/<org-id>/projects/p/{project_name}"


def create_run_config() -> RunnableConfig:
    """
    Create RunnableConfig with LangSmith metadata and tags.
    
    Returns:
        Configured RunnableConfig for tracing
    """
    return RunnableConfig(
        metadata={
            "env": settings.environment,
            "project": settings.langchain_project,
        },
        tags=["oracle-v1", "research-agent", f"env-{settings.environment}"],
    )


async def main():
    """Entry point for testing the graph."""
    graph = create_graph()
    app = graph.compile()

    # Configure LangSmith tracing
    run_config = create_run_config()
    
    # Print trace URL at start (Rule 3.C: Observability)
    trace_url = get_langsmith_trace_url()
    if trace_url:
        print(f"🛠️  View Trace: {trace_url}\n")

    # Test with mock query
    initial_state: AgentState = {
        "user_query": "What are the latest developments in AI safety?",
        "research_plan": None,
        "research_results": None,
        "critique": None,
        "final_report": None,
        "current_node": "planner",
        "iteration_count": 0,
        "error": None,
    }

    print("🚀 Starting The Oracle...")
    print(f"Query: {initial_state['user_query']}\n")

    # Execute graph with LangSmith config (use ainvoke for async nodes)
    final_state = await app.ainvoke(initial_state, config=run_config)

    # Display results
    if final_state.get("error"):
        print(f"❌ Error: {final_state['error']}")
    else:
        report = final_state.get("final_report")
        if report:
            print("✅ Research Complete!")
            print(f"\n📊 Report:\n{report.content}\n")
            print(f"📚 Sources ({len(report.sources)}):")
            for source in report.sources:
                print(f"  - {source}")
        else:
            print("⚠️  No report generated")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
