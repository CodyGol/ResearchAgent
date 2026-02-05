"""Script to run The Oracle research agent with a custom query."""

import asyncio
import sys

from graph import create_graph
from state import AgentState


async def run_research(query: str):
    """
    Run The Oracle research agent with a specific query.
    
    Args:
        query: The research question to investigate
    """
    graph = create_graph()
    app = graph.compile()

    # Initialize state
    initial_state: AgentState = {
        "user_query": query,
        "research_plan": None,
        "research_results": None,
        "critique": None,
        "final_report": None,
        "current_node": "planner",
        "iteration_count": 0,
        "error": None,
    }

    print("🚀 Starting The Oracle...")
    print(f"Query: {query}\n")
    print("=" * 80)

    # Execute graph
    final_state = await app.ainvoke(initial_state)

    # Display results
    print("\n" + "=" * 80)
    if final_state.get("error"):
        print(f"❌ Error: {final_state['error']}")
        return

    report = final_state.get("final_report")
    if report:
        print("✅ Research Complete!\n")
        print(f"📊 Report:\n{report.content}\n")
        print(f"📚 Sources ({len(report.sources)}):")
        for i, source in enumerate(report.sources, 1):
            print(f"  {i}. {source}")
        
        # Show metadata if available
        if hasattr(report, 'confidence'):
            print(f"\n🎯 Confidence Score: {report.confidence:.2f}")
        
        # Show iteration count
        iteration_count = final_state.get("iteration_count", 0)
        if iteration_count > 0:
            print(f"🔄 Research Iterations: {iteration_count + 1}")
    else:
        print("⚠️  No report generated")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Query from command line arguments
        query = " ".join(sys.argv[1:])
    else:
        # Default query
        query = "What is the AI Application Layer? What are its key components, architecture patterns, and how is it being implemented in modern systems?"

    asyncio.run(run_research(query))


if __name__ == "__main__":
    main()
