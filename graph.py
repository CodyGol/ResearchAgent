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
from nodes.router import router_node
from nodes.fast_path import fast_path_node
from nodes.claim_extractor import claim_extractor_node
from nodes.claim_verifier import claim_verifier_node
from nodes.critic import critic_node
from nodes.evidence_extractor import evidence_extractor_node
from nodes.planner import planner_node
from nodes.researcher import researcher_node
from nodes.decision_framer import decision_framer_node
from nodes.knowledge_state import knowledge_state_node
from nodes.writer import writer_node
from services.pipeline_init import create_initial_state
from state import AgentState


def create_graph() -> StateGraph:
    """
    Create and configure the LangGraph StateGraph for The Oracle.

    Graph structure:
        START -> router -> [fast_path | decision_framer]
        fast_path -> [END | decision_framer]  (escalation)
        decision_framer -> planner -> researcher -> ... -> knowledge_state -> writer -> END
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("fast_path", fast_path_node)
    graph.add_node("decision_framer", decision_framer_node)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("evidence_extractor", evidence_extractor_node)
    graph.add_node("claim_extractor", claim_extractor_node)
    graph.add_node("claim_verifier", claim_verifier_node)
    graph.add_node("critic", critic_node)
    graph.add_node("knowledge_state", knowledge_state_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "router")

    def route_after_router(state: AgentState) -> str:
        classification = state.get("query_classification") or {}
        route = classification.get("route", "standard")
        if route == "simple_fact" and not state.get("escalated_from_fast_path"):
            return "fast_path"
        return "decision_framer"

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"fast_path": "fast_path", "decision_framer": "decision_framer"},
    )

    def route_after_fast_path(state: AgentState) -> str:
        if state.get("escalate_to_standard"):
            return "decision_framer"
        return "end"

    graph.add_conditional_edges(
        "fast_path",
        route_after_fast_path,
        {"decision_framer": "decision_framer", "end": END},
    )

    graph.add_edge("decision_framer", "planner")

    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "evidence_extractor")
    graph.add_edge("evidence_extractor", "claim_extractor")
    graph.add_edge("claim_extractor", "claim_verifier")
    graph.add_edge("claim_verifier", "critic")

    def route_after_critic(state: AgentState) -> str:
        critique = state.get("critique")
        if not critique:
            return "knowledge_state"

        if critique.is_sufficient:
            return "knowledge_state"

        iteration = state.get("iteration_count", 0)
        classification = state.get("query_classification") or {}
        budget = classification.get("research_budget", {})
        max_iter = budget.get("max_iterations", settings.max_research_iterations)
        if iteration >= max_iter:
            return "knowledge_state"
        return "researcher"

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"researcher": "researcher", "knowledge_state": "knowledge_state"},
    )

    graph.add_edge("knowledge_state", "writer")
    graph.add_edge("writer", END)

    return graph


def get_langsmith_trace_url() -> str | None:
    if not os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        return None
    try:
        from langsmith import Client
        client = Client()
        project_name = settings.langchain_project
        org_id = os.getenv("LANGCHAIN_ORG_ID") or "default"
        base_url = os.getenv("LANGCHAIN_ENDPOINT", "https://smith.langchain.com")
        return f"{base_url}/o/{org_id}/projects/p/{project_name}"
    except Exception:
        return f"https://smith.langchain.com/o/<org-id>/projects/p/{settings.langchain_project}"


def create_run_config() -> RunnableConfig:
    return RunnableConfig(
        metadata={
            "env": settings.environment,
            "project": settings.langchain_project,
        },
        tags=["oracle-v1", "research-agent", f"env-{settings.environment}"],
    )


async def main():
    graph = create_graph()
    app = graph.compile()
    run_config = create_run_config()
    trace_url = get_langsmith_trace_url()
    if trace_url:
        print(f"🛠️  View Trace: {trace_url}\n")

    query = "What are the latest developments in AI safety?"
    initial_state, _ctx = await create_initial_state(query)

    print("🚀 Starting The Oracle...")
    print(f"Query: {initial_state['user_query']}\n")

    final_state = await app.ainvoke(initial_state, config=run_config)

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
