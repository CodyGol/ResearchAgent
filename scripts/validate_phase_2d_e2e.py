"""Constrained inspection for Phase 2D knowledge state."""

import asyncio
import json
import sys

from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state, finalize_from_state


DEFAULT_QUERY = "Who won the 2023 Formula 1 World Championship?"


async def inspect_knowledge_state(query: str) -> dict:
    graph = create_graph()
    app = graph.compile()
    initial_state, ctx = await create_initial_state(query)
    config = create_run_config()

    final_state = await app.ainvoke(initial_state, config=config)
    await finalize_from_state(final_state, ctx)

    ks = final_state.get("knowledge_state")
    route = (final_state.get("query_classification") or {}).get("route")

    report = {
        "query": query,
        "route": route,
        "knowledge_state_present": ks is not None,
        "fast_path_limitation": (
            "Fast-path runs bypass claim verification and do not receive Phase 2D knowledge state."
            if route == "simple_fact" and ks is None
            else None
        ),
    }

    if ks:
        report["metrics"] = ks.get("metrics", {})
        report["buckets"] = {
            "known": [e["claim_id"] for e in ks.get("known", [])],
            "likely": [e["claim_id"] for e in ks.get("likely", [])],
            "disputed": [e["claim_id"] for e in ks.get("disputed", [])],
            "unknown": [e["claim_id"] for e in ks.get("unknown", [])],
            "contradicted": [e["claim_id"] for e in ks.get("contradicted", [])],
            "unverifiable": [e["claim_id"] for e in ks.get("unverifiable", [])],
        }
        report["information_gaps"] = ks.get("information_gaps", [])

    return report


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    result = asyncio.run(inspect_knowledge_state(query))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
