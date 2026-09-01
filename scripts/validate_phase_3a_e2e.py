"""Constrained inspection for Phase 3A decision framing."""

import asyncio
import json
import sys

from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state, finalize_from_state


DEFAULT_QUERY = "Should we use OpenAI or Anthropic for our enterprise support agent?"


async def inspect_decision_frame(query: str) -> dict:
    graph = create_graph()
    app = graph.compile()
    initial_state, ctx = await create_initial_state(query)
    config = create_run_config()

    final_state = await app.ainvoke(initial_state, config=config)
    await finalize_from_state(final_state, ctx)

    route = (final_state.get("query_classification") or {}).get("route")
    frame = final_state.get("decision_frame")
    metrics = final_state.get("decision_frame_metrics") or {}

    return {
        "query": query,
        "route": route,
        "decision_frame": frame,
        "decision_frame_metrics": metrics,
        "fast_path_note": (
            "SIMPLE_FACT fast path skips decision framer"
            if route == "simple_fact" and frame is None
            else None
        ),
    }


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    result = asyncio.run(inspect_decision_frame(query))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
