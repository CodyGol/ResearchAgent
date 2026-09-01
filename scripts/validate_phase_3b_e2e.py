"""Constrained inspection for Phase 3B option evaluation."""

import asyncio
import json
import sys

from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state, finalize_from_state

DEFAULT_QUERY = (
    "Which LLM provider should we choose between OpenAI and Anthropic "
    "for an enterprise support agent, considering cost and security?"
)


async def inspect_option_evaluation(query: str) -> dict:
    graph = create_graph()
    app = graph.compile()
    initial_state, ctx = await create_initial_state(query)
    config = create_run_config()

    final_state = await app.ainvoke(initial_state, config=config)
    await finalize_from_state(final_state, ctx)

    frame = final_state.get("decision_frame")
    oe = final_state.get("option_evaluation")
    metrics = final_state.get("option_evaluation_metrics") or {}

    report = {
        "query": query,
        "decision_frame_present": frame is not None,
        "option_count": len((frame or {}).get("options") or []),
        "criterion_count": len((frame or {}).get("criteria") or []),
        "option_evaluation_present": oe is not None,
        "option_evaluation_metrics": metrics,
    }

    if oe:
        report["evaluation_summary"] = {
            "decision": oe.get("decision"),
            "options_evaluated": len(oe.get("option_evaluations") or []),
            "sample_assessments": [
                {
                    "option": opt.get("option_label"),
                    "criterion": ce.get("criterion_label"),
                    "assessment": ce.get("assessment"),
                    "coverage": ce.get("knowledge_coverage"),
                    "claim_count": len(ce.get("claim_ids") or []),
                }
                for opt in (oe.get("option_evaluations") or [])[:2]
                for ce in (opt.get("criteria_evaluations") or [])[:2]
            ],
        }

    return report


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    result = asyncio.run(inspect_option_evaluation(query))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
