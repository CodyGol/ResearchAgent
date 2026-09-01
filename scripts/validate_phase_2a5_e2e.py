"""Phase 2A.5 E2E validation script."""

import asyncio
import json
import sys

from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state, finalize_from_state


QUERIES = [
    "Who won the 2023 F1 World Championship?",
    "What is the capital of Japan?",
    "What was Apple's revenue in fiscal 2025?",
    "Who is the current President of the United States as of 2025?",
]


async def run_query(query: str) -> dict:
    graph = create_graph()
    app = graph.compile()
    initial_state, ctx = await create_initial_state(query)
    config = create_run_config()

    final_state = await app.ainvoke(initial_state, config=config)
    await finalize_from_state(final_state, ctx)

    report = final_state.get("final_report")
    return {
        "query": query,
        "error": final_state.get("error"),
        "source_dedup": final_state.get("source_dedup_metrics"),
        "evidence_metrics": final_state.get("evidence_metrics"),
        "report_metrics": final_state.get("report_metrics"),
        "critique": final_state.get("critique").model_dump() if final_state.get("critique") else None,
        "confidence_level": report.confidence_level if report else None,
        "confidence": report.confidence if report else None,
        "confidence_reasoning": report.confidence_reasoning if report else None,
        "sources_cited": report.sources if report else [],
        "evidence_ids_used": report.evidence_ids_used if report else [],
        "report_excerpt": (report.content[:800] + "...") if report and len(report.content) > 800 else (report.content if report else None),
        "report_length": len(report.content) if report else 0,
        "has_23_races": "23 race" in (report.content.lower() if report else ""),
        "has_22_races": "22 race" in (report.content.lower() if report else ""),
    }


async def main():
    queries = QUERIES if len(sys.argv) <= 1 else sys.argv[1:]
    results = []
    for q in queries:
        print(f"\n{'='*60}\nRunning: {q}\n{'='*60}")
        try:
            r = await run_query(q)
            results.append(r)
            print(f"  Evidence: {r['evidence_metrics'].get('validated_count', 0) if r['evidence_metrics'] else 0}")
            print(f"  Confidence: {r['confidence_level']} ({r['confidence']})")
            print(f"  Sources cited: {len(r['sources_cited'])}")
            print(f"  Report length: {r['report_length']} chars")
            if r.get("error"):
                print(f"  ERROR: {r['error']}")
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append({"query": q, "exception": str(e)})

    print("\n\n=== FULL RESULTS ===")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
