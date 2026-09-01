"""Phase 2B.6 E2E — fast path comparison."""

import asyncio
import json
import sys
import time

from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state, finalize_from_state


QUERIES = [
    "What is the capital of Japan?",
    "Who won the 2023 F1 World Championship?",
]


async def run_query(query: str) -> dict:
    start = time.monotonic()
    graph = create_graph()
    app = graph.compile()
    initial_state, ctx = await create_initial_state(query)
    config = create_run_config()

    final_state = await app.ainvoke(initial_state, config=config)
    await finalize_from_state(final_state, ctx)
    elapsed_ms = round((time.monotonic() - start) * 1000, 2)

    classification = final_state.get("query_classification") or {}
    claim_metrics = final_state.get("claim_metrics") or {}
    cost_metrics = final_state.get("cost_metrics") or {}
    fast_metrics = final_state.get("fast_path_metrics") or {}
    evidence_metrics = final_state.get("evidence_metrics") or {}
    report = final_state.get("final_report")

    return {
        "query": query,
        "route": classification.get("route"),
        "elapsed_ms": elapsed_ms,
        "fast_path": fast_metrics.get("fast_path_entered", False),
        "escalated": final_state.get("escalated_from_fast_path", False),
        "evidence_count": evidence_metrics.get("validated_count", 0),
        "core_claims": claim_metrics.get("unique_claims_persisted", 0),
        "llm_validation_calls": claim_metrics.get("llm_validation_calls", 0),
        "claim_extractor_skipped": claim_metrics.get("full_claim_extractor_skipped", False),
        "fast_llm_calls": fast_metrics.get("llm_calls", 0),
        "sources_processed": fast_metrics.get("sources_processed", 0),
        "stop_reason": fast_metrics.get("stop_reason", ""),
        "answer": report.content[:200] if report else None,
        "answer_confidence": report.answer_confidence_level if report else None,
    }


async def main():
    queries = QUERIES if len(sys.argv) <= 1 else sys.argv[1:]
    results = []
    for q in queries:
        print(f"\n{'='*60}\n{q}\n{'='*60}")
        try:
            r = await run_query(q)
            results.append(r)
            print(f"  Route: {r['route']}")
            print(f"  Elapsed: {r['elapsed_ms']}ms")
            print(f"  Fast path: {r['fast_path']}")
            print(f"  Evidence: {r['evidence_count']}")
            print(f"  Core claims: {r['core_claims']}")
            print(f"  LLM validation calls: {r['llm_validation_calls']}")
            print(f"  Answer: {r['answer']}")
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append({"query": q, "exception": str(e)})

    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
