"""Phase 2B.5 E2E comparison script — adaptive research metrics."""

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
    evidence_metrics = final_state.get("evidence_metrics") or {}
    report = final_state.get("final_report")

    return {
        "query": query,
        "complexity": classification.get("complexity"),
        "elapsed_ms": elapsed_ms,
        "evidence_count": evidence_metrics.get("validated_count", 0),
        "candidate_claims": claim_metrics.get("candidate_claims_generated", 0),
        "claims_accepted": claim_metrics.get("claims_accepted", 0),
        "material_claims": claim_metrics.get("material_claims_count", 0),
        "unique_claims": claim_metrics.get("unique_claims_persisted", 0),
        "deterministic_rejects": claim_metrics.get("claims_rejected_deterministic", 0),
        "relevance_rejects": claim_metrics.get("claims_rejected_relevance", 0),
        "llm_validation_calls": claim_metrics.get("llm_validation_calls", 0),
        "validation_batches": claim_metrics.get("validation_batches", 0),
        "claim_processing_ms": claim_metrics.get("processing_time_ms", 0),
        "search_queries": cost_metrics.get("search_queries_executed", 0),
        "canonical_sources": cost_metrics.get("canonical_sources", 0),
        "short_circuited": cost_metrics.get("short_circuited", False),
        "answer_confidence": report.answer_confidence_level if report else None,
        "research_completeness": report.research_completeness_level if report else None,
        "report_length": len(report.content) if report else 0,
    }


async def main():
    queries = QUERIES if len(sys.argv) <= 1 else sys.argv[1:]
    results = []
    for q in queries:
        print(f"\n{'='*60}\nRunning: {q}\n{'='*60}")
        try:
            r = await run_query(q)
            results.append(r)
            print(f"  Complexity: {r['complexity']}")
            print(f"  Elapsed: {r['elapsed_ms']}ms")
            print(f"  Evidence: {r['evidence_count']}")
            print(f"  Material claims: {r['material_claims']}")
            print(f"  LLM validation calls: {r['llm_validation_calls']}")
            print(f"  Batches: {r['validation_batches']}")
            print(f"  Answer confidence: {r['answer_confidence']}")
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append({"query": q, "exception": str(e)})

    print("\n\n=== COMPARISON RESULTS ===")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
