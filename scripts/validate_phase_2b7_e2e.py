"""Phase 2B.7 E2E — generalized fast fact engine validation."""

import asyncio
import json
import sys
import time

from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state, finalize_from_state

DEFAULT_QUERIES = [
    "What is the capital of Japan?",
    "Who won the 2023 F1 World Championship?",
    "What was Apple's revenue in fiscal 2025?",
    "Who is the CEO of Apple?",
    "When was Python first released?",
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
    fast_metrics = final_state.get("fast_path_metrics") or {}
    evidence_metrics = final_state.get("evidence_metrics") or {}
    report = final_state.get("final_report")

    return {
        "query": query,
        "route": classification.get("route"),
        "elapsed_ms": elapsed_ms,
        "fast_path": fast_metrics.get("fast_path_entered", False),
        "escalated": final_state.get("escalated_from_fast_path", False),
        "escalation_reason": final_state.get("escalation_reason", ""),
        "evidence_count": evidence_metrics.get("validated_count", 0),
        "core_claims": claim_metrics.get("unique_claims_persisted", 0),
        "fact_values_extracted": fast_metrics.get("fact_values_extracted", 0),
        "llm_calls": fast_metrics.get("llm_calls", 0),
        "llm_calls_evidence": fast_metrics.get("llm_calls_evidence", 0),
        "llm_calls_core_claim": fast_metrics.get("llm_calls_core_claim", 0),
        "sources_processed": fast_metrics.get("sources_processed", 0),
        "stop_reason": fast_metrics.get("stop_reason", ""),
        "cache_key": fast_metrics.get("cache_key", ""),
        "freshness": fast_metrics.get("freshness", ""),
        "answer": report.content[:300] if report else None,
        "answer_confidence": report.answer_confidence_level if report else None,
        "research_completeness": report.research_completeness_level if report else None,
        "fact_value": (report.report_metrics or {}).get("fact_value") if report else None,
    }


async def main():
    queries = DEFAULT_QUERIES if len(sys.argv) <= 1 else sys.argv[1:]
    results = []
    for q in queries:
        print(f"\n{'='*60}\n{q}\n{'='*60}")
        try:
            r = await run_query(q)
            results.append(r)
            print(f"  Route: {r['route']}")
            print(f"  Elapsed: {r['elapsed_ms']}ms")
            print(f"  Fast path: {r['fast_path']}, Escalated: {r['escalated']}")
            if r["escalated"]:
                print(f"  Escalation: {r['escalation_reason']}")
            print(f"  Evidence: {r['evidence_count']}, Fact values: {r['fact_values_extracted']}")
            print(f"  LLM calls: {r['llm_calls']} (evidence={r['llm_calls_evidence']}, claim={r['llm_calls_core_claim']})")
            print(f"  Confidence: {r['answer_confidence']} / completeness: {r['research_completeness']}")
            print(f"  Answer: {r['answer']}")
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append({"query": q, "exception": str(e)})

    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
