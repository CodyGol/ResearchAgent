"""Phase 2B E2E validation script — claim extraction inspection."""

import asyncio
import json
import sys

from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state, finalize_from_state


QUERIES = [
    "Who won the 2023 F1 World Championship?",
    "What was Apple's revenue in fiscal 2025?",
    "What is the capital of Japan?",
    "Who is the current President of the United States as of 2025?",
    "What was US unemployment as of December 2024?",
]


async def run_query(query: str) -> dict:
    graph = create_graph()
    app = graph.compile()
    initial_state, ctx = await create_initial_state(query)
    config = create_run_config()

    final_state = await app.ainvoke(initial_state, config=config)
    await finalize_from_state(final_state, ctx)

    claims = final_state.get("validated_claims") or []
    claim_metrics = final_state.get("claim_metrics") or {}
    report = final_state.get("final_report")

    claim_samples = [
        {
            "text": c.text,
            "type": c.claim_type.value,
            "importance": c.metadata.get("importance"),
            "support_basis": c.metadata.get("support_basis"),
            "temporal_scope": c.temporal_scope,
            "qualifiers": c.qualifiers,
        }
        for c in claims[:10]
    ]

    return {
        "query": query,
        "error": final_state.get("error"),
        "evidence_count": (final_state.get("evidence_metrics") or {}).get("validated_count", 0),
        "claim_metrics": claim_metrics,
        "claims_sample": claim_samples,
        "claims_total": len(claims),
        "report_length": len(report.content) if report else 0,
        "confidence_level": report.confidence_level if report else None,
        "has_report": report is not None,
    }


async def main():
    queries = QUERIES if len(sys.argv) <= 1 else sys.argv[1:]
    results = []
    for q in queries:
        print(f"\n{'='*60}\nRunning: {q}\n{'='*60}")
        try:
            r = await run_query(q)
            results.append(r)
            cm = r.get("claim_metrics") or {}
            print(f"  Evidence: {r['evidence_count']}")
            print(f"  Claims accepted: {cm.get('claims_accepted', 0)}")
            print(f"  Unique claims: {cm.get('unique_claims_persisted', 0)}")
            print(f"  Rejected: {cm.get('claims_rejected', 0)}")
            print(f"  Merged: {cm.get('duplicate_claims_merged', 0)}")
            if r.get("error"):
                print(f"  ERROR: {r['error']}")
            for i, c in enumerate(r.get("claims_sample", [])[:3]):
                print(f"  Claim {i+1}: {c['text'][:100]}")
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append({"query": q, "exception": str(e)})

    print("\n\n=== FULL RESULTS ===")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
