"""Real-world evidence extraction validation (Phase 2A)."""

import asyncio
import json
import sys

from services.evidence_pipeline import process_sources_for_evidence
from services.research_run_service import start_research_run
from services.source_normalizer import normalize_search_results
from tools.search import search_tavily_with_retry


QUERIES = [
    "Who won the 2023 F1 World Championship?",
    "What is the capital of Japan?",
    "When was Python programming language first released?",
]


async def validate_query(query: str) -> dict:
    ctx = await start_research_run(query)
    run_id = ctx.run.id

    results = await search_tavily_with_retry(query, max_results=3)
    sources = normalize_search_results(results, run_id or 1)

    if ctx.is_persisted and run_id:
        from services.research_run_service import persist_sources
        sources = await persist_sources(ctx, sources)

    evidence, metrics = await process_sources_for_evidence(
        sources=sources,
        research_question=query,
        research_run_id=run_id or 1,
        is_persisted=ctx.is_persisted,
    )

    return {
        "query": query,
        "run_id": run_id,
        "sources": len(sources),
        "evidence": [
            {
                "text": e.exact_text[:120],
                "match_type": e.match_type.value if e.match_type else None,
                "source_url": e.metadata.get("source_url"),
                "validated": e.is_validated,
            }
            for e in evidence
        ],
        "metrics": metrics.to_dict(),
    }


async def main():
    results = []
    for query in QUERIES:
        print(f"\n--- Query: {query} ---")
        try:
            result = await validate_query(query)
            results.append(result)
            print(f"  Sources: {result['sources']}")
            print(f"  Validated evidence: {result['metrics']['validated_count']}")
            print(f"  Rejected: {result['metrics']['rejected_count']}")
            for ev in result["evidence"][:3]:
                print(f"    [{ev['match_type']}] {ev['text']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"query": query, "error": str(e)})

    print("\n=== SUMMARY ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
