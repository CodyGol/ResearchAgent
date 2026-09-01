"""Constrained live inspection for Phase 2C claim verification."""

import asyncio
import json
import sys

from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state, finalize_from_state


DEFAULT_QUERY = "Who won the 2023 Formula 1 World Championship?"


async def inspect_verification(query: str) -> dict:
    graph = create_graph()
    app = graph.compile()
    initial_state, ctx = await create_initial_state(query)
    config = create_run_config()

    final_state = await app.ainvoke(initial_state, config=config)
    await finalize_from_state(final_state, ctx)

    material = final_state.get("material_claims") or []
    relations = final_state.get("claim_evidence_relations") or []
    verifications = final_state.get("verification_results") or []
    evidence = final_state.get("validated_evidence") or []
    sources = final_state.get("normalized_sources") or []
    metrics = final_state.get("verification_metrics") or {}

    source_by_id = {s.id: s for s in sources if s.id}
    evidence_by_id = {e.id: e for e in evidence if e.id}

    claims_report = []
    for claim in material[:5]:
        cid = claim.id
        claim_rels = [r for r in relations if r.claim_id == cid]
        vr = next((v for v in verifications if v.claim_id == cid), None)
        rel_details = []
        for rel in claim_rels:
            ev = evidence_by_id.get(rel.evidence_id)
            src = source_by_id.get(ev.source_id) if ev else None
            rel_details.append({
                "relationship": rel.relationship.value,
                "evidence_id": rel.evidence_id,
                "source_domain": (src.metadata or {}).get("domain") if src else None,
                "source_url": src.url if src else None,
                "evidence_excerpt": ev.exact_text[:120] if ev else None,
                "reasoning": rel.reasoning,
            })
        claims_report.append({
            "claim_id": cid,
            "claim_text": claim.text,
            "verification_status": vr.status.value if vr else None,
            "confidence": vr.confidence.value if vr else None,
            "verification_reasoning": vr.reasoning if vr else None,
            "relations": rel_details,
        })

    return {
        "query": query,
        "research_run_id": final_state.get("research_run_id"),
        "material_claims_count": len(material),
        "verification_metrics": metrics,
        "claims": claims_report,
    }


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    result = await inspect_verification(query)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
