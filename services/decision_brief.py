"""Deterministic Decision Brief presentation payload (no LLM)."""

from __future__ import annotations

import logging
from typing import Any

from domain.models import Claim, ClaimEvidenceRelation, Evidence, Source, VerificationResult
from services.decision_synthesis_schemas import RecommendationStatus
from state import AgentState

logger = logging.getLogger(__name__)

_VALID_STATUSES = {s.value for s in RecommendationStatus}
_KS_BUCKETS = ("known", "likely", "disputed", "unknown", "contradicted", "unverifiable")


def _as_dict(obj: Any) -> dict | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return None


def _claim_id(obj: Any) -> int | None:
    if isinstance(obj, Claim):
        return obj.id
    if isinstance(obj, dict):
        cid = obj.get("id")
        return int(cid) if cid is not None else None
    return None


def _claim_text(obj: Any) -> str:
    if isinstance(obj, Claim):
        return obj.text
    if isinstance(obj, dict):
        return str(obj.get("text") or "")
    return ""


def _collect_referenced_claim_ids(
    option_evaluation: dict | None,
    decision_synthesis: dict,
) -> set[int]:
    ids: set[int] = set()

    if option_evaluation:
        for opt in option_evaluation.get("option_evaluations") or []:
            for ce in opt.get("criteria_evaluations") or []:
                for cid in ce.get("claim_ids") or []:
                    if isinstance(cid, int):
                        ids.add(cid)

    for key in ("supporting_criteria", "limiting_criteria"):
        for ref in decision_synthesis.get(key) or []:
            for cid in ref.get("claim_ids") or []:
                if isinstance(cid, int):
                    ids.add(cid)

    for ca in decision_synthesis.get("constraint_assessments") or []:
        for cid in ca.get("claim_ids") or []:
            if isinstance(cid, int):
                ids.add(cid)

    for cc in decision_synthesis.get("change_conditions") or []:
        for cid in cc.get("related_claim_ids") or []:
            if isinstance(cid, int):
                ids.add(cid)

    return ids


def _build_claim_meta(
    knowledge_state: dict | None,
    verification_results: list[Any],
) -> dict[int, dict[str, str | None]]:
    meta: dict[int, dict[str, str | None]] = {}

    if knowledge_state:
        for bucket in _KS_BUCKETS:
            for entry in knowledge_state.get(bucket) or []:
                if not isinstance(entry, dict):
                    continue
                cid = entry.get("claim_id")
                if cid is None:
                    continue
                kc = entry.get("knowledge_category")
                vs = entry.get("verification_status")
                meta[int(cid)] = {
                    "knowledge_category": kc if isinstance(kc, str) else (
                        kc.value if hasattr(kc, "value") else bucket
                    ),
                    "verification_status": vs if isinstance(vs, str) else (
                        vs.value if hasattr(vs, "value") else None
                    ),
                }

    for vr in verification_results:
        if isinstance(vr, VerificationResult):
            cid = vr.claim_id
            if cid is None:
                continue
            if cid not in meta:
                kc = vr.knowledge_category.value if vr.knowledge_category else None
                meta[cid] = {
                    "knowledge_category": kc,
                    "verification_status": vr.status.value,
                }
        elif isinstance(vr, dict) and vr.get("claim_id") is not None:
            cid = int(vr["claim_id"])
            if cid not in meta:
                kc = vr.get("knowledge_category")
                vs = vr.get("status")
                meta[cid] = {
                    "knowledge_category": kc if isinstance(kc, str) else (
                        kc.value if hasattr(kc, "value") else None
                    ),
                    "verification_status": vs if isinstance(vs, str) else (
                        vs.value if hasattr(vs, "value") else None
                    ),
                }

    return meta


def _evidence_key(evidence_id: int | None, display_id: str | None, snippet: str | None) -> str:
    if evidence_id is not None:
        return f"id:{evidence_id}"
    if display_id:
        return f"display:{display_id}"
    return f"snippet:{snippet or ''}"


def _build_claim_lineage(
    referenced_ids: set[int],
    material_claims: list[Any],
    claim_evidence_relations: list[Any],
    validated_evidence: list[Any],
    normalized_sources: list[Any],
    claim_meta: dict[int, dict[str, str | None]],
) -> dict[str, dict[str, Any]]:
    claim_by_id: dict[int, Any] = {}
    for claim in material_claims:
        cid = _claim_id(claim)
        if cid is not None:
            claim_by_id[cid] = claim

    evidence_by_id: dict[int, Evidence] = {}
    for ev in validated_evidence:
        if isinstance(ev, Evidence):
            if ev.id is not None:
                evidence_by_id[ev.id] = ev
        elif isinstance(ev, dict) and ev.get("id") is not None:
            evidence_by_id[int(ev["id"])] = Evidence(**ev)

    source_by_id: dict[int, Source] = {}
    for src in normalized_sources:
        if isinstance(src, Source):
            if src.id is not None:
                source_by_id[src.id] = src
        elif isinstance(src, dict) and src.get("id") is not None:
            source_by_id[int(src["id"])] = Source(**src)

    relations_by_claim: dict[int, list[int]] = {}
    for rel in claim_evidence_relations:
        if isinstance(rel, ClaimEvidenceRelation):
            cid, eid = rel.claim_id, rel.evidence_id
        elif isinstance(rel, dict):
            cid, eid = rel.get("claim_id"), rel.get("evidence_id")
        else:
            continue
        if cid is not None and eid is not None:
            relations_by_claim.setdefault(int(cid), []).append(int(eid))

    lineage: dict[str, dict[str, Any]] = {}

    for cid in sorted(referenced_ids):
        claim = claim_by_id.get(cid)
        if claim is None:
            continue

        meta = claim_meta.get(cid, {})
        seen_evidence: set[str] = set()
        evidence_items: list[dict[str, str | None]] = []

        for eid in relations_by_claim.get(cid, []):
            ev = evidence_by_id.get(eid)
            if ev is None:
                continue
            display_id = ev.metadata.get("display_id") if ev.metadata else None
            snippet = (ev.exact_text or "")[:500] or None
            source = source_by_id.get(ev.source_id)
            source_title = source.title if source else None
            source_url = source.url if source else None

            key = _evidence_key(eid, display_id, snippet)
            if key in seen_evidence:
                continue
            seen_evidence.add(key)
            evidence_items.append({
                "display_id": display_id,
                "snippet": snippet,
                "source_title": source_title or None,
                "source_url": source_url or None,
            })

        lineage[str(cid)] = {
            "text": _claim_text(claim),
            "knowledge_category": meta.get("knowledge_category"),
            "verification_status": meta.get("verification_status"),
            "evidence": evidence_items,
        }

    return lineage


def build_decision_brief_payload(state: AgentState) -> dict | None:
    """
    Build a compact Decision Brief presentation payload from terminal agent state.

    Returns None when decision_synthesis is absent or invalid.
  """
    synthesis_raw = state.get("decision_synthesis")
    if not synthesis_raw or not isinstance(synthesis_raw, dict):
        return None

    status = synthesis_raw.get("recommendation_status")
    if status not in _VALID_STATUSES:
        return None

    option_evaluation = _as_dict(state.get("option_evaluation"))
    decision_frame = _as_dict(state.get("decision_frame"))
    knowledge_state = _as_dict(state.get("knowledge_state"))

    referenced_ids = _collect_referenced_claim_ids(option_evaluation, synthesis_raw)
    claim_meta = _build_claim_meta(
        knowledge_state,
        state.get("verification_results") or [],
    )
    claim_lineage = _build_claim_lineage(
        referenced_ids,
        state.get("material_claims") or [],
        state.get("claim_evidence_relations") or [],
        state.get("validated_evidence") or [],
        state.get("normalized_sources") or [],
        claim_meta,
    )

    return {
        "decision_frame": decision_frame,
        "option_evaluation": option_evaluation,
        "decision_synthesis": synthesis_raw,
        "claim_lineage": claim_lineage,
    }
