"""Deterministic knowledge state derivation (Phase 2D)."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from domain.models import (
    Claim,
    ClaimEvidenceRelation,
    EvidenceConfidence,
    KnowledgeCategory,
    VerificationResult,
    VerificationStatus,
)
from services.knowledge_state_schemas import (
    InformationGap,
    KnowledgeState,
    KnowledgeStateEntry,
)
from state import Critique

logger = logging.getLogger(__name__)


def map_verification_to_category(
    status: VerificationStatus,
    confidence: EvidenceConfidence,
) -> KnowledgeCategory | None:
    """Map verification outcome to persisted knowledge category."""
    if status == VerificationStatus.SUPPORTED and confidence == EvidenceConfidence.HIGH:
        return KnowledgeCategory.KNOWN
    if status == VerificationStatus.PARTIALLY_SUPPORTED:
        return KnowledgeCategory.LIKELY
    if status == VerificationStatus.UNCERTAIN:
        return KnowledgeCategory.DISPUTED
    if status == VerificationStatus.INSUFFICIENT_EVIDENCE:
        return KnowledgeCategory.UNKNOWN
    if status in (
        VerificationStatus.CONTRADICTED,
        VerificationStatus.UNVERIFIABLE,
    ):
        return None
    return None


def _bucket_for_status(status: VerificationStatus) -> str:
    if status == VerificationStatus.SUPPORTED:
        return "known"
    if status == VerificationStatus.PARTIALLY_SUPPORTED:
        return "likely"
    if status == VerificationStatus.UNCERTAIN:
        return "disputed"
    if status == VerificationStatus.INSUFFICIENT_EVIDENCE:
        return "unknown"
    if status == VerificationStatus.CONTRADICTED:
        return "contradicted"
    if status == VerificationStatus.UNVERIFIABLE:
        return "unverifiable"
    return "unknown"


def _relations_for_claim(
    claim_id: int,
    relations: list[ClaimEvidenceRelation],
) -> tuple[list[int], list[int]]:
    claim_relations = [rel for rel in relations if rel.claim_id == claim_id]
    relation_ids = [rel.id for rel in claim_relations if rel.id is not None]
    evidence_ids = sorted({rel.evidence_id for rel in claim_relations})
    return relation_ids, evidence_ids


def derive_knowledge_state(
    *,
    material_claims: list[Claim],
    verification_results: list[VerificationResult],
    claim_evidence_relations: list[ClaimEvidenceRelation],
    critique: Critique | None = None,
) -> tuple[KnowledgeState, list[VerificationResult]]:
    """
    Derive run-level knowledge state from verification results and critic gaps.

    Phase 2D applies only to full-pipeline runs with verification_results present.
    """
    start = time.monotonic()
    verification_by_claim = {
        vr.claim_id: vr for vr in verification_results if vr.claim_id is not None
    }

    buckets: dict[str, list[KnowledgeStateEntry]] = defaultdict(list)
    updated_verifications: list[VerificationResult] = []
    orphan_material_claims = 0
    orphan_claim_ids: list[int] = []

    for claim in material_claims:
        if claim.id is None:
            continue

        verification = verification_by_claim.get(claim.id)
        if verification is None:
            orphan_material_claims += 1
            orphan_claim_ids.append(claim.id)
            logger.warning(
                "Orphan material claim without verification result: claim_id=%s",
                claim.id,
            )
            continue

        knowledge_category = map_verification_to_category(
            verification.status,
            verification.confidence,
        )
        relation_ids, evidence_ids = _relations_for_claim(
            claim.id,
            claim_evidence_relations,
        )
        entry = KnowledgeStateEntry(
            claim_id=claim.id,
            verification_id=verification.id,
            knowledge_category=knowledge_category,
            verification_status=verification.status,
            confidence=verification.confidence,
            relation_ids=relation_ids,
            evidence_ids=evidence_ids,
        )
        buckets[_bucket_for_status(verification.status)].append(entry)

        updated = verification.model_copy(update={"knowledge_category": knowledge_category})
        updated_verifications.append(updated)

    information_gaps = [
        InformationGap(description=area.strip(), source="critic_unsupported_area")
        for area in (critique.unsupported_areas if critique else [])
        if area and area.strip()
    ]

    metrics = {
        "known_count": len(buckets["known"]),
        "likely_count": len(buckets["likely"]),
        "disputed_count": len(buckets["disputed"]),
        "unknown_count": len(buckets["unknown"]),
        "contradicted_count": len(buckets["contradicted"]),
        "unverifiable_count": len(buckets["unverifiable"]),
        "information_gap_count": len(information_gaps),
        "orphan_material_claims": orphan_material_claims,
        "orphan_claim_ids": orphan_claim_ids,
        "derivation_time_ms": round((time.monotonic() - start) * 1000, 2),
    }

    knowledge_state = KnowledgeState(
        known=buckets["known"],
        likely=buckets["likely"],
        disputed=buckets["disputed"],
        unknown=buckets["unknown"],
        contradicted=buckets["contradicted"],
        unverifiable=buckets["unverifiable"],
        information_gaps=information_gaps,
        metrics=metrics,
    )
    return knowledge_state, updated_verifications


def knowledge_state_snapshot(knowledge_state: KnowledgeState) -> dict:
    """Compact JSON-serializable snapshot for research-run metadata."""
    return knowledge_state.model_dump(mode="json")
