"""Claim extraction pipeline: extract → validate → normalize → deduplicate → persist."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from domain.models import (
    Claim,
    ClaimEvidenceRelation,
    ClaimEvidenceRelationship,
    ClaimImportance,
    ClaimSupportBasis,
    ClaimType,
    Evidence,
    Source,
)
from services.claim_deduplicator import ClaimDeduplicator
from services.claim_extractor import extract_claims_from_evidence
from services.claim_normalizer import claim_fingerprint, normalize_claim_for_dedup
from services.claim_relevance import (
    ClaimRelevance,
    assess_claim_relevance,
    is_material_claim,
    should_validate_expensively,
)
from services.claim_schemas import CandidateClaimItem
from services.claim_validator import (
    ClaimValidationResult,
    validate_claim_support_deterministic,
    validate_claims_batch,
)
from utils.observability import trace_llm_call

logger = logging.getLogger(__name__)

_CLAIM_TYPE_MAP: dict[str, ClaimType] = {
    "factual": ClaimType.FACTUAL,
    "statistical": ClaimType.STATISTICAL,
    "comparative": ClaimType.COMPARATIVE,
    "causal": ClaimType.CAUSAL,
    "predictive": ClaimType.PREDICTIVE,
    "analytical": ClaimType.ANALYTICAL,
    "opinion": ClaimType.OPINION,
    "definitional": ClaimType.DEFINITIONAL,
}


@dataclass
class _PendingRelation:
    fingerprint: str
    evidence_id: int
    reasoning: str


@dataclass
class _PendingCandidate:
    index: int
    candidate: CandidateClaimItem
    relevance: ClaimRelevance


@dataclass
class ClaimExtractionMetrics:
    """Observability metrics for claim extraction pass."""

    evidence_items_processed: int = 0
    candidate_claims_generated: int = 0
    claims_accepted: int = 0
    claims_rejected: int = 0
    claims_rejected_unsupported: int = 0
    claims_rejected_non_direct: int = 0
    claims_rejected_deterministic: int = 0
    claims_rejected_relevance: int = 0
    duplicate_claims_merged: int = 0
    unique_claims_persisted: int = 0
    material_claims_count: int = 0
    importance_high: int = 0
    importance_medium: int = 0
    importance_low: int = 0
    extraction_failures: int = 0
    validation_failures: int = 0
    llm_validation_calls: int = 0
    validation_batches: int = 0
    evidence_with_zero_claims: int = 0
    processing_time_ms: float = 0.0
    model_name: str = ""
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        claims_per_evidence = (
            round(self.claims_accepted / self.evidence_items_processed, 2)
            if self.evidence_items_processed
            else 0.0
        )
        return {
            "evidence_items_processed": self.evidence_items_processed,
            "candidate_claims_generated": self.candidate_claims_generated,
            "claims_accepted": self.claims_accepted,
            "claims_rejected": self.claims_rejected,
            "claims_rejected_unsupported": self.claims_rejected_unsupported,
            "claims_rejected_non_direct": self.claims_rejected_non_direct,
            "claims_rejected_deterministic": self.claims_rejected_deterministic,
            "claims_rejected_relevance": self.claims_rejected_relevance,
            "duplicate_claims_merged": self.duplicate_claims_merged,
            "unique_claims_persisted": self.unique_claims_persisted,
            "material_claims_count": self.material_claims_count,
            "importance_high": self.importance_high,
            "importance_medium": self.importance_medium,
            "importance_low": self.importance_low,
            "extraction_failures": self.extraction_failures,
            "validation_failures": self.validation_failures,
            "llm_validation_calls": self.llm_validation_calls,
            "validation_batches": self.validation_batches,
            "evidence_with_zero_claims": self.evidence_with_zero_claims,
            "claims_per_evidence_item": claims_per_evidence,
            "processing_time_ms": self.processing_time_ms,
            "model_name": self.model_name,
            "failure_count": len(self.failures),
        }


def _map_claim_type(raw: str) -> ClaimType:
    return _CLAIM_TYPE_MAP.get(raw.lower().strip(), ClaimType.FACTUAL)


def _map_importance(raw: str) -> ClaimImportance:
    try:
        return ClaimImportance(raw.lower().strip())
    except ValueError:
        return ClaimImportance.MEDIUM


def _ensure_evidence_ids(evidence_list: list[Evidence]) -> list[Evidence]:
    result = []
    for i, ev in enumerate(evidence_list):
        if ev.id is None:
            ev = ev.model_copy(update={"id": -(i + 1)})
        result.append(ev)
    return result


def _candidate_to_claim(
    candidate: CandidateClaimItem,
    research_run_id: int,
    *,
    fingerprint: str,
    normalized_key: str,
    validation_reason: str,
    relevance: ClaimRelevance,
    is_material: bool,
) -> Claim:
    return Claim(
        research_run_id=research_run_id,
        text=candidate.claim_text.strip(),
        claim_type=_map_claim_type(candidate.claim_type),
        temporal_scope=candidate.temporal_scope,
        geographic_scope=candidate.geographic_scope,
        raw_value=candidate.raw_value,
        unit=candidate.unit,
        currency=candidate.currency,
        qualifiers=candidate.qualifiers,
        metadata={
            "importance": _map_importance(candidate.importance).value,
            "support_basis": candidate.support_basis.lower(),
            "entities": candidate.entities,
            "fingerprint": fingerprint,
            "normalized_key": normalized_key,
            "validation_reason": validation_reason,
            "relevance": relevance.value,
            "is_material": is_material,
        },
    )


def _accept_claim(
    candidate: CandidateClaimItem,
    validation: ClaimValidationResult,
    relevance: ClaimRelevance,
    *,
    research_run_id: int,
    evidence_id: int,
    deduplicator: ClaimDeduplicator,
    claims_by_fingerprint: dict[str, Claim],
    pending_relations: list[_PendingRelation],
    metrics: ClaimExtractionMetrics,
) -> None:
    normalized_key = normalize_claim_for_dedup(
        candidate.claim_text,
        candidate.temporal_scope,
        candidate.geographic_scope,
    )
    fingerprint = claim_fingerprint(research_run_id, normalized_key)
    material = is_material_claim(relevance)

    with trace_llm_call("claim_deduplicator", "deduplicate_claim"):
        existing = deduplicator.find_canonical(
            candidate.claim_text,
            temporal_scope=candidate.temporal_scope,
            geographic_scope=candidate.geographic_scope,
            fingerprint=fingerprint,
        )

    if existing is not None:
        deduplicator.record_merge()
        metrics.duplicate_claims_merged += 1
        existing_fp = existing.metadata.get("fingerprint", fingerprint)
        pending_relations.append(
            _PendingRelation(
                fingerprint=existing_fp,
                evidence_id=evidence_id,
                reasoning=validation.reason,
            )
        )
    else:
        claim = _candidate_to_claim(
            candidate,
            research_run_id,
            fingerprint=fingerprint,
            normalized_key=normalized_key,
            validation_reason=validation.reason,
            relevance=relevance,
            is_material=material,
        )
        claims_by_fingerprint[fingerprint] = claim
        deduplicator.register(fingerprint, claim)
        pending_relations.append(
            _PendingRelation(
                fingerprint=fingerprint,
                evidence_id=evidence_id,
                reasoning=validation.reason,
            )
        )

    metrics.claims_accepted += 1
    if material:
        metrics.material_claims_count += 1
    _increment_importance(metrics, candidate.importance)


async def process_evidence_for_claims(
    validated_evidence: list[Evidence],
    research_question: str,
    research_run_id: int,
    *,
    is_persisted: bool = False,
    sources: list[Source] | None = None,
    llm: Any | None = None,
    model_name: str = "",
    use_llm_validation: bool = True,
    claim_depth: str = "moderate",
    use_batch_validation: bool = True,
) -> tuple[list[Claim], list[Claim], list[ClaimEvidenceRelation], ClaimExtractionMetrics]:
    """
    Extract, validate, deduplicate, and persist claims from validated evidence.

    Returns (all_claims, material_claims, relations, metrics).
    """
    from config import settings
    from db.evidence_repositories import get_claim_repo, is_persistence_enabled

    start = time.monotonic()
    metrics = ClaimExtractionMetrics(model_name=model_name or settings.model_name)
    deduplicator = ClaimDeduplicator()
    claims_by_fingerprint: dict[str, Claim] = {}
    pending_relations: list[_PendingRelation] = []

    sources_by_id: dict[int, Source] = {}
    if sources:
        for source in sources:
            if source.id is not None:
                sources_by_id[source.id] = source

    validated_evidence = _ensure_evidence_ids(validated_evidence)
    candidate_counter = 0

    for evidence in validated_evidence:
        metrics.evidence_items_processed += 1
        evidence_id = evidence.id
        if evidence_id is None:
            continue

        source = sources_by_id.get(evidence.source_id)
        claims_from_evidence = 0

        try:
            with trace_llm_call("claim_extractor", "extract_claims_for_evidence"):
                candidates = await extract_claims_from_evidence(
                    evidence,
                    research_question,
                    source=source,
                    llm=llm,
                    claim_depth=claim_depth,
                )
        except Exception as e:
            metrics.extraction_failures += 1
            metrics.failures.append({
                "evidence_id": evidence_id,
                "failure_type": "extraction_error",
                "error": str(e),
            })
            logger.warning(
                "Claim extraction failed for evidence %s: %s", evidence_id, e
            )
            continue

        metrics.candidate_claims_generated += len(candidates)

        # Phase 1: deterministic + relevance pre-filter
        pending_for_batch: list[_PendingCandidate] = []

        for candidate in candidates:
            candidate_counter += 1
            idx = candidate_counter

            basis = candidate.support_basis.lower().strip()
            if basis != ClaimSupportBasis.DIRECT.value:
                metrics.claims_rejected += 1
                metrics.claims_rejected_non_direct += 1
                metrics.failures.append({
                    "evidence_id": evidence_id,
                    "failure_type": "non_direct_claim",
                    "claim_text": candidate.claim_text[:120],
                    "support_basis": basis,
                })
                continue

            det = validate_claim_support_deterministic(
                candidate.claim_text,
                evidence.exact_text,
                support_basis=candidate.support_basis,
            )
            if not det.is_supported:
                metrics.claims_rejected += 1
                metrics.claims_rejected_deterministic += 1
                metrics.claims_rejected_unsupported += 1
                metrics.failures.append({
                    "evidence_id": evidence_id,
                    "failure_type": "deterministic_reject",
                    "claim_text": candidate.claim_text[:120],
                    "reason": det.reason,
                })
                continue

            relevance = assess_claim_relevance(
                candidate, research_question, claim_depth=claim_depth
            )
            if not should_validate_expensively(relevance, claim_depth):
                metrics.claims_rejected += 1
                metrics.claims_rejected_relevance += 1
                metrics.failures.append({
                    "evidence_id": evidence_id,
                    "failure_type": "relevance_reject",
                    "claim_text": candidate.claim_text[:120],
                    "relevance": relevance.value,
                })
                continue

            pending_for_batch.append(
                _PendingCandidate(index=idx, candidate=candidate, relevance=relevance)
            )

        # Phase 2: batch LLM validation for remaining candidates
        if pending_for_batch and use_llm_validation and llm is not None:
            if use_batch_validation and len(pending_for_batch) > 1:
                metrics.validation_batches += 1
                metrics.llm_validation_calls += 1
                try:
                    with trace_llm_call("claim_validator", "validate_claims_batch"):
                        batch_results = await validate_claims_batch(
                            [
                                (p.index, p.candidate.claim_text)
                                for p in pending_for_batch
                            ],
                            evidence.exact_text,
                            llm=llm,
                        )
                except Exception as e:
                    metrics.validation_failures += 1
                    metrics.failures.append({
                        "evidence_id": evidence_id,
                        "failure_type": "batch_validation_error",
                        "error": str(e),
                    })
                    batch_results = {}
            else:
                batch_results = {}
                for p in pending_for_batch:
                    metrics.llm_validation_calls += 1
                    try:
                        with trace_llm_call("claim_validator", "validate_claim_support"):
                            from services.claim_validator import validate_claim_support

                            result = await validate_claim_support(
                                p.candidate.claim_text,
                                evidence.exact_text,
                                support_basis=p.candidate.support_basis,
                                llm=llm,
                                use_llm=True,
                            )
                        batch_results[p.index] = result
                    except Exception as e:
                        metrics.validation_failures += 1
                        from domain.models import ClaimSupportStatus

                        batch_results[p.index] = ClaimValidationResult(
                            is_supported=False,
                            status=ClaimSupportStatus.NOT_SUPPORTED_BY_ORIGIN_EVIDENCE,
                            reason=str(e),
                        )

            for p in pending_for_batch:
                validation = batch_results.get(p.index)
                if validation is None:
                    validation = validate_claim_support_deterministic(
                        p.candidate.claim_text, evidence.exact_text
                    )

                if not validation.is_supported:
                    metrics.claims_rejected += 1
                    metrics.claims_rejected_unsupported += 1
                    metrics.failures.append({
                        "evidence_id": evidence_id,
                        "failure_type": "unsupported_claim",
                        "claim_text": p.candidate.claim_text[:120],
                        "reason": validation.reason,
                    })
                    continue

                _accept_claim(
                    p.candidate,
                    validation,
                    p.relevance,
                    research_run_id=research_run_id,
                    evidence_id=evidence_id,
                    deduplicator=deduplicator,
                    claims_by_fingerprint=claims_by_fingerprint,
                    pending_relations=pending_relations,
                    metrics=metrics,
                )
                claims_from_evidence += 1

        elif pending_for_batch:
            # No LLM — accept deterministic-passing relevant claims
            for p in pending_for_batch:
                det_ok = validate_claim_support_deterministic(
                    p.candidate.claim_text, evidence.exact_text
                )
                _accept_claim(
                    p.candidate,
                    det_ok,
                    p.relevance,
                    research_run_id=research_run_id,
                    evidence_id=evidence_id,
                    deduplicator=deduplicator,
                    claims_by_fingerprint=claims_by_fingerprint,
                    pending_relations=pending_relations,
                    metrics=metrics,
                )
                claims_from_evidence += 1

        if claims_from_evidence == 0:
            metrics.evidence_with_zero_claims += 1

    unique_claims = list(claims_by_fingerprint.values())
    material_claims = [
        c for c in unique_claims if c.metadata.get("is_material", False)
    ]
    relations: list[ClaimEvidenceRelation] = []

    if unique_claims and is_persistence_enabled() and is_persisted:
        try:
            with trace_llm_call("claim_repository", "persist_claim"):
                repo = get_claim_repo()
                saved_claims = await repo.save_claims(unique_claims)
                fp_to_id: dict[str, int] = {}
                for saved in saved_claims:
                    fp = saved.metadata.get("fingerprint")
                    if fp and saved.id is not None:
                        fp_to_id[fp] = saved.id

                relation_records = []
                for pending in pending_relations:
                    claim_id = fp_to_id.get(pending.fingerprint)
                    if claim_id is None:
                        continue
                    relation_records.append(
                        ClaimEvidenceRelation(
                            claim_id=claim_id,
                            evidence_id=pending.evidence_id,
                            relationship=ClaimEvidenceRelationship.SUPPORTS,
                            reasoning=pending.reasoning,
                        )
                    )

                if relation_records:
                    with trace_llm_call("claim_repository", "persist_claim_evidence"):
                        relations = await repo.save_claim_evidence(relation_records)

                unique_claims = saved_claims
                material_claims = [
                    c for c in unique_claims if c.metadata.get("is_material", False)
                ]
        except Exception as e:
            logger.warning("Failed to persist claims: %s", e)
            metrics.failures.append({
                "failure_type": "persistence_error",
                "error": str(e),
            })
    else:
        fp_to_id: dict[str, int] = {}
        for i, (fp, claim) in enumerate(claims_by_fingerprint.items()):
            claim_id = -(i + 1)
            claim = claim.model_copy(update={"id": claim_id})
            claims_by_fingerprint[fp] = claim
            fp_to_id[fp] = claim_id
        unique_claims = list(claims_by_fingerprint.values())
        material_claims = [
            c for c in unique_claims if c.metadata.get("is_material", False)
        ]

        for pending in pending_relations:
            claim_id = fp_to_id.get(pending.fingerprint)
            if claim_id is None:
                continue
            relations.append(
                ClaimEvidenceRelation(
                    claim_id=claim_id,
                    evidence_id=pending.evidence_id,
                    relationship=ClaimEvidenceRelationship.SUPPORTS,
                    reasoning=pending.reasoning,
                )
            )

    metrics.unique_claims_persisted = len(unique_claims)
    metrics.material_claims_count = len(material_claims)
    metrics.processing_time_ms = round((time.monotonic() - start) * 1000, 2)

    logger.info(
        "Claim extraction complete: %d accepted (%d material), %d rejected, "
        "%d merged, %d unique, %d LLM calls, %d batches (%.0fms)",
        metrics.claims_accepted,
        metrics.material_claims_count,
        metrics.claims_rejected,
        metrics.duplicate_claims_merged,
        metrics.unique_claims_persisted,
        metrics.llm_validation_calls,
        metrics.validation_batches,
        metrics.processing_time_ms,
    )

    return unique_claims, material_claims, relations, metrics


def _increment_importance(metrics: ClaimExtractionMetrics, raw: str) -> None:
    importance = _map_importance(raw)
    if importance == ClaimImportance.HIGH:
        metrics.importance_high += 1
    elif importance == ClaimImportance.LOW:
        metrics.importance_low += 1
    else:
        metrics.importance_medium += 1
