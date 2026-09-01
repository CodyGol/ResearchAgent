"""Evidence extraction pipeline: extract → validate → deduplicate → persist."""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from domain.models import Evidence, EvidenceMatchType, ExtractionMethod, Source
from services.evidence_extractor import extract_candidates_from_source, map_evidence_type
from services.evidence_validator import extract_context, validate_evidence_text
from services.source_normalizer import normalize_claim_text

logger = logging.getLogger(__name__)


@dataclass
class EvidenceExtractionMetrics:
    """Observability metrics for a single evidence extraction pass."""

    sources_processed: int = 0
    sources_with_evidence: int = 0
    candidate_count: int = 0
    validated_count: int = 0
    rejected_count: int = 0
    duplicate_count: int = 0
    extraction_failures: int = 0
    validation_failures: int = 0
    processing_time_ms: float = 0.0
    model_name: str = ""
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources_processed": self.sources_processed,
            "sources_with_evidence": self.sources_with_evidence,
            "candidate_count": self.candidate_count,
            "validated_count": self.validated_count,
            "rejected_count": self.rejected_count,
            "duplicate_count": self.duplicate_count,
            "extraction_failures": self.extraction_failures,
            "validation_failures": self.validation_failures,
            "processing_time_ms": self.processing_time_ms,
            "model_name": self.model_name,
            "failure_count": len(self.failures),
        }


def _evidence_fingerprint(source_id: int, normalized_text: str) -> str:
    return hashlib.sha256(f"{source_id}:{normalized_text}".encode()).hexdigest()


def _ensure_source_ids(sources: list[Source]) -> list[Source]:
    """Assign temporary IDs to sources that lack database IDs (transient runs)."""
    result = []
    for i, source in enumerate(sources):
        if source.id is None:
            source = source.model_copy(update={"id": -(i + 1)})
        result.append(source)
    return result


async def process_sources_for_evidence(
    sources: list[Source],
    research_question: str,
    research_run_id: int,
    *,
    is_persisted: bool = False,
    llm: Any | None = None,
    model_name: str = "",
) -> tuple[list[Evidence], EvidenceExtractionMetrics]:
    """
    Extract, validate, deduplicate, and persist evidence from all sources.

    Individual source failures do not abort the run.

    Args:
        sources: Normalized sources with snippet content
        research_question: Research question for relevance filtering
        research_run_id: Parent research run ID
        is_persisted: Whether to persist to database
        llm: Optional injectable LLM for testing
        model_name: Model name for metrics

    Returns:
        Tuple of (validated evidence list, extraction metrics)
    """
    from config import settings
    from db.evidence_repositories import get_evidence_repo, is_persistence_enabled

    start = time.monotonic()
    metrics = EvidenceExtractionMetrics(model_name=model_name or settings.model_name)
    validated: list[Evidence] = []
    seen_fingerprints: set[str] = set()

    sources = _ensure_source_ids(sources)

    for source in sources:
        metrics.sources_processed += 1
        source_id = source.id
        if source_id is None:
            continue

        if not source.content or not source.content.strip():
            metrics.failures.append({
                "source_id": source_id,
                "source_url": source.url,
                "failure_type": "empty_content",
                "error": "Source has no content to extract from",
            })
            continue

        # --- Extraction ---
        try:
            candidates = await extract_candidates_from_source(
                source, research_question, llm=llm
            )
        except Exception as e:
            metrics.extraction_failures += 1
            metrics.failures.append({
                "source_id": source_id,
                "source_url": source.url,
                "failure_type": "extraction_error",
                "error": str(e),
            })
            logger.warning(
                "Evidence extraction failed for source %s (%s): %s",
                source_id,
                source.url,
                e,
            )
            continue

        metrics.candidate_count += len(candidates)
        source_had_evidence = False

        for candidate in candidates:
            # --- Validation (mandatory) ---
            validation = validate_evidence_text(candidate.text, source.content)

            if not validation.is_valid:
                metrics.rejected_count += 1
                metrics.validation_failures += 1
                metrics.failures.append({
                    "source_id": source_id,
                    "source_url": source.url,
                    "failure_type": "validation_rejected",
                    "candidate_text": candidate.text[:120],
                    "match_type": validation.match_type.value,
                    "reason": validation.reason,
                })
                logger.info(
                    "Rejected fabricated/invalid evidence from %s: %s",
                    source.url,
                    validation.reason,
                )
                continue

            # --- Deduplication ---
            norm_text = validation.normalized_text or normalize_claim_text(candidate.text)
            fingerprint = _evidence_fingerprint(source_id, norm_text)
            if fingerprint in seen_fingerprints:
                metrics.duplicate_count += 1
                continue
            seen_fingerprints.add(fingerprint)

            # --- Context extraction ---
            context_before, context_after = extract_context(
                candidate.text, source.content, context_chars=80
            )

            evidence = Evidence(
                source_id=source_id,
                research_run_id=research_run_id,
                exact_text=candidate.text,
                normalized_text=norm_text,
                locator=candidate.locator,
                context_before=context_before,
                context_after=context_after,
                evidence_type=map_evidence_type(candidate.evidence_type),
                extraction_method=ExtractionMethod.LLM,
                match_type=validation.match_type,
                is_validated=True,
                metadata={
                    "relevance": candidate.relevance,
                    "llm_context": candidate.context,
                    "content_scope": "search_snippet",
                    "fingerprint": fingerprint,
                    "match_ratio": validation.match_ratio,
                    "validation_reason": validation.reason,
                    "source_url": source.url,
                    "source_title": source.title,
                },
            )
            validated.append(evidence)
            source_had_evidence = True
            metrics.validated_count += 1

        if source_had_evidence:
            metrics.sources_with_evidence += 1

    # --- Persistence ---
    if validated and is_persistence_enabled() and is_persisted:
        try:
            repo = get_evidence_repo()
            validated = await repo.save_evidence(validated)
            logger.info(
                "Persisted %d evidence records for run %s",
                len(validated),
                research_run_id,
            )
        except Exception as e:
            logger.warning("Failed to persist evidence: %s", e)
            metrics.failures.append({
                "failure_type": "persistence_error",
                "error": str(e),
            })

    metrics.processing_time_ms = round((time.monotonic() - start) * 1000, 2)

    logger.info(
        "Evidence extraction complete: %d validated, %d rejected, %d duplicates, "
        "%d extraction failures (%.0fms)",
        metrics.validated_count,
        metrics.rejected_count,
        metrics.duplicate_count,
        metrics.extraction_failures,
        metrics.processing_time_ms,
    )

    return validated, metrics
