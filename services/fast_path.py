"""SIMPLE_FACT fast path orchestration."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from domain.models import Claim, ClaimEvidenceRelation, ClaimEvidenceRelationship, Evidence
from services.core_claim import generate_core_claim
from services.evidence_context import assign_evidence_ids
from services.fact_sufficiency import (
    check_fact_sufficiency,
    detect_conflicting_values,
)
from services.fact_target import (
    AnswerTarget,
    build_targeted_search_query,
    official_domains_for_target,
)
from services.fact_value import cache_key_for_target, extract_fact_value
from services.fast_evidence import candidate_to_evidence, extract_decisive_evidence
from services.fast_writer import build_fast_answer
from services.research_run_service import persist_sources_for_run
from services.source_authority import prioritize_sources_for_domain
from services.source_normalizer import normalize_search_results_with_metrics
from tools.search import search_tavily_with_retry
from utils.observability import trace_llm_call

logger = logging.getLogger(__name__)

MAX_CORROBORATION = 1  # optional second evidence if cheap
MAX_SOURCES_TO_TRY = 3


@dataclass
class FastPathMetrics:
    """Observability for fast path execution."""

    fast_path_entered: bool = True
    target_attribute: str = ""
    target_domain: str = ""
    sources_retrieved: int = 0
    sources_processed: int = 0
    evidence_before_stop: int = 0
    stop_reason: str = ""
    core_claims_generated: int = 0
    full_claim_extractor_skipped: bool = True
    full_writer_skipped: bool = True
    llm_calls: int = 0
    llm_calls_evidence: int = 0
    llm_calls_core_claim: int = 0
    fact_values_extracted: int = 0
    cache_key: str = ""
    freshness: str = ""
    escalated: bool = False
    escalation_reason: str = ""
    processing_time_ms: float = 0.0
    model_calls_avoided_estimate: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fast_path_entered": self.fast_path_entered,
            "target_attribute": self.target_attribute,
            "target_domain": self.target_domain,
            "sources_retrieved": self.sources_retrieved,
            "sources_processed": self.sources_processed,
            "evidence_before_stop": self.evidence_before_stop,
            "stop_reason": self.stop_reason,
            "core_claims_generated": self.core_claims_generated,
            "full_claim_extractor_skipped": self.full_claim_extractor_skipped,
            "full_writer_skipped": self.full_writer_skipped,
            "llm_calls": self.llm_calls,
            "llm_calls_evidence": self.llm_calls_evidence,
            "llm_calls_core_claim": self.llm_calls_core_claim,
            "fact_values_extracted": self.fact_values_extracted,
            "cache_key": self.cache_key,
            "freshness": self.freshness,
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "processing_time_ms": self.processing_time_ms,
            "model_calls_avoided_estimate": self.model_calls_avoided_estimate,
        }


@dataclass
class FastPathResult:
    """Outcome of fast path execution."""

    success: bool
    escalate: bool = False
    escalation_reason: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    sources: list = field(default_factory=list)
    core_claim: Claim | None = None
    fact_value: Any | None = None
    relations: list[ClaimEvidenceRelation] = field(default_factory=list)
    metrics: FastPathMetrics = field(default_factory=FastPathMetrics)


async def run_fast_path(
    query: str,
    target: AnswerTarget,
    research_run_id: int,
    *,
    is_persisted: bool = False,
    llm: Any | None = None,
) -> FastPathResult:
    """
    Execute SIMPLE_FACT fast path: search → decisive evidence → core claim → answer.

    Stops as soon as sufficient evidence is found.
    Escalates to STANDARD pipeline on failure.
    """
    from langchain_anthropic import ChatAnthropic
    from config import settings

    start = time.monotonic()
    metrics = FastPathMetrics(
        target_attribute=target.attribute,
        target_domain=target.domain.value,
        cache_key=cache_key_for_target(target),
        freshness=target.freshness.value,
    )
    model = llm or ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        temperature=0.0,
    )

    search_query = build_targeted_search_query(target)
    official_domains = list(official_domains_for_target(target))

    async def _fetch_results() -> list:
        results: list = []
        if official_domains:
            try:
                official = await search_tavily_with_retry(
                    query=search_query,
                    max_results=3,
                    domains=official_domains,
                )
                results.extend(official)
            except Exception as e:
                logger.warning("Official domain search failed: %s", e)
        try:
            general = await search_tavily_with_retry(
                query=search_query,
                max_results=3,
            )
            results.extend(general)
        except Exception as e:
            if not results:
                raise e
        return results

    with trace_llm_call("fast_path", "search") as span:
        span.set_input({"query": search_query, "target": target.attribute})
        try:
            results = await _fetch_results()
            metrics.llm_calls += 0  # search is API not LLM
        except Exception as e:
            metrics.escalated = True
            metrics.escalation_reason = f"Search failed: {e}"
            metrics.processing_time_ms = round((time.monotonic() - start) * 1000, 2)
            return FastPathResult(
                success=False,
                escalate=True,
                escalation_reason=metrics.escalation_reason,
                metrics=metrics,
            )

    metrics.sources_retrieved = len(results)

    sources, _dedup = normalize_search_results_with_metrics(results, research_run_id)
    sources = prioritize_sources_for_domain(sources, target.domain)
    sources = sources[:MAX_SOURCES_TO_TRY]

    if is_persisted:
        sources = await persist_sources_for_run(
            research_run_id, query, sources, is_persisted=True
        )

    validated_evidence: list[Evidence] = []
    decisive_evidence: Evidence | None = None

    async def _process_sources(source_list: list) -> bool:
        nonlocal decisive_evidence
        for source in source_list:
            metrics.sources_processed += 1

            with trace_llm_call("fast_path", "extract_decisive_evidence"):
                candidate, used_llm = await extract_decisive_evidence(
                    source, target, llm=model, use_llm=True
                )
                if used_llm:
                    metrics.llm_calls += 1
                    metrics.llm_calls_evidence += 1

            if not candidate:
                continue

            evidence = candidate_to_evidence(
                candidate, source, research_run_id, used_llm=used_llm
            )
            validated_evidence.append(evidence)

            sufficiency = check_fact_sufficiency(
                target, evidence, source, existing_evidence=validated_evidence
            )

            if sufficiency.is_sufficient:
                decisive_evidence = evidence
                metrics.evidence_before_stop = len(validated_evidence)
                metrics.stop_reason = sufficiency.reason
                if sufficiency.fact_value:
                    metrics.fact_values_extracted = 1
                return True
        return False

    if not await _process_sources(sources):
        metrics.escalated = True
        metrics.escalation_reason = (
            "No decisive evidence found with adequate source authority"
        )
        metrics.processing_time_ms = round((time.monotonic() - start) * 1000, 2)
        return FastPathResult(
            success=False,
            escalate=True,
            escalation_reason=metrics.escalation_reason,
            evidence=validated_evidence,
            metrics=metrics,
        )

    conflict = detect_conflicting_values(
        [decisive_evidence] if decisive_evidence else validated_evidence, target
    )
    if conflict:
        metrics.escalated = True
        metrics.escalation_reason = conflict
        metrics.processing_time_ms = round((time.monotonic() - start) * 1000, 2)
        return FastPathResult(
            success=False,
            escalate=True,
            escalation_reason=conflict,
            evidence=validated_evidence,
            metrics=metrics,
        )

    # Optional corroboration from next source (only if we haven't stopped early)
    # Already stopped at decisive - skip extra sources

    # Persist evidence
    if validated_evidence and is_persisted:
        from db.evidence_repositories import get_evidence_repo, is_persistence_enabled

        if is_persistence_enabled():
            try:
                repo = get_evidence_repo()
                validated_evidence = await repo.save_evidence(validated_evidence)
                decisive_evidence = validated_evidence[0]
            except Exception as e:
                logger.warning("Failed to persist fast-path evidence: %s", e)

    validated_evidence = assign_evidence_ids(validated_evidence)
    decisive_evidence = validated_evidence[0]
    display_id = decisive_evidence.metadata.get("display_id", "E1")

    fact_value = extract_fact_value(decisive_evidence.exact_text, target)
    if fact_value is None:
        metrics.escalated = True
        metrics.escalation_reason = "Could not extract structured target value"
        metrics.processing_time_ms = round((time.monotonic() - start) * 1000, 2)
        return FastPathResult(
            success=False,
            escalate=True,
            escalation_reason=metrics.escalation_reason,
            evidence=validated_evidence,
            metrics=metrics,
        )

    metrics.fact_values_extracted = 1

    core_claim = await generate_core_claim(
        target,
        decisive_evidence,
        research_run_id,
        fact_value=fact_value,
        llm=model,
        use_llm=False,
    )

    if core_claim is None:
        metrics.escalated = True
        metrics.escalation_reason = "Core claim validation failed"
        metrics.processing_time_ms = round((time.monotonic() - start) * 1000, 2)
        return FastPathResult(
            success=False,
            escalate=True,
            escalation_reason=metrics.escalation_reason,
            evidence=validated_evidence,
            metrics=metrics,
        )

    metrics.core_claims_generated = 1

    # Persist claim + relation
    relations: list[ClaimEvidenceRelation] = []
    if is_persisted:
        from db.evidence_repositories import get_claim_repo, is_persistence_enabled

        if is_persistence_enabled():
            try:
                repo = get_claim_repo()
                saved = await repo.save_claims([core_claim])
                core_claim = saved[0]
                if core_claim.id and decisive_evidence.id:
                    rel = ClaimEvidenceRelation(
                        claim_id=core_claim.id,
                        evidence_id=decisive_evidence.id,
                        relationship=ClaimEvidenceRelationship.SUPPORTS,
                        reasoning="Core claim from decisive fast-path evidence",
                    )
                    relations = await repo.save_claim_evidence([rel])
            except Exception as e:
                logger.warning("Failed to persist core claim: %s", e)

    metrics.model_calls_avoided_estimate = 40  # vs full pipeline claim extraction
    metrics.processing_time_ms = round((time.monotonic() - start) * 1000, 2)

    return FastPathResult(
        success=True,
        evidence=validated_evidence,
        sources=sources,
        core_claim=core_claim,
        fact_value=fact_value,
        relations=relations,
        metrics=metrics,
    )
