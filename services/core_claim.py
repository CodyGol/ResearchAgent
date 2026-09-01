"""Core canonical claim generation from structured fact values."""

import logging
from typing import Any

from domain.models import Claim, ClaimType, Evidence
from services.claim_normalizer import claim_fingerprint, normalize_claim_for_dedup
from services.claim_validator import validate_claim_support_deterministic
from services.fact_target import AnswerTarget
from services.fact_value import (
    StructuredFactValue,
    build_canonical_claim_from_value,
    extract_fact_value,
    validate_fact_value_in_evidence,
)

logger = logging.getLogger(__name__)


def build_core_claim_from_value(
    fact_value: StructuredFactValue,
    evidence: Evidence,
    research_run_id: int,
) -> Claim | None:
    """Build and validate canonical claim from structured fact value."""
    claim_text = build_canonical_claim_from_value(fact_value, evidence.exact_text)
    validation = validate_claim_support_deterministic(claim_text, evidence.exact_text)
    if not validation.is_supported:
        logger.info("Canonical claim failed validation: %s", validation.reason)
        return None

    norm_key = normalize_claim_for_dedup(
        claim_text, fact_value.temporal_scope, None
    )
    fingerprint = claim_fingerprint(research_run_id, norm_key)

    return Claim(
        research_run_id=research_run_id,
        text=claim_text,
        claim_type=ClaimType.FACTUAL,
        temporal_scope=fact_value.temporal_scope,
        raw_value=fact_value.value if fact_value.value_type.value == "number" else None,
        unit=fact_value.unit,
        currency=fact_value.currency,
        qualifiers=fact_value.qualifiers,
        metadata={
            "importance": "high",
            "support_basis": "direct",
            "relevance": "critical",
            "is_material": True,
            "is_core_claim": True,
            "fingerprint": fingerprint,
            "normalized_key": norm_key,
            "validation_reason": validation.reason,
            "fact_value": fact_value.model_dump(),
            "extraction_mode": "deterministic",
        },
    )


async def generate_core_claim(
    target: AnswerTarget,
    evidence: Evidence,
    research_run_id: int,
    *,
    fact_value: StructuredFactValue | None = None,
    llm: Any | None = None,
    use_llm: bool = False,
) -> Claim | None:
    """
    Generate core claim from validated structured fact value.

    LLM fallback disabled by default for fast path — value-driven only.
    """
    fv = fact_value or extract_fact_value(evidence.exact_text, target)
    if fv is None:
        return None

    valid, reason = validate_fact_value_in_evidence(fv, evidence.exact_text)
    if not valid:
        logger.info("Fact value validation failed: %s", reason)
        return None

    return build_core_claim_from_value(fv, evidence, research_run_id)
