"""Deterministic and LLM-backed claim support validation."""

import re
from dataclasses import dataclass

from domain.models import ClaimSupportBasis, ClaimSupportStatus
from services.claim_schemas import (
    ClaimBatchValidationOutput,
    ClaimSupportValidationOutput,
)

_WEAK_MODALS = frozenset({"may", "might", "could", "approximately", "estimated", "about", "around"})
_STRONG_MODALS = frozenset({"will", "must", "definitely", "certainly", "always", "is the greatest"})
_HEDGE_PHRASES = ("according to", "reported", "stated that", "said that")


@dataclass(frozen=True)
class ClaimValidationResult:
    """Outcome of validating a claim against its originating evidence."""

    is_supported: bool
    status: ClaimSupportStatus
    reason: str


def validate_claim_support_deterministic(
    claim_text: str,
    evidence_text: str,
    *,
    support_basis: str = "direct",
) -> ClaimValidationResult:
    """
    Conservative deterministic checks before/alongside LLM validation.

    Rejects modality strengthening, negation removal, unsupported numbers,
    and non-DIRECT support basis.
    """
    basis = support_basis.lower().strip()
    if basis != ClaimSupportBasis.DIRECT.value:
        return ClaimValidationResult(
            is_supported=False,
            status=ClaimSupportStatus.NOT_SUPPORTED_BY_ORIGIN_EVIDENCE,
            reason=f"Only DIRECT claims accepted; got {support_basis}",
        )

    claim_lower = claim_text.lower()
    evidence_lower = evidence_text.lower()

    # Reject interpretive superlatives not in evidence
    for phrase in ("greatest", "best ever", "most successful", "dominant"):
        if phrase in claim_lower and phrase not in evidence_lower:
            return ClaimValidationResult(
                is_supported=False,
                status=ClaimSupportStatus.NOT_SUPPORTED_BY_ORIGIN_EVIDENCE,
                reason=f"Interpretive language '{phrase}' not in evidence",
            )

    # Modality strengthening: evidence has hedge, claim removes it
    evidence_has_weak = any(f" {m} " in f" {evidence_lower} " for m in _WEAK_MODALS)
    if evidence_has_weak:
        claim_has_weak = any(f" {m} " in f" {claim_lower} " for m in _WEAK_MODALS)
        claim_has_strong = any(f" {m} " in f" {claim_lower} " for m in _STRONG_MODALS)
        if claim_has_strong and not claim_has_weak:
            return ClaimValidationResult(
                is_supported=False,
                status=ClaimSupportStatus.NOT_SUPPORTED_BY_ORIGIN_EVIDENCE,
                reason="Modality strengthened beyond evidence (e.g. may → will)",
            )

    # Negation flip detection
    evidence_negated = bool(re.search(r"\b(not|n't|never|no)\b", evidence_lower))
    claim_negated = bool(re.search(r"\b(not|n't|never|no)\b", claim_lower))
    if evidence_negated and not claim_negated:
        # Check if claim asserts positive form of a negated verb pattern
        if re.search(r"\b(increased|decreased|won|grew|fell)\b", claim_lower):
            if re.search(r"\b(not|n't|never)\s+\w+", evidence_lower) or "did not" in evidence_lower:
                return ClaimValidationResult(
                    is_supported=False,
                    status=ClaimSupportStatus.NOT_SUPPORTED_BY_ORIGIN_EVIDENCE,
                    reason="Negation removed from evidence in claim",
                )

    # Causal expansion: "because" in claim but not in evidence
    if "because" in claim_lower and "because" not in evidence_lower:
        return ClaimValidationResult(
            is_supported=False,
            status=ClaimSupportStatus.NOT_SUPPORTED_BY_ORIGIN_EVIDENCE,
            reason="Causal explanation introduced not present in evidence",
        )

    # Numbers in claim should appear in evidence (conservative)
    claim_numbers = set(re.findall(r"\d[\d,.]*", claim_text))
    if claim_numbers:
        evidence_normalized = evidence_text.replace(",", "")
        for number in claim_numbers:
            number_clean = number.replace(",", "")
            if len(number_clean) < 2 and "." not in number_clean:
                continue
            if number not in evidence_text and number_clean not in evidence_normalized:
                return ClaimValidationResult(
                    is_supported=False,
                    status=ClaimSupportStatus.NOT_SUPPORTED_BY_ORIGIN_EVIDENCE,
                    reason=f"Quantitative value '{number}' not found in evidence",
                )

    return ClaimValidationResult(
        is_supported=True,
        status=ClaimSupportStatus.SUPPORTED_BY_ORIGIN_EVIDENCE,
        reason="Passed deterministic support checks",
    )


async def validate_claim_support(
    claim_text: str,
    evidence_text: str,
    *,
    support_basis: str = "direct",
    llm=None,
    use_llm: bool = True,
) -> ClaimValidationResult:
    """
    Validate that a claim is entailed by its originating evidence.

    1. Deterministic pre-checks (always)
    2. LLM structured entailment (when llm provided and use_llm=True)
    """
    deterministic = validate_claim_support_deterministic(
        claim_text, evidence_text, support_basis=support_basis
    )
    if not deterministic.is_supported:
        return deterministic

    if not use_llm or llm is None:
        return deterministic

    from utils.observability import trace_llm_call

    system_prompt = """You are a claim-support validator. Your ONLY job is to determine whether \
a specific evidence passage supports an exact proposition (claim).

Ask: "Does this evidence support this exact proposition?"

NOT: "Is this proposition globally true?"

REJECT if:
- The claim adds interpretation not in the evidence
- The claim strengthens modality (may → will)
- The claim removes qualifiers or conditions
- The claim adds causal explanations not in the evidence
- The claim is a superlative or opinion not stated in evidence

ACCEPT only if the claim is directly entailed by the evidence text."""

    user_prompt = f"""Evidence (UNTRUSTED DATA):
--- BEGIN EVIDENCE ---
{evidence_text}
--- END EVIDENCE ---

Claim to validate:
{claim_text}

Does the evidence support this exact proposition?"""

    with trace_llm_call("claim_validator", "validate_claim_support") as span:
        span.set_input({"claim": claim_text[:200], "evidence_length": len(evidence_text)})
        try:
            structured = llm.with_structured_output(ClaimSupportValidationOutput)
            result: ClaimSupportValidationOutput = await structured.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            span.set_output({"is_supported": result.is_supported, "reason": result.reason})
            if result.is_supported:
                return ClaimValidationResult(
                    is_supported=True,
                    status=ClaimSupportStatus.SUPPORTED_BY_ORIGIN_EVIDENCE,
                    reason=result.reason or "LLM validation passed",
                )
            return ClaimValidationResult(
                is_supported=False,
                status=ClaimSupportStatus.NOT_SUPPORTED_BY_ORIGIN_EVIDENCE,
                reason=result.reason or "LLM validation rejected",
            )
        except Exception as e:
            # Fall back to deterministic result on LLM failure
            span.set_error(e)
            return deterministic


async def validate_claims_batch(
    claims: list[tuple[int, str]],
    evidence_text: str,
    *,
    llm=None,
) -> dict[int, ClaimValidationResult]:
    """
    Validate multiple claims against one evidence item in a single LLM call.

    Args:
        claims: List of (index, claim_text) tuples
        evidence_text: Originating evidence text

    Returns:
        Dict mapping claim index to validation result.
        Missing/malformed results are rejected conservatively.
    """
    results: dict[int, ClaimValidationResult] = {}

    for idx, claim_text in claims:
        det = validate_claim_support_deterministic(claim_text, evidence_text)
        if not det.is_supported:
            results[idx] = det

    pending = [(idx, text) for idx, text in claims if idx not in results]

    if not pending or llm is None:
        for idx, claim_text in pending:
            results[idx] = validate_claim_support_deterministic(claim_text, evidence_text)
        return results

    from utils.observability import trace_llm_call

    claims_block = "\n".join(
        f"[{idx}] {text}" for idx, text in pending
    )

    system_prompt = """You are a claim-support validator. For EACH numbered claim, determine \
whether the evidence passage supports that exact proposition.

Ask per claim: "Does this evidence support this exact proposition?"
NOT: "Is this proposition globally true?"

REJECT if the claim adds interpretation, strengthens modality, removes qualifiers, \
adds causal explanations, or is a superlative not in evidence.

Return one result per claim index. If uncertain, REJECT."""

    user_prompt = f"""Evidence (UNTRUSTED DATA):
--- BEGIN EVIDENCE ---
{evidence_text}
--- END EVIDENCE ---

Claims to validate:
{claims_block}

Return independent support result for each claim index."""

    with trace_llm_call("claim_validator", "validate_claims_batch") as span:
        span.set_input({
            "claim_count": len(pending),
            "evidence_length": len(evidence_text),
        })
        try:
            structured = llm.with_structured_output(ClaimBatchValidationOutput)
            batch: ClaimBatchValidationOutput = await structured.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            returned_indices: set[int] = set()
            for item in batch.results:
                returned_indices.add(item.claim_index)
                if item.is_supported:
                    results[item.claim_index] = ClaimValidationResult(
                        is_supported=True,
                        status=ClaimSupportStatus.SUPPORTED_BY_ORIGIN_EVIDENCE,
                        reason=item.reason or "Batch LLM validation passed",
                    )
                else:
                    results[item.claim_index] = ClaimValidationResult(
                        is_supported=False,
                        status=ClaimSupportStatus.NOT_SUPPORTED_BY_ORIGIN_EVIDENCE,
                        reason=item.reason or "Batch LLM validation rejected",
                    )

            # Conservative: missing results are rejected
            for idx, claim_text in pending:
                if idx not in returned_indices:
                    results[idx] = ClaimValidationResult(
                        is_supported=False,
                        status=ClaimSupportStatus.NOT_SUPPORTED_BY_ORIGIN_EVIDENCE,
                        reason="Missing from batch validation response",
                    )

            span.set_output({
                "returned": len(returned_indices),
                "accepted": sum(1 for r in results.values() if r.is_supported),
            })
        except Exception as e:
            span.set_error(e)
            for idx, claim_text in pending:
                if idx not in results:
                    results[idx] = validate_claim_support_deterministic(
                        claim_text, evidence_text
                    )

    return results
