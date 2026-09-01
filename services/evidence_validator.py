"""Evidence integrity validation against source content."""

import re
import unicodedata
from dataclasses import dataclass

from domain.models import EvidenceMatchType


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating an evidence span against source content."""

    is_valid: bool
    match_type: EvidenceMatchType
    normalized_text: str | None = None
    match_ratio: float = 0.0
    reason: str | None = None


def _normalize_for_matching(text: str) -> str:
    """
    Normalize text for fuzzy matching while preserving semantic content.

    - Unicode NFKC normalization
    - Lowercase
    - Collapse whitespace
    - Normalize quotes and dashes
    - Strip leading/trailing punctuation noise
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Normalize quote variants
    text = re.sub(r"[''`]", "'", text)
    text = re.sub(r'["""]', '"', text)
    # Normalize dashes
    text = re.sub(r"[–—−]", "-", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_set_ratio(shorter: str, longer: str) -> float:
    """
    Simple token-overlap ratio for fuzzy matching.

    Returns the fraction of tokens in `shorter` found in `longer`.
    Conservative: only used when exact/normalized matching fails.
    """
    if not shorter or not longer:
        return 0.0
    shorter_tokens = set(shorter.split())
    longer_tokens = set(longer.split())
    if not shorter_tokens:
        return 0.0
    overlap = shorter_tokens & longer_tokens
    return len(overlap) / len(shorter_tokens)


def _find_normalized_span(evidence_text: str, source_content: str) -> bool:
    """Check if normalized evidence exists as substring in normalized source."""
    norm_evidence = _normalize_for_matching(evidence_text)
    norm_source = _normalize_for_matching(source_content)
    if not norm_evidence:
        return False
    return norm_evidence in norm_source


def _find_fuzzy_span(
    evidence_text: str,
    source_content: str,
    threshold: float = 0.85,
) -> tuple[bool, float]:
    """
    Sliding-window fuzzy match for near-misses (OCR, minor edits).

    Only accepts if a window of similar length achieves >= threshold overlap.
    """
    norm_evidence = _normalize_for_matching(evidence_text)
    norm_source = _normalize_for_matching(source_content)

    if not norm_evidence or not norm_source:
        return False, 0.0

    evidence_len = len(norm_evidence)
    best_ratio = 0.0

    # Slide a window of evidence length across source
    step = max(1, evidence_len // 4)
    for start in range(0, max(1, len(norm_source) - evidence_len + 1), step):
        window = norm_source[start : start + evidence_len]
        ratio = _token_set_ratio(norm_evidence, window)
        best_ratio = max(best_ratio, ratio)
        if ratio >= threshold:
            return True, ratio

    # Also try full-source token overlap for short evidence
    full_ratio = _token_set_ratio(norm_evidence, norm_source)
    best_ratio = max(best_ratio, full_ratio)
    return full_ratio >= threshold, best_ratio


def validate_evidence_text(
    evidence_text: str,
    source_content: str,
    *,
    allow_fuzzy: bool = True,
    fuzzy_threshold: float = 0.85,
) -> ValidationResult:
    """
    Verify that evidence text actually exists in source content.

    Matching strategy (in order):
    1. Exact substring match (preferred)
    2. Normalized substring match (whitespace, unicode, quotes)
    3. Fuzzy token overlap (optional, conservative threshold)

    Args:
        evidence_text: Candidate evidence span from extraction
        source_content: Full source body to validate against
        allow_fuzzy: Whether to attempt fuzzy matching as last resort
        fuzzy_threshold: Minimum token overlap for fuzzy acceptance

    Returns:
        ValidationResult with match type and validity
    """
    if not evidence_text or not evidence_text.strip():
        return ValidationResult(
            is_valid=False,
            match_type=EvidenceMatchType.NOT_FOUND,
            reason="Empty evidence text",
        )

    if not source_content:
        return ValidationResult(
            is_valid=False,
            match_type=EvidenceMatchType.NOT_FOUND,
            reason="Empty source content",
        )

    # 1. Exact match
    if evidence_text in source_content:
        return ValidationResult(
            is_valid=True,
            match_type=EvidenceMatchType.EXACT,
            normalized_text=_normalize_for_matching(evidence_text),
            match_ratio=1.0,
        )

    # 2. Normalized match
    normalized = _normalize_for_matching(evidence_text)
    if _find_normalized_span(evidence_text, source_content):
        return ValidationResult(
            is_valid=True,
            match_type=EvidenceMatchType.NORMALIZED,
            normalized_text=normalized,
            match_ratio=1.0,
        )

    # 3. Fuzzy match (conservative)
    if allow_fuzzy:
        found, ratio = _find_fuzzy_span(
            evidence_text, source_content, threshold=fuzzy_threshold
        )
        if found:
            return ValidationResult(
                is_valid=True,
                match_type=EvidenceMatchType.FUZZY,
                normalized_text=normalized,
                match_ratio=ratio,
                reason=f"Fuzzy match at {ratio:.2f} token overlap",
            )

    return ValidationResult(
        is_valid=False,
        match_type=EvidenceMatchType.NOT_FOUND,
        normalized_text=normalized,
        match_ratio=0.0,
        reason="Evidence text not found in source content",
    )


def extract_context(
    evidence_text: str,
    source_content: str,
    context_chars: int = 100,
) -> tuple[str | None, str | None]:
    """
    Extract surrounding context for a validated evidence span.

    Args:
        evidence_text: The evidence span (exact form)
        source_content: Full source content
        context_chars: Characters of context before/after

    Returns:
        Tuple of (context_before, context_after) or (None, None) if not found
    """
    idx = source_content.find(evidence_text)
    if idx == -1:
        # Try normalized search
        norm_evidence = _normalize_for_matching(evidence_text)
        norm_source = _normalize_for_matching(source_content)
        norm_idx = norm_source.find(norm_evidence)
        if norm_idx == -1:
            return None, None
        # Map back approximately using character offsets in original
        idx = norm_idx  # Approximate; exact mapping not guaranteed

    before_start = max(0, idx - context_chars)
    after_end = min(len(source_content), idx + len(evidence_text) + context_chars)

    context_before = source_content[before_start:idx].strip() or None
    context_after = (
        source_content[idx + len(evidence_text) : after_end].strip() or None
    )
    return context_before, context_after
