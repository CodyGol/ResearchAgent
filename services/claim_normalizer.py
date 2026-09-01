"""Claim text normalization for deduplication (preserves negation and qualifiers)."""

import hashlib
import re


def normalize_claim_for_dedup(
    text: str,
    temporal_scope: str | None = None,
    geographic_scope: str | None = None,
) -> str:
    """
    Normalize claim text for conservative duplicate detection.

    Preserves negation and does not strip numbers/units.
    """
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    # Normalize quotes/dashes but keep negation words
    normalized = normalized.replace("'", "'")
    parts = [normalized]
    if temporal_scope:
        parts.append(f"t:{temporal_scope.lower().strip()}")
    if geographic_scope:
        parts.append(f"g:{geographic_scope.lower().strip()}")
    return "|".join(parts)


def claim_fingerprint(
    research_run_id: int,
    normalized_key: str,
) -> str:
    """Stable fingerprint for idempotent claim persistence within a run."""
    return hashlib.sha256(f"{research_run_id}:{normalized_key}".encode()).hexdigest()


def token_jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity on word tokens — used for conservative semantic dedup."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
