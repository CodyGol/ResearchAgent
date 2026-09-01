"""Conservative claim deduplication within a research run."""

import logging

from domain.models import Claim
from services.claim_normalizer import normalize_claim_for_dedup, token_jaccard_similarity

logger = logging.getLogger(__name__)

# Conservative threshold — false merges are worse than duplicates
_SEMANTIC_MERGE_THRESHOLD = 0.85


class ClaimDeduplicator:
    """
    Tracks canonical claims within a run for deduplication.

    Uses deterministic normalized matching first, then conservative semantic similarity.
    """

    def __init__(self) -> None:
        self._by_fingerprint: dict[str, Claim] = {}
        self._canonical_claims: list[Claim] = []
        self.duplicates_merged: int = 0

    @property
    def unique_count(self) -> int:
        return len(self._canonical_claims)

    def find_canonical(
        self,
        claim_text: str,
        *,
        temporal_scope: str | None = None,
        geographic_scope: str | None = None,
        fingerprint: str,
    ) -> Claim | None:
        """
        Find an existing canonical claim matching this proposition.

        Returns None if no confident match exists (keep separate).
        """
        if fingerprint in self._by_fingerprint:
            return self._by_fingerprint[fingerprint]

        norm_key = normalize_claim_for_dedup(
            claim_text, temporal_scope, geographic_scope
        )

        for existing in self._canonical_claims:
            if not _scopes_match(
                existing.temporal_scope,
                temporal_scope,
                existing.geographic_scope,
                geographic_scope,
            ):
                continue

            existing_norm = normalize_claim_for_dedup(
                existing.text,
                existing.temporal_scope,
                existing.geographic_scope,
            )
            if norm_key == existing_norm:
                return existing

            similarity = token_jaccard_similarity(claim_text, existing.text)
            if similarity >= _SEMANTIC_MERGE_THRESHOLD:
                logger.info(
                    "Semantic dedup merge (%.2f): '%s' -> '%s'",
                    similarity,
                    claim_text[:60],
                    existing.text[:60],
                )
                return existing

        return None

    def register(self, fingerprint: str, claim: Claim) -> None:
        """Register a new canonical claim."""
        self._by_fingerprint[fingerprint] = claim
        self._canonical_claims.append(claim)

    def record_merge(self) -> None:
        """Increment duplicate merge counter."""
        self.duplicates_merged += 1


def _scopes_match(
    temporal_a: str | None,
    temporal_b: str | None,
    geographic_a: str | None,
    geographic_b: str | None,
) -> bool:
    """Scopes must match exactly (including both None) to allow merge."""
    return (
        _normalize_scope(temporal_a) == _normalize_scope(temporal_b)
        and _normalize_scope(geographic_a) == _normalize_scope(geographic_b)
    )


def _normalize_scope(scope: str | None) -> str:
    if scope is None:
        return ""
    return scope.lower().strip()
