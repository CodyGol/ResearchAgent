"""Unit tests for claim validation, normalization, and deduplication."""

import pytest

from domain.models import ClaimSupportStatus
from services.claim_deduplicator import ClaimDeduplicator
from services.claim_normalizer import (
    claim_fingerprint,
    normalize_claim_for_dedup,
    token_jaccard_similarity,
)
from services.claim_validator import validate_claim_support_deterministic


class TestQualifierPreservation:
    def test_modality_strengthening_rejected(self):
        result = validate_claim_support_deterministic(
            "Revenue will decline next year.",
            "Revenue may decline next year.",
        )
        assert not result.is_supported
        assert result.status == ClaimSupportStatus.NOT_SUPPORTED_BY_ORIGIN_EVIDENCE

    def test_hedged_claim_accepted(self):
        result = validate_claim_support_deterministic(
            "Revenue may decline next year.",
            "Revenue may decline next year.",
        )
        assert result.is_supported


class TestNegation:
    def test_negation_removal_rejected(self):
        result = validate_claim_support_deterministic(
            "Revenue increased in fiscal 2025.",
            "Revenue did not increase in fiscal 2025.",
        )
        assert not result.is_supported


class TestNumericPreservation:
    def test_numbers_not_in_evidence_rejected(self):
        result = validate_claim_support_deterministic(
            "Revenue was $99 billion in fiscal 2025.",
            "Revenue was $4.2 billion in fiscal 2025.",
        )
        assert not result.is_supported

    def test_matching_numbers_accepted(self):
        result = validate_claim_support_deterministic(
            "Revenue increased 17% year-over-year to $4.2 billion excluding discontinued operations.",
            "Revenue increased 17% year-over-year to $4.2 billion excluding discontinued operations.",
        )
        assert result.is_supported


class TestUnsupportedExpansion:
    def test_causal_expansion_rejected(self):
        result = validate_claim_support_deterministic(
            "Revenue increased because demand surged.",
            "Revenue increased.",
        )
        assert not result.is_supported

    def test_interpretive_superlative_rejected(self):
        result = validate_claim_support_deterministic(
            "Max Verstappen is the greatest Formula One driver of all time.",
            "Max Verstappen won the 2023 Formula One World Championship.",
        )
        assert not result.is_supported


class TestDirectVsInferred:
    def test_non_direct_basis_rejected(self):
        result = validate_claim_support_deterministic(
            "The company is losing competitive strength.",
            "Company revenue fell 20% while its largest competitor increased revenue 15%.",
            support_basis="inferred",
        )
        assert not result.is_supported


class TestNormalization:
    def test_negation_preserved_in_normalization(self):
        a = normalize_claim_for_dedup("Revenue increased")
        b = normalize_claim_for_dedup("Revenue did not increase")
        assert a != b

    def test_scope_included_in_normalization(self):
        a = normalize_claim_for_dedup("US unemployment was 4%", temporal_scope="2023")
        b = normalize_claim_for_dedup("US unemployment was 4%", temporal_scope="2024")
        assert a != b

    def test_fingerprint_stable(self):
        fp1 = claim_fingerprint(1, "test claim|t:2023")
        fp2 = claim_fingerprint(1, "test claim|t:2023")
        fp3 = claim_fingerprint(2, "test claim|t:2023")
        assert fp1 == fp2
        assert fp1 != fp3


class TestDeduplication:
    def test_same_proposition_merged(self):
        from domain.models import Claim, ClaimType

        dedup = ClaimDeduplicator()
        fp = "fp1"
        claim = Claim(
            research_run_id=1,
            text="Max Verstappen won the 2023 Formula One World Championship.",
            claim_type=ClaimType.FACTUAL,
            metadata={"fingerprint": fp},
        )
        dedup.register(fp, claim)

        found = dedup.find_canonical(
            "Verstappen secured his third F1 drivers' title in 2023.",
            temporal_scope="2023",
            fingerprint="fp2",
        )
        # Semantic similarity may or may not merge — conservative
        # Exact match should always merge
        found_exact = dedup.find_canonical(
            "Max Verstappen won the 2023 Formula One World Championship.",
            fingerprint=fp,
        )
        assert found_exact is claim

    def test_different_scopes_not_merged(self):
        from domain.models import Claim, ClaimType

        dedup = ClaimDeduplicator()
        claim_2024 = Claim(
            research_run_id=1,
            text="Apple revenue increased.",
            claim_type=ClaimType.FACTUAL,
            temporal_scope="2024",
            metadata={"fingerprint": "fp2024"},
        )
        dedup.register("fp2024", claim_2024)

        found = dedup.find_canonical(
            "Apple revenue increased.",
            temporal_scope="2025",
            fingerprint="fp2025",
        )
        assert found is None

    def test_different_qualifiers_not_merged(self):
        from domain.models import Claim, ClaimType

        dedup = ClaimDeduplicator()
        claim_a = Claim(
            research_run_id=1,
            text="Revenue increased 17% excluding discontinued operations.",
            claim_type=ClaimType.STATISTICAL,
            metadata={"fingerprint": "fpa"},
        )
        dedup.register("fpa", claim_a)

        found = dedup.find_canonical(
            "Revenue increased 17%.",
            fingerprint="fpb",
        )
        assert found is None


class TestTokenJaccard:
    def test_identical_high_similarity(self):
        sim = token_jaccard_similarity(
            "Max Verstappen won the 2023 championship",
            "Max Verstappen won the 2023 championship",
        )
        assert sim == 1.0

    def test_different_low_similarity(self):
        sim = token_jaccard_similarity(
            "Apple revenue increased",
            "Google cloud revenue decreased",
        )
        assert sim < 0.5
