"""Tests for claim relevance filtering."""

import pytest

from services.claim_relevance import (
    ClaimRelevance,
    assess_claim_relevance,
    is_material_claim,
    should_validate_expensively,
)
from services.claim_schemas import CandidateClaimItem


F1_QUESTION = "Who won the 2023 Formula 1 World Championship?"


class TestRelevanceAssessment:
    def test_championship_claim_is_critical(self):
        candidate = CandidateClaimItem(
            claim_text="Max Verstappen won the 2023 Formula 1 World Championship.",
            importance="high",
            support_basis="direct",
        )
        relevance = assess_claim_relevance(
            candidate, F1_QUESTION, claim_depth="minimal"
        )
        assert relevance == ClaimRelevance.CRITICAL

    def test_jeddah_finish_is_irrelevant(self):
        candidate = CandidateClaimItem(
            claim_text="Max Verstappen finished in second place at the flag in Jeddah.",
            importance="low",
            support_basis="direct",
        )
        relevance = assess_claim_relevance(
            candidate, F1_QUESTION, claim_depth="minimal"
        )
        assert relevance == ClaimRelevance.IRRELEVANT

    def test_nationality_is_irrelevant_for_championship(self):
        candidate = CandidateClaimItem(
            claim_text="Max Verstappen's nationality is NED (Netherlands).",
            importance="low",
            support_basis="direct",
        )
        relevance = assess_claim_relevance(
            candidate, F1_QUESTION, claim_depth="minimal"
        )
        assert relevance == ClaimRelevance.IRRELEVANT


class TestValidationOrdering:
    def test_irrelevant_skips_expensive_validation(self):
        assert not should_validate_expensively(ClaimRelevance.IRRELEVANT, "minimal")
        assert not should_validate_expensively(ClaimRelevance.CONTEXTUAL, "minimal")

    def test_critical_gets_validation(self):
        assert should_validate_expensively(ClaimRelevance.CRITICAL, "minimal")


class TestMaterialClaims:
    def test_critical_is_material(self):
        assert is_material_claim(ClaimRelevance.CRITICAL)

    def test_irrelevant_not_material(self):
        assert not is_material_claim(ClaimRelevance.IRRELEVANT)
