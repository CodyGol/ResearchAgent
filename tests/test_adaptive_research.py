"""Tests for Phase 2B.5 adaptive research optimizations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.models import Evidence, EvidenceType, ExtractionMethod, Source, SourceQuality, SourceType
from services.answer_confidence import compute_confidence_assessment
from services.claim_pipeline import process_evidence_for_claims
from services.claim_schemas import CandidateClaimItem, ClaimExtractionOutput
from services.claim_validator import validate_claims_batch
from services.query_router import QueryComplexity
from services.research_sufficiency import check_research_sufficiency


def _make_evidence(text: str, eid: int = 1, source_id: int = 1) -> Evidence:
    return Evidence(
        id=eid,
        source_id=source_id,
        research_run_id=1,
        exact_text=text,
        evidence_type=EvidenceType.DIRECT_QUOTE,
        extraction_method=ExtractionMethod.LLM,
        is_validated=True,
    )


def _make_source(source_id: int = 1, quality: SourceQuality = SourceQuality.OFFICIAL) -> Source:
    return Source(
        id=source_id,
        research_run_id=1,
        url="https://example.gov/fact",
        title="Official",
        content="",
        content_hash="abc",
        source_quality=quality,
        source_type=SourceType.OFFICIAL,
    )


class TestResearchSufficiency:
    def test_simple_question_sufficient_with_authoritative_evidence(self):
        evidence = [_make_evidence("Tokyo is the capital of Japan.")]
        sources = [_make_source()]
        from services.query_router import BUDGETS

        result = check_research_sufficiency(
            "What is the capital of Japan?",
            evidence,
            sources,
            complexity=QueryComplexity.SIMPLE,
            budget=BUDGETS[QueryComplexity.SIMPLE],
        )
        assert result.is_sufficient

    def test_insufficient_without_evidence(self):
        from services.query_router import BUDGETS

        result = check_research_sufficiency(
            "What is the capital of Japan?",
            [],
            [],
            complexity=QueryComplexity.SIMPLE,
            budget=BUDGETS[QueryComplexity.SIMPLE],
        )
        assert not result.is_sufficient

    def test_conflicts_prevent_short_circuit(self):
        evidence = [_make_evidence("Tokyo is the capital of Japan.")]
        sources = [_make_source()]
        from services.query_router import BUDGETS

        result = check_research_sufficiency(
            "What is the capital of Japan?",
            evidence,
            sources,
            complexity=QueryComplexity.SIMPLE,
            budget=BUDGETS[QueryComplexity.SIMPLE],
            potential_conflicts=["Conflicting capitals reported"],
        )
        assert not result.is_sufficient


class TestDeterministicRejectionSkipsLLM:
    @pytest.mark.asyncio
    async def test_unsupported_claim_no_llm_call(self):
        candidates = [
            CandidateClaimItem(
                claim_text="Max Verstappen is the greatest Formula One driver of all time.",
                support_basis="direct",
            ),
        ]
        output = ClaimExtractionOutput(claims=candidates)
        mock = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(return_value=output)
        mock.with_structured_output = MagicMock(return_value=structured)

        batch_mock = AsyncMock()
        with patch(
            "services.claim_pipeline.validate_claims_batch", batch_mock
        ), patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            _, _, _, metrics = await process_evidence_for_claims(
                [_make_evidence("Max Verstappen won the 2023 championship.")],
                "Who won the 2023 F1 World Championship?",
                research_run_id=1,
                llm=mock,
                use_llm_validation=True,
                claim_depth="minimal",
            )

        batch_mock.assert_not_called()
        assert metrics.claims_rejected_deterministic >= 1


class TestRelevanceBeforeValidation:
    @pytest.mark.asyncio
    async def test_irrelevant_claim_rejected_without_llm(self):
        candidates = [
            CandidateClaimItem(
                claim_text="Max Verstappen finished in second place at the flag in Jeddah.",
                importance="low",
                support_basis="direct",
            ),
        ]
        output = ClaimExtractionOutput(claims=candidates)
        mock = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(return_value=output)
        mock.with_structured_output = MagicMock(return_value=structured)

        batch_mock = AsyncMock()
        with patch(
            "services.claim_pipeline.validate_claims_batch", batch_mock
        ), patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            _, _, _, metrics = await process_evidence_for_claims(
                [_make_evidence(
                    "Max Verstappen finished in second place at the flag in Jeddah."
                )],
                "Who won the 2023 F1 World Championship?",
                research_run_id=1,
                llm=mock,
                use_llm_validation=True,
                claim_depth="minimal",
            )

        batch_mock.assert_not_called()
        assert metrics.claims_rejected_relevance >= 1


class TestBatchValidation:
    @pytest.mark.asyncio
    async def test_batch_returns_results_for_all_claims(self):
        mock = MagicMock()
        from services.claim_schemas import ClaimBatchValidationItem, ClaimBatchValidationOutput

        batch_output = ClaimBatchValidationOutput(
            results=[
                ClaimBatchValidationItem(claim_index=0, is_supported=True, reason="ok"),
                ClaimBatchValidationItem(claim_index=1, is_supported=False, reason="no"),
            ]
        )
        structured = MagicMock()
        structured.ainvoke = AsyncMock(return_value=batch_output)
        mock.with_structured_output = MagicMock(return_value=structured)

        results = await validate_claims_batch(
            [(0, "Tokyo is the capital of Japan."), (1, "Tokyo has 14 million people.")],
            "Tokyo is the capital of Japan.",
            llm=mock,
        )
        assert 0 in results and results[0].is_supported
        assert 1 in results and not results[1].is_supported

    @pytest.mark.asyncio
    async def test_missing_batch_result_rejected(self):
        mock = MagicMock()
        from services.claim_schemas import ClaimBatchValidationOutput

        batch_output = ClaimBatchValidationOutput(results=[])
        structured = MagicMock()
        structured.ainvoke = AsyncMock(return_value=batch_output)
        mock.with_structured_output = MagicMock(return_value=structured)

        results = await validate_claims_batch(
            [(0, "Tokyo is the capital of Japan.")],
            "Tokyo is the capital of Japan.",
            llm=mock,
        )
        assert not results[0].is_supported
        assert "Missing" in results[0].reason


class TestAnswerConfidenceSeparation:
    def test_simple_question_high_answer_low_completeness_ok(self):
        evidence = [_make_evidence("Tokyo is the capital of Japan.")]
        sources = [_make_source()]

        assessment = compute_confidence_assessment(
            "What is the capital of Japan?",
            evidence,
            sources,
            complexity=QueryComplexity.SIMPLE,
        )
        assert assessment.answer_confidence.value in ("high", "medium")
        # Completeness may be medium for simple targeted research — that's OK

    def test_answer_and_completeness_are_separate_fields(self):
        evidence = [_make_evidence("Tokyo is the capital of Japan.")]
        sources = [_make_source()]

        assessment = compute_confidence_assessment(
            "What is the capital of Japan?",
            evidence,
            sources,
            complexity=QueryComplexity.SIMPLE,
        )
        assert hasattr(assessment, "answer_confidence")
        assert hasattr(assessment, "research_completeness")
