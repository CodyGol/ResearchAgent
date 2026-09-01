"""Tests for the claim extraction pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.models import ClaimType, Evidence, EvidenceType, ExtractionMethod
from services.claim_pipeline import process_evidence_for_claims
from services.claim_schemas import CandidateClaimItem, ClaimExtractionOutput


F1_QUESTION = "Who won the 2023 Formula 1 World Championship?"

F1_EVIDENCE_TEXT = (
    "Max Verstappen secured his third Formula 1 world championship "
    "during the Qatar Sprint in October 2023."
)


def _make_evidence(
    text: str = F1_EVIDENCE_TEXT,
    evidence_id: int = 1,
) -> Evidence:
    return Evidence(
        id=evidence_id,
        source_id=1,
        research_run_id=1,
        exact_text=text,
        evidence_type=EvidenceType.DIRECT_QUOTE,
        extraction_method=ExtractionMethod.LLM,
        is_validated=True,
    )


def _mock_llm(candidates: list[CandidateClaimItem]) -> MagicMock:
    output = ClaimExtractionOutput(claims=candidates)
    mock = MagicMock()
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=output)
    mock.with_structured_output = MagicMock(return_value=structured)
    return mock


class TestClaimPipelineAcceptance:
    @pytest.mark.asyncio
    async def test_direct_claims_accepted(self):
        candidates = [
            CandidateClaimItem(
                claim_text="Max Verstappen won the 2023 Formula 1 World Championship.",
                claim_type="factual",
                importance="high",
                temporal_scope="2023",
                support_basis="direct",
            ),
            CandidateClaimItem(
                claim_text="The 2023 title was Max Verstappen's third Formula 1 World Championship.",
                claim_type="factual",
                importance="high",
                temporal_scope="2023",
                support_basis="direct",
            ),
            CandidateClaimItem(
                claim_text="Verstappen secured the title during the Qatar Sprint.",
                claim_type="factual",
                importance="medium",
                support_basis="direct",
            ),
        ]
        llm = _mock_llm(candidates)
        evidence = _make_evidence()

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            claims, material, relations, metrics = await process_evidence_for_claims(
                [evidence],
                F1_QUESTION,
                research_run_id=1,
                llm=llm,
                use_llm_validation=False,
            )

        assert metrics.claims_accepted == 3
        assert metrics.claims_rejected == 0
        assert len(claims) == 3
        assert len(relations) == 3
        assert all(r.claim_id is not None for r in relations)
        assert all(r.evidence_id == 1 for r in relations)


class TestUnsupportedClaimRejected:
    @pytest.mark.asyncio
    async def test_greatest_driver_rejected(self):
        candidates = [
            CandidateClaimItem(
                claim_text="Max Verstappen is the greatest Formula One driver of all time.",
                claim_type="opinion",
                importance="low",
                support_basis="direct",
            ),
        ]
        llm = _mock_llm(candidates)
        evidence = _make_evidence(
            "Max Verstappen won the 2023 Formula One World Championship."
        )

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            claims, material, relations, metrics = await process_evidence_for_claims(
                [evidence],
                F1_QUESTION,
                research_run_id=1,
                llm=llm,
                use_llm_validation=False,
            )

        assert metrics.claims_rejected_unsupported == 1
        assert len(claims) == 0
        assert len(relations) == 0


class TestInferredClaimRejected:
    @pytest.mark.asyncio
    async def test_inferred_basis_rejected(self):
        candidates = [
            CandidateClaimItem(
                claim_text="The company is losing competitive strength.",
                claim_type="analytical",
                support_basis="inferred",
            ),
        ]
        llm = _mock_llm(candidates)
        evidence = _make_evidence(
            "Company revenue fell 20% while its largest competitor increased revenue 15%."
        )

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            _, _, _, metrics = await process_evidence_for_claims(
                [evidence],
                "How is the company performing?",
                research_run_id=1,
                llm=llm,
                use_llm_validation=False,
            )

        assert metrics.claims_rejected_non_direct == 1


class TestMultiEvidenceDedup:
    @pytest.mark.asyncio
    async def test_same_proposition_deduplicated(self):
        e1_candidates = [
            CandidateClaimItem(
                claim_text="Max Verstappen won the 2023 Formula One World Championship.",
                claim_type="factual",
                importance="high",
                temporal_scope="2023",
                support_basis="direct",
            ),
        ]
        e2_candidates = [
            CandidateClaimItem(
                claim_text="Max Verstappen won the 2023 Formula One World Championship.",
                claim_type="factual",
                importance="high",
                temporal_scope="2023",
                support_basis="direct",
            ),
        ]

        call_count = 0

        def make_llm(candidates):
            mock = MagicMock()
            structured = MagicMock()

            async def ainvoke(messages):
                return ClaimExtractionOutput(claims=candidates)

            structured.ainvoke = ainvoke
            mock.with_structured_output = MagicMock(return_value=structured)
            return mock

        llm_e1 = make_llm(e1_candidates)
        llm_e2 = make_llm(e2_candidates)

        evidence_items = [
            _make_evidence(
                "Max Verstappen won the 2023 Formula One World Championship.",
                evidence_id=1,
            ),
            _make_evidence(
                "Verstappen secured his third F1 drivers' title in 2023.",
                evidence_id=2,
            ),
        ]

        # Use a single mock that returns different outputs per call
        outputs = [e1_candidates, e2_candidates]
        mock = MagicMock()
        structured = MagicMock()

        async def ainvoke_seq(messages):
            nonlocal call_count
            idx = min(call_count, len(outputs) - 1)
            call_count += 1
            return ClaimExtractionOutput(claims=outputs[idx])

        structured.ainvoke = ainvoke_seq
        mock.with_structured_output = MagicMock(return_value=structured)

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            claims, material, relations, metrics = await process_evidence_for_claims(
                evidence_items,
                F1_QUESTION,
                research_run_id=1,
                llm=mock,
                use_llm_validation=False,
            )

        assert metrics.duplicate_claims_merged >= 1 or len(claims) == 1
        assert len(relations) == 2
        if len(claims) == 1:
            assert relations[0].claim_id == relations[1].claim_id


class TestInvalidLLMOutput:
    @pytest.mark.asyncio
    async def test_extraction_failure_does_not_crash(self):
        mock = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(side_effect=ValueError("Invalid JSON"))
        mock.with_structured_output = MagicMock(return_value=structured)

        evidence = _make_evidence()

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            claims, material, relations, metrics = await process_evidence_for_claims(
                [evidence],
                F1_QUESTION,
                research_run_id=1,
                llm=mock,
                use_llm_validation=False,
            )

        assert metrics.extraction_failures == 1
        assert len(claims) == 0


class TestQualifierInPipeline:
    @pytest.mark.asyncio
    async def test_modality_strengthening_rejected_in_pipeline(self):
        candidates = [
            CandidateClaimItem(
                claim_text="Revenue will decline next year.",
                claim_type="predictive",
                support_basis="direct",
            ),
        ]
        llm = _mock_llm(candidates)
        evidence = _make_evidence("Revenue may decline next year.")

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            _, _, _, metrics = await process_evidence_for_claims(
                [evidence],
                "What is the revenue outlook?",
                research_run_id=1,
                llm=llm,
                use_llm_validation=False,
            )

        assert metrics.claims_rejected_unsupported == 1


class TestProvenance:
    @pytest.mark.asyncio
    async def test_every_claim_has_evidence_link(self):
        candidates = [
            CandidateClaimItem(
                claim_text="Max Verstappen won the 2023 Formula 1 World Championship.",
                claim_type="factual",
                importance="high",
                support_basis="direct",
            ),
        ]
        llm = _mock_llm(candidates)
        evidence = _make_evidence(
            "Max Verstappen won the 2023 Formula One World Championship."
        )

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            claims, _, relations, _ = await process_evidence_for_claims(
                [evidence],
                F1_QUESTION,
                research_run_id=1,
                llm=llm,
                use_llm_validation=False,
            )

        assert len(claims) >= 1
        claim_ids = {c.id for c in claims}
        for rel in relations:
            assert rel.claim_id in claim_ids
            assert rel.evidence_id == 1
