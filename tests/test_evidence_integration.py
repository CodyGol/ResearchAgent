"""End-to-end integration test for evidence extraction with fixture source."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from domain.models import EvidenceMatchType, Source, SourceQuality, SourceType
from services.evidence_pipeline import process_sources_for_evidence
from services.evidence_schemas import CandidateEvidenceItem, EvidenceExtractionOutput


FIXTURE_SOURCE_TEXT = (
    "Company X reported revenue of $4.2 billion in fiscal 2025, "
    "up 17% from the previous year."
)
FIXTURE_QUESTION = "What was Company X's revenue in 2025?"


def _mock_llm(candidates: list[CandidateEvidenceItem]) -> MagicMock:
    output = EvidenceExtractionOutput(evidence=candidates)
    mock = MagicMock()
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=output)
    mock.with_structured_output = MagicMock(return_value=structured)
    return mock


@pytest.fixture
def fixture_source() -> Source:
    return Source(
        id=1,
        research_run_id=10,
        url="https://example.com/company-x-report",
        title="Company X FY2025 Results",
        content=FIXTURE_SOURCE_TEXT,
        content_hash="fixture_hash",
        source_type=SourceType.OFFICIAL,
        source_quality=SourceQuality.OFFICIAL,
    )


class TestEvidenceE2E:
    @pytest.mark.asyncio
    async def test_fixture_source_valid_evidence_accepted(self, fixture_source):
        """Valid candidate → validator accepts → evidence returned with correct links."""
        valid_candidate = CandidateEvidenceItem(
            text="Company X reported revenue of $4.2 billion in fiscal 2025",
            evidence_type="statistic",
            relevance="Directly states Company X revenue for fiscal 2025",
            locator="sentence 1",
            context="Part of annual results announcement",
        )
        llm = _mock_llm([valid_candidate])

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            evidence, metrics = await process_sources_for_evidence(
                [fixture_source],
                FIXTURE_QUESTION,
                research_run_id=10,
                llm=llm,
            )

        assert metrics.candidate_count == 1
        assert metrics.validated_count == 1
        assert metrics.rejected_count == 0
        assert len(evidence) == 1

        ev = evidence[0]
        assert ev.is_validated is True
        assert ev.match_type in (EvidenceMatchType.EXACT, EvidenceMatchType.NORMALIZED)
        assert ev.source_id == 1
        assert ev.research_run_id == 10
        assert "$4.2 billion" in ev.exact_text
        assert ev.metadata["content_scope"] == "search_snippet"
        assert "revenue" in ev.metadata["relevance"].lower()

    @pytest.mark.asyncio
    async def test_fixture_fabricated_evidence_rejected(self, fixture_source):
        """Fabricated candidate → validator rejects → not persisted."""
        fabricated = CandidateEvidenceItem(
            text="Company X generated $9.9 trillion in revenue",
            evidence_type="statistic",
            relevance="Revenue figure",
        )
        llm = _mock_llm([fabricated])

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            evidence, metrics = await process_sources_for_evidence(
                [fixture_source],
                FIXTURE_QUESTION,
                research_run_id=10,
                llm=llm,
            )

        assert len(evidence) == 0
        assert metrics.rejected_count == 1
        assert metrics.validation_failures == 1
        assert any(
            f["failure_type"] == "validation_rejected" for f in metrics.failures
        )

    @pytest.mark.asyncio
    async def test_mixed_valid_and_fabricated(self, fixture_source):
        """Only valid evidence survives when LLM returns mixed candidates."""
        candidates = [
            CandidateEvidenceItem(
                text="Company X reported revenue of $4.2 billion in fiscal 2025",
                evidence_type="statistic",
                relevance="Valid revenue",
            ),
            CandidateEvidenceItem(
                text="Company X generated $9.9 trillion in revenue",
                evidence_type="statistic",
                relevance="Fabricated",
            ),
        ]
        llm = _mock_llm(candidates)

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            evidence, metrics = await process_sources_for_evidence(
                [fixture_source],
                FIXTURE_QUESTION,
                research_run_id=10,
                llm=llm,
            )

        assert len(evidence) == 1
        assert metrics.validated_count == 1
        assert metrics.rejected_count == 1
