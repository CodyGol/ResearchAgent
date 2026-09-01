"""Tests for the evidence extraction pipeline."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.models import EvidenceMatchType, EvidenceType, Source, SourceQuality, SourceType
from services.evidence_pipeline import process_sources_for_evidence
from services.evidence_schemas import CandidateEvidenceItem, EvidenceExtractionOutput


FIXTURE_CONTENT = (
    "Company X reported revenue of $4.2 billion in fiscal 2025, "
    "up 17% from the previous year."
)
FIXTURE_QUESTION = "What was Company X's revenue in 2025?"


def _make_source(
    content: str = FIXTURE_CONTENT,
    source_id: int = 1,
    url: str = "https://example.com/report",
) -> Source:
    return Source(
        id=source_id,
        research_run_id=1,
        url=url,
        title="Company X Annual Report",
        content=content,
        content_hash="abc123",
        source_type=SourceType.WEB,
        source_quality=SourceQuality.GENERAL_SECONDARY,
    )


def _mock_llm(candidates: list[CandidateEvidenceItem]) -> MagicMock:
    """Create a mock LLM that returns structured evidence extraction output."""
    output = EvidenceExtractionOutput(evidence=candidates)
    mock = MagicMock()
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=output)
    mock.with_structured_output = MagicMock(return_value=structured)
    return mock


class TestValidEvidencePersisted:
    @pytest.mark.asyncio
    async def test_valid_evidence_accepted(self):
        candidates = [
            CandidateEvidenceItem(
                text="Company X reported revenue of $4.2 billion in fiscal 2025",
                evidence_type="statistic",
                relevance="Directly answers the revenue question",
                locator="sentence 1",
            )
        ]
        llm = _mock_llm(candidates)
        source = _make_source()

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            evidence, metrics = await process_sources_for_evidence(
                [source], FIXTURE_QUESTION, research_run_id=1, llm=llm
            )

        assert len(evidence) == 1
        assert evidence[0].is_validated is True
        assert evidence[0].match_type == EvidenceMatchType.EXACT
        assert evidence[0].source_id == 1
        assert evidence[0].research_run_id == 1
        assert "$4.2 billion" in evidence[0].exact_text
        assert metrics.validated_count == 1
        assert metrics.rejected_count == 0


class TestFabricatedEvidenceRejected:
    @pytest.mark.asyncio
    async def test_fabricated_evidence_rejected(self):
        candidates = [
            CandidateEvidenceItem(
                text="Company X generated $9.9 trillion in revenue",
                evidence_type="statistic",
                relevance="Revenue figure",
            )
        ]
        llm = _mock_llm(candidates)

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            evidence, metrics = await process_sources_for_evidence(
                [_make_source()], FIXTURE_QUESTION, research_run_id=1, llm=llm
            )

        assert len(evidence) == 0
        assert metrics.rejected_count == 1
        assert metrics.validation_failures == 1


class TestNormalizedEvidence:
    @pytest.mark.asyncio
    async def test_whitespace_variation_accepted(self):
        candidates = [
            CandidateEvidenceItem(
                text="Company X reported revenue of $4.2 billion in fiscal 2025, up 17% from the previous year.",
                evidence_type="statistic",
                relevance="Revenue and growth",
            )
        ]
        llm = _mock_llm(candidates)

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            evidence, metrics = await process_sources_for_evidence(
                [_make_source()], FIXTURE_QUESTION, research_run_id=1, llm=llm
            )

        assert len(evidence) == 1
        assert evidence[0].match_type in (
            EvidenceMatchType.EXACT,
            EvidenceMatchType.NORMALIZED,
        )


class TestMultipleEvidenceItems:
    @pytest.mark.asyncio
    async def test_multiple_items_from_one_source(self):
        candidates = [
            CandidateEvidenceItem(
                text="Company X reported revenue of $4.2 billion in fiscal 2025",
                evidence_type="statistic",
                relevance="Revenue figure",
            ),
            CandidateEvidenceItem(
                text="up 17% from the previous year",
                evidence_type="statistic",
                relevance="Growth rate",
            ),
        ]
        llm = _mock_llm(candidates)

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            evidence, metrics = await process_sources_for_evidence(
                [_make_source()], FIXTURE_QUESTION, research_run_id=1, llm=llm
            )

        assert len(evidence) == 2
        assert metrics.validated_count == 2


class TestDuplicateControl:
    @pytest.mark.asyncio
    async def test_duplicate_evidence_deduplicated(self):
        candidates = [
            CandidateEvidenceItem(
                text="Company X reported revenue of $4.2 billion in fiscal 2025",
                evidence_type="statistic",
                relevance="Revenue",
            ),
            CandidateEvidenceItem(
                text="Company X reported revenue of $4.2 billion in fiscal 2025",
                evidence_type="statistic",
                relevance="Same revenue again",
            ),
        ]
        llm = _mock_llm(candidates)

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            evidence, metrics = await process_sources_for_evidence(
                [_make_source()], FIXTURE_QUESTION, research_run_id=1, llm=llm
            )

        assert len(evidence) == 1
        assert metrics.duplicate_count == 1


class TestEmptySourceContent:
    @pytest.mark.asyncio
    async def test_empty_content_skipped(self):
        llm = _mock_llm([])

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            evidence, metrics = await process_sources_for_evidence(
                [_make_source(content="")], FIXTURE_QUESTION, research_run_id=1, llm=llm
            )

        assert len(evidence) == 0
        assert metrics.sources_processed == 1
        assert any(f["failure_type"] == "empty_content" for f in metrics.failures)


class TestInvalidLLMOutput:
    @pytest.mark.asyncio
    async def test_llm_failure_does_not_crash_run(self):
        llm = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(side_effect=ValueError("Invalid JSON"))
        llm.with_structured_output = MagicMock(return_value=structured)
        llm.ainvoke = AsyncMock(side_effect=ValueError("Invalid JSON"))

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            evidence, metrics = await process_sources_for_evidence(
                [_make_source()], FIXTURE_QUESTION, research_run_id=1, llm=llm
            )

        assert len(evidence) == 0
        assert metrics.extraction_failures == 1


class TestOneFailedSourceDoesNotCrashRun:
    @pytest.mark.asyncio
    async def test_partial_source_failure(self):
        good_candidates = [
            CandidateEvidenceItem(
                text="Company X reported revenue of $4.2 billion in fiscal 2025",
                evidence_type="statistic",
                relevance="Revenue",
            )
        ]
        good_llm = _mock_llm(good_candidates)
        bad_llm = MagicMock()
        bad_structured = MagicMock()
        bad_structured.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))
        bad_llm.with_structured_output = MagicMock(return_value=bad_structured)
        bad_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))

        sources = [
            _make_source(source_id=1, url="https://example.com/good"),
            _make_source(source_id=2, url="https://example.com/bad"),
        ]

        call_count = 0

        async def side_effect(source, question, llm=None):
            nonlocal call_count
            call_count += 1
            if source.url.endswith("/bad"):
                raise RuntimeError("API error")
            return good_candidates

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False), \
             patch("services.evidence_pipeline.extract_candidates_from_source", side_effect=side_effect):
            evidence, metrics = await process_sources_for_evidence(
                sources, FIXTURE_QUESTION, research_run_id=1
            )

        assert len(evidence) == 1
        assert metrics.extraction_failures == 1
        assert metrics.sources_processed == 2


class TestEvidenceRelationships:
    @pytest.mark.asyncio
    async def test_evidence_linked_to_source_and_run(self):
        candidates = [
            CandidateEvidenceItem(
                text="Company X reported revenue of $4.2 billion in fiscal 2025",
                evidence_type="statistic",
                relevance="Revenue",
            )
        ]
        llm = _mock_llm(candidates)
        source = _make_source(source_id=42)

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            evidence, _ = await process_sources_for_evidence(
                [source], FIXTURE_QUESTION, research_run_id=99, llm=llm
            )

        assert evidence[0].source_id == 42
        assert evidence[0].research_run_id == 99
        assert evidence[0].metadata["content_scope"] == "search_snippet"
        assert evidence[0].metadata["relevance"] == "Revenue"


class TestPersistence:
    @pytest.mark.asyncio
    async def test_valid_evidence_persisted_via_repository(self):
        candidates = [
            CandidateEvidenceItem(
                text="Company X reported revenue of $4.2 billion in fiscal 2025",
                evidence_type="statistic",
                relevance="Revenue",
            )
        ]
        llm = _mock_llm(candidates)
        saved_evidence = []

        async def mock_save(evidence_list):
            for i, ev in enumerate(evidence_list):
                saved_evidence.append(ev.model_copy(update={"id": i + 100}))
            return saved_evidence

        mock_repo = MagicMock()
        mock_repo.save_evidence = AsyncMock(side_effect=mock_save)

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=True), \
             patch("db.evidence_repositories.get_evidence_repo", return_value=mock_repo):
            evidence, metrics = await process_sources_for_evidence(
                [_make_source()],
                FIXTURE_QUESTION,
                research_run_id=1,
                is_persisted=True,
                llm=llm,
            )

        assert len(evidence) == 1
        assert evidence[0].id == 100
        mock_repo.save_evidence.assert_called_once()


class TestEvidenceExtractorNode:
    @pytest.mark.asyncio
    async def test_node_integration(self):
        from nodes.evidence_extractor import evidence_extractor_node

        source = _make_source()
        state = {
            "user_query": FIXTURE_QUESTION,
            "research_run_id": 1,
            "is_run_persisted": False,
            "normalized_sources": [source],
            "current_node": "evidence_extractor",
        }

        candidates = [
            CandidateEvidenceItem(
                text="Company X reported revenue of $4.2 billion in fiscal 2025",
                evidence_type="statistic",
                relevance="Revenue",
            )
        ]
        llm = _mock_llm(candidates)

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False), \
             patch("nodes.evidence_extractor.process_sources_for_evidence") as mock_process:
            from services.evidence_pipeline import EvidenceExtractionMetrics
            from domain.models import Evidence, ExtractionMethod

            mock_evidence = Evidence(
                source_id=1,
                research_run_id=1,
                exact_text="Company X reported revenue of $4.2 billion in fiscal 2025",
                is_validated=True,
                match_type=EvidenceMatchType.EXACT,
                extraction_method=ExtractionMethod.LLM,
            )
            mock_process.return_value = (
                [mock_evidence],
                EvidenceExtractionMetrics(validated_count=1, sources_processed=1),
            )

            result = await evidence_extractor_node(state)

        assert result["current_node"] == "claim_extractor"
        assert len(result["validated_evidence"]) == 1
        assert result["evidence_metrics"]["validated_count"] == 1
