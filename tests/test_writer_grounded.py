"""Tests for evidence-grounded writer behavior (mocked LLM)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from domain.models import Evidence, EvidenceMatchType, EvidenceType, ExtractionMethod, Source, SourceQuality, SourceType
from services.writer_schemas import EvidenceGroundedWriterOutput
from state import Critique


def _evidence(text: str, eid: str = "E1", source_id: int = 1) -> Evidence:
    return Evidence(
        id=int(eid.replace("E", "")),
        source_id=source_id,
        research_run_id=1,
        exact_text=text,
        is_validated=True,
        match_type=EvidenceMatchType.EXACT,
        evidence_type=EvidenceType.DIRECT_QUOTE,
        extraction_method=ExtractionMethod.LLM,
        metadata={"display_id": eid, "source_url": "https://example.com", "relevance": "test"},
    )


def _source() -> Source:
    return Source(
        id=1,
        research_run_id=1,
        url="https://example.com",
        title="Example",
        content="Tokyo is the capital of Japan.",
        content_hash="abc",
        source_type=SourceType.OFFICIAL,
        source_quality=SourceQuality.OFFICIAL,
        metadata={"domain": "example.com"},
    )


class TestWriterGrounding:
    @pytest.mark.asyncio
    async def test_writer_uses_only_evidence_citations(self):
        from nodes.writer import writer_node
        from state import ResearchPlan

        grounded_output = EvidenceGroundedWriterOutput(
            content="Tokyo is the capital of Japan [E1].",
            evidence_ids_used=["E1"],
            factual_summary="Tokyo is the capital.",
        )

        mock_llm = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(return_value=grounded_output)
        mock_llm.with_structured_output = MagicMock(return_value=structured)

        state = {
            "user_query": "What is the capital of Japan?",
            "research_plan": ResearchPlan(query="What is the capital of Japan?", sub_queries=[]),
            "validated_evidence": [_evidence("Tokyo is the capital of Japan.")],
            "normalized_sources": [_source()],
            "critique": Critique(
                quality_score=0.85,
                is_sufficient=True,
                coverage="Good",
            ),
            "evidence_metrics": {"validated_count": 1},
            "research_run_id": 1,
            "is_run_persisted": False,
            "iteration_count": 0,
            "current_node": "writer",
        }

        with patch("nodes.writer.ChatAnthropic", return_value=mock_llm), \
             patch("nodes.writer.settings") as mock_settings:
            mock_settings.model_name = "test-model"
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.supabase_url = None
            mock_settings.supabase_key = None

            result = await writer_node(state)

        report = result["final_report"]
        assert report is not None
        assert "[E1]" in report.content
        assert "https://example.com" in report.sources
        assert report.confidence_level in ("high", "medium", "low")
        assert report.confidence <= 0.85  # Never max confidence from LLM
        assert "15 million" not in report.content  # No unsupported embellishment

    @pytest.mark.asyncio
    async def test_writer_fails_without_evidence(self):
        from nodes.writer import writer_node
        from state import ResearchPlan

        state = {
            "user_query": "Test",
            "research_plan": ResearchPlan(query="Test", sub_queries=[]),
            "validated_evidence": [],
            "normalized_sources": [],
            "current_node": "writer",
        }

        result = await writer_node(state)
        assert result.get("error") is not None
        assert "No validated evidence" in result["error"]
