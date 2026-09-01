"""Tests for Phase 2B.6 fast path."""

import pytest

from domain.models import Evidence, EvidenceType, ExtractionMethod, Source, SourceQuality, SourceType
from services.fact_sufficiency import check_fact_sufficiency, detect_conflicting_values
from services.fact_target import extract_fact_target, is_causal_or_analytical
from services.query_router import ExecutionRoute, classify_query
from services.source_authority import is_source_adequate_for_domain
from services.fact_target import FactDomain


class TestSimpleFactIdentification:
    def test_japan_capital_is_simple_fact(self):
        result = classify_query("What is the capital of Japan?")
        assert result.route == ExecutionRoute.SIMPLE_FACT
        assert result.direct_answer_expected is True
        assert result.fact_target is not None
        assert result.fact_target.attribute == "capital"

    def test_f1_winner_is_simple_fact(self):
        result = classify_query("Who won the 2023 F1 World Championship?")
        assert result.route == ExecutionRoute.SIMPLE_FACT
        assert result.fact_target.attribute == "winner"

    def test_apple_revenue_is_simple_fact(self):
        result = classify_query("What was Apple's revenue in fiscal 2025?")
        assert result.route == ExecutionRoute.SIMPLE_FACT
        assert result.fact_target.domain == FactDomain.FINANCIAL

    def test_why_question_not_simple_fact(self):
        result = classify_query("Why did Apple revenue increase?")
        assert result.route != ExecutionRoute.SIMPLE_FACT
        assert is_causal_or_analytical("Why did Apple revenue increase?")

    def test_deep_question_unchanged(self):
        result = classify_query("Should Company X acquire Company Y?")
        assert result.route == ExecutionRoute.DEEP


class TestFactTargetExtraction:
    def test_capital_target(self):
        target = extract_fact_target("What is the capital of Japan?")
        assert target is not None
        assert target.entity.lower().startswith("japan")
        assert target.attribute == "capital"

    def test_revenue_target_preserves_fiscal_scope(self):
        target = extract_fact_target("What was Apple's revenue in fiscal 2025?")
        assert target is not None
        assert "apple" in target.entity.lower()
        assert "fiscal" in (target.temporal_scope or "").lower()

    def test_f1_winner_target(self):
        target = extract_fact_target("Who won the 2023 F1 World Championship?")
        assert target is not None
        assert target.temporal_scope == "2023"


class TestFactSufficiency:
    def _source(self, quality=SourceQuality.REPUTABLE_SECONDARY) -> Source:
        return Source(
            id=1,
            research_run_id=1,
            url="https://britannica.com/place/Tokyo",
            title="Tokyo",
            content="Tokyo is the capital of Japan.",
            content_hash="abc",
            source_quality=quality,
        )

    def _evidence(self, text: str) -> Evidence:
        return Evidence(
            id=1,
            source_id=1,
            research_run_id=1,
            exact_text=text,
            evidence_type=EvidenceType.DIRECT_QUOTE,
            extraction_method=ExtractionMethod.LLM,
            is_validated=True,
        )

    def test_decisive_evidence_sufficient(self):
        target = extract_fact_target("What is the capital of Japan?")
        source = self._source()
        evidence = self._evidence("Tokyo is the capital of Japan.")
        result = check_fact_sufficiency(target, evidence, source)
        assert result.is_sufficient

    def test_geographic_reputable_secondary_adequate(self):
        source = self._source(SourceQuality.REPUTABLE_SECONDARY)
        assert is_source_adequate_for_domain(source, FactDomain.GEOGRAPHIC)

    def test_temporal_mismatch_blocks(self):
        target = extract_fact_target("Who won the 2023 F1 World Championship?")
        source = self._source()
        evidence = self._evidence("Lewis Hamilton won the 2022 championship.")
        result = check_fact_sufficiency(target, evidence, source)
        assert not result.is_sufficient

    def test_conflict_detection(self):
        target = extract_fact_target("What is the capital of Japan?")
        ev1 = self._evidence("Tokyo is the capital of Japan.")
        ev2 = self._evidence("Osaka is the capital of Japan.")
        conflict = detect_conflicting_values([ev1, ev2], target)
        assert conflict is not None


class TestFastPathServices:
    @pytest.mark.asyncio
    async def test_core_claim_deterministic_capital(self):
        from services.core_claim import generate_core_claim

        target = extract_fact_target("What is the capital of Japan?")
        evidence = Evidence(
            id=1,
            source_id=1,
            research_run_id=1,
            exact_text="Tokyo is the capital of Japan.",
            evidence_type=EvidenceType.DIRECT_QUOTE,
            extraction_method=ExtractionMethod.LLM,
            is_validated=True,
        )
        claim = await generate_core_claim(
            target, evidence, 1, llm=None, use_llm=False
        )
        assert claim is not None
        assert "tokyo" in claim.text.lower()
        assert claim.metadata.get("is_core_claim")

    def test_fast_writer_concise(self):
        from services.core_claim import build_core_claim_from_value
        from services.fast_writer import build_fast_answer
        from services.fact_value import extract_fact_value

        target = extract_fact_target("What is the capital of Japan?")
        evidence = Evidence(
            id=1,
            source_id=1,
            research_run_id=1,
            exact_text="Tokyo is the capital of Japan.",
            evidence_type=EvidenceType.DIRECT_QUOTE,
            extraction_method=ExtractionMethod.LLM,
            is_validated=True,
            metadata={"source_url": "https://example.com"},
        )
        fact_value = extract_fact_value(evidence.exact_text, target)
        claim = build_core_claim_from_value(fact_value, evidence, 1)
        source = Source(
            id=1,
            research_run_id=1,
            url="https://britannica.com/place/Tokyo",
            title="Tokyo",
            content="Tokyo is the capital of Japan.",
            content_hash="abc",
            source_quality=SourceQuality.REPUTABLE_SECONDARY,
        )
        report = build_fast_answer(
            target, evidence, claim, [source], fact_value=fact_value
        )
        assert "[E1]" in report.content
        assert report.report_metrics.get("full_writer_skipped")
        assert report.answer_confidence_level == "high"
        assert len(report.content) < 500


class TestStandardRouteUnchanged:
    def test_comparison_uses_standard(self):
        result = classify_query("Compare OpenAI and Anthropic enterprise offerings.")
        assert result.route in (ExecutionRoute.STANDARD, ExecutionRoute.DEEP)
        assert result.route != ExecutionRoute.SIMPLE_FACT
