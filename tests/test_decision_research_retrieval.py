"""Tests for decision-critical retrieval hardening."""

from unittest.mock import AsyncMock, patch

import pytest

from domain.models import Source, SourceQuality, SourceType
from services.decision_framing_schemas import DecisionCriterion, DecisionFrame, DecisionOption, DecisionType
from services.decision_research_coverage import (
    build_authority_seeking_query,
    build_coverage_pair_specs,
    infer_official_domain_candidates,
    is_vendor_controlled_criterion,
    pin_coverage_sources,
    result_has_authoritative_hit,
)
from services.source_normalizer import normalize_search_result
from state import SearchResult


def _vendor_frame(**kwargs) -> DecisionFrame:
    return DecisionFrame(
        decision=kwargs.get("decision", "Which vendor?"),
        decision_type=DecisionType.VENDOR_SELECTION,
        options=kwargs.get("options", [
            DecisionOption(label="Vendor A", origin="explicit"),
            DecisionOption(label="Vendor B", origin="explicit"),
        ]),
        criteria=kwargs.get("criteria", [
            DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
        ]),
    )


class TestAuthoritySeekingQueries:
    def test_cost_queries_are_authority_seeking(self):
        frame = _vendor_frame()
        specs = build_coverage_pair_specs(frame)
        assert len(specs) == 2
        assert all("official" in s.primary_query.lower() for s in specs)
        assert "Vendor A" in specs[0].primary_query
        assert "Vendor B" in specs[1].primary_query

    def test_non_vendor_criterion_not_forced(self):
        frame = _vendor_frame(criteria=[
            DecisionCriterion(label="Competitive intensity", origin="explicit", priority="primary"),
        ])
        spec = build_coverage_pair_specs(frame)[0]
        assert is_vendor_controlled_criterion(spec.criterion_label) is False
        assert "official documentation" not in spec.primary_query.lower()

    def test_inferred_domain_candidates_are_general(self):
        domains = infer_official_domain_candidates("OpenAI")
        assert "openai.com" in domains
        domains_b = infer_official_domain_candidates("Anthropic")
        assert "anthropic.com" in domains_b


class TestSourcePriority:
    def test_official_pricing_survives_over_youtube(self):
        official = normalize_search_result(
            SearchResult(
                title="Pricing",
                url="https://openai.com/api/pricing",
                content="API pricing details",
                score=0.7,
            ),
            research_run_id=1,
        )
        youtube = normalize_search_result(
            SearchResult(
                title="Commentary",
                url="https://www.youtube.com/watch?v=abc",
                content="YouTube commentary",
                score=0.95,
            ),
            research_run_id=1,
        )
        frame = DecisionFrame(
            decision="Which vendor?",
            options=[DecisionOption(label="OpenAI", origin="explicit")],
            criteria=[DecisionCriterion(label="Cost", origin="explicit", priority="primary")],
        )
        specs = build_coverage_pair_specs(frame)
        pinned = pin_coverage_sources([youtube, official], specs)
        assert pinned[0].url.startswith("https://openai.com")
        assert official.source_quality == SourceQuality.OFFICIAL
        assert youtube.source_quality == SourceQuality.USER_GENERATED

    def test_first_party_hit_detection(self):
        frame = _vendor_frame(options=[DecisionOption(label="OpenAI", origin="explicit")])
        spec = build_coverage_pair_specs(frame)[0]
        results = [
            SearchResult(
                title="Pricing",
                url="https://openai.com/pricing",
                content="token pricing",
                score=0.8,
            ),
            SearchResult(
                title="Video",
                url="https://youtube.com/watch?v=1",
                content="commentary",
                score=0.9,
            ),
        ]
        assert result_has_authoritative_hit(results, spec) is True

    def test_enterprise_page_not_pricing_hit_for_cost(self):
        frame = DecisionFrame(
            decision="Which vendor?",
            options=[DecisionOption(label="Anthropic", origin="explicit")],
            criteria=[DecisionCriterion(label="Cost", origin="explicit", priority="primary")],
        )
        spec = build_coverage_pair_specs(frame)[0]
        results = [
            SearchResult(
                title="Enterprise",
                url="https://www.anthropic.com/enterprise",
                content="Anthropic enterprise",
                score=0.8,
            ),
        ]
        assert result_has_authoritative_hit(results, spec) is False


class TestResearcherRetry:
    @pytest.mark.asyncio
    async def test_one_authoritative_retry_when_initial_misses(self):
        from nodes.researcher import _execute_coverage_search
        from services.decision_research_coverage import CoveragePairSpec, DecisionCoverageMetrics

        spec = CoveragePairSpec(
            option_label="Anthropic",
            criterion_label="Cost",
            vendor_controlled=True,
            primary_query="Anthropic API pricing official",
            retry_query="Anthropic API pricing official documentation",
            official_domain_candidates=["anthropic.com"],
        )
        metrics = DecisionCoverageMetrics()

        low_quality = [
            SearchResult(title="YT", url="https://youtube.com/watch?v=1", content="commentary", score=0.9),
        ]
        official = [
            SearchResult(
                title="Claude API Pricing",
                url="https://claude.com/pricing",
                content="Anthropic Claude API pricing",
                score=0.7,
            ),
        ]

        with patch("nodes.researcher.search_tavily_with_retry", new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = [low_quality, official]
            results = await _execute_coverage_search(
                spec,
                max_results=5,
                planner_domains=None,
                metrics=metrics,
            )

        assert mock_search.call_count == 2
        assert mock_search.call_args_list[1].kwargs.get("domains") is None
        assert metrics.authoritative_retries == 1
        assert len(results) == 2
        assert any("claude.com" in r.url for r in results)

    @pytest.mark.asyncio
    async def test_no_retry_when_primary_has_authoritative_hit(self):
        from nodes.researcher import _execute_coverage_search
        from services.decision_research_coverage import CoveragePairSpec, DecisionCoverageMetrics

        spec = CoveragePairSpec(
            option_label="OpenAI",
            criterion_label="Cost",
            vendor_controlled=True,
            primary_query="OpenAI API pricing official",
            retry_query="OpenAI API pricing official documentation",
            official_domain_candidates=["openai.com"],
        )
        metrics = DecisionCoverageMetrics()
        official = [
            SearchResult(
                title="Pricing",
                url="https://openai.com/pricing",
                content="pricing",
                score=0.8,
            ),
        ]

        with patch("nodes.researcher.search_tavily_with_retry", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = official
            await _execute_coverage_search(
                spec,
                max_results=5,
                planner_domains=None,
                metrics=metrics,
            )

        assert mock_search.call_count == 1
        assert metrics.authoritative_retries == 0


    @pytest.mark.asyncio
    async def test_retry_failure_does_not_fabricate(self):
        from nodes.researcher import _execute_coverage_search
        from services.decision_research_coverage import CoveragePairSpec, DecisionCoverageMetrics

        spec = CoveragePairSpec(
            option_label="OpenAI",
            criterion_label="Cost",
            vendor_controlled=True,
            primary_query="OpenAI API pricing official",
            retry_query="OpenAI API pricing official documentation",
            official_domain_candidates=["openai.com"],
        )
        metrics = DecisionCoverageMetrics()
        low_quality = [
            SearchResult(title="YT", url="https://youtube.com/watch?v=1", content="commentary", score=0.9),
        ]

        with patch("nodes.researcher.search_tavily_with_retry", new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = [low_quality, low_quality]
            results = await _execute_coverage_search(
                spec,
                max_results=5,
                planner_domains=None,
                metrics=metrics,
            )

        assert mock_search.call_count == 2
        assert metrics.decision_coverage_pairs_without_evidence == 1
        assert all("youtube.com" in r.url for r in results)


class TestNonDecisionBehavior:
    def test_competitive_intensity_query_not_vendor_forced(self):
        query = build_authority_seeking_query(
            DecisionOption(label="Vendor A", origin="explicit"),
            "Competitive intensity",
        )
        assert query == "Vendor A Competitive intensity"

    def test_brand_split_domain_still_first_party_via_candidates(self):
        from services.decision_research_coverage import (
            domain_matches_candidate,
            is_first_party_source,
        )
        from domain.models import Source, SourceQuality, SourceType
        from datetime import datetime, timezone

        source = Source(
            research_run_id=1,
            url="https://claude.com/pricing",
            title="Claude Pricing",
            publisher="Claude",
            accessed_at=datetime.now(timezone.utc),
            source_type=SourceType.WEB,
            source_quality=SourceQuality.GENERAL_SECONDARY,
            content="Anthropic Claude pricing",
            content_hash="abc",
            relevance_score=0.9,
            metadata={"domain": "claude.com"},
        )
        assert is_first_party_source(
            source,
            "Anthropic",
            extra_domains=["claude.com", "platform.claude.com"],
        )
        assert domain_matches_candidate(
            "platform.claude.com",
            ["anthropic.com", "claude.com", "platform.claude.com"],
        )
