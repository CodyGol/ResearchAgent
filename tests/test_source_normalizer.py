"""Tests for source normalization."""

import pytest

from domain.models import SourceQuality, SourceType
from services.source_normalizer import (
    normalize_claim_text,
    normalize_search_result,
    normalize_search_results,
    normalize_search_results_with_metrics,
)
from state import SearchResult


class TestNormalizeSearchResult:
    def test_creates_source_with_hash(self):
        result = SearchResult(
            title="AI Paper",
            url="https://arxiv.org/abs/1234.5678",
            content="Neural networks achieve state of the art results.",
            score=0.9,
        )
        source = normalize_search_result(result, research_run_id=1)

        assert source.research_run_id == 1
        assert source.url == result.url
        assert source.title == "AI Paper"
        assert source.content == result.content
        assert len(source.content_hash) == 64  # SHA-256 hex
        assert source.source_type == SourceType.ACADEMIC
        assert source.source_quality == SourceQuality.ACADEMIC

    def test_classifies_official_domain(self):
        result = SearchResult(
            title="SEC Filing",
            url="https://www.sec.gov/Archives/edgar/data/123",
            content="Annual report content.",
            score=0.8,
        )
        source = normalize_search_result(result, research_run_id=2)
        assert source.source_type == SourceType.OFFICIAL
        assert source.source_quality == SourceQuality.OFFICIAL

    def test_classifies_news_domain(self):
        result = SearchResult(
            title="Market News",
            url="https://www.reuters.com/markets/article",
            content="Markets rose today.",
            score=0.7,
        )
        source = normalize_search_result(result, research_run_id=3)
        assert source.source_type == SourceType.NEWS
        assert source.source_quality == SourceQuality.REPUTABLE_SECONDARY


class TestDeduplication:
    def test_deduplicates_same_url_and_content(self):
        results = [
            SearchResult(title="A", url="https://example.com/a", content="same", score=0.9),
            SearchResult(title="A dup", url="https://example.com/a", content="same", score=0.8),
            SearchResult(title="B", url="https://example.com/b", content="different", score=0.7),
        ]
        sources = normalize_search_results(results, research_run_id=1)
        assert len(sources) == 2

    def test_merges_same_canonical_url_different_content(self):
        results = [
            SearchResult(title="A", url="https://example.com/a", content="version 1", score=0.9),
            SearchResult(title="A", url="https://www.example.com/a/", content="version 2 extra", score=0.8),
        ]
        sources, metrics = normalize_search_results_with_metrics(results, research_run_id=1)
        assert len(sources) == 1
        assert metrics.duplicates_removed == 1
        assert "version 1" in sources[0].content
        assert "version 2" in sources[0].content

    def test_canonicalizes_url_variants(self):
        results = [
            SearchResult(title="A", url="https://www.example.com/page?utm_source=twitter", content="data", score=0.9),
            SearchResult(title="B", url="https://example.com/page", content="more data", score=0.8),
        ]
        sources, metrics = normalize_search_results_with_metrics(results, research_run_id=1)
        assert len(sources) == 1
        assert metrics.duplicates_removed == 1


class TestClaimTextNormalization:
    def test_normalizes_whitespace_and_case(self):
        assert normalize_claim_text("  Company X  grew  ") == "company x grew"

    def test_preserves_numbers_and_currency(self):
        normalized = normalize_claim_text("Revenue: $4.2B in 2025")
        assert "4.2" in normalized
        assert "$" in normalized or "4.2b" in normalized
