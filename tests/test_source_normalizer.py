"""Tests for source normalization."""

import pytest

from domain.models import SourceQuality, SourceType
from services.source_normalizer import (
    is_likely_first_party_vendor_site,
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

    def test_classifies_vendor_pricing_page_as_official(self):
        result = SearchResult(
            title="API Pricing",
            url="https://openai.com/api/pricing",
            content="Token pricing.",
            score=0.8,
        )
        source = normalize_search_result(result, research_run_id=20)
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


class TestFirstPartyOwnershipClassification:
    def test_third_party_api_path_not_official(self):
        result = SearchResult(
            title="Claude API Cost",
            url="https://apidog.com/blog/claude-api-cost",
            content="Third-party commentary on Claude API costs.",
            score=0.9,
        )
        source = normalize_search_result(result, research_run_id=30)
        assert source.source_quality != SourceQuality.OFFICIAL
        assert source.source_type != SourceType.OFFICIAL

    def test_third_party_pricing_path_not_official(self):
        result = SearchResult(
            title="Vendor A Pricing",
            url="https://exampleblog.com/pricing/vendor-a",
            content="Blog roundup of vendor pricing.",
            score=0.9,
        )
        source = normalize_search_result(result, research_run_id=31)
        assert source.source_quality != SourceQuality.OFFICIAL

    def test_legitimate_first_party_pricing_on_owned_domain(self):
        result = SearchResult(
            title="Pricing",
            url="https://vendor.com/pricing",
            content="Official pricing table.",
            score=0.9,
        )
        source = normalize_search_result(result, research_run_id=32)
        assert source.source_quality == SourceQuality.OFFICIAL

    def test_legitimate_first_party_docs_subdomain(self):
        result = SearchResult(
            title="API Pricing",
            url="https://docs.vendor.example/api/pricing",
            content="Official API pricing documentation.",
            score=0.9,
        )
        source = normalize_search_result(result, research_run_id=33)
        assert source.source_quality == SourceQuality.OFFICIAL

    def test_brand_split_product_domain_pricing_stays_official(self):
        result = SearchResult(
            title="Claude Pricing",
            url="https://claude.com/pricing",
            content="Anthropic Claude pricing.",
            score=0.9,
        )
        source = normalize_search_result(result, research_run_id=34)
        assert source.source_quality == SourceQuality.OFFICIAL

    @pytest.mark.parametrize(
        ("url", "expected_official"),
        [
            ("https://apidog.com/blog/claude-api-cost", False),
            ("https://openai.com/api/pricing", True),
            ("https://docs.vendor.example/api/pricing", True),
        ],
    )
    def test_deterministic_classification_examples(self, url: str, expected_official: bool):
        from urllib.parse import urlparse

        domain = urlparse(url).netloc.removeprefix("www.")
        is_official = is_likely_first_party_vendor_site(domain, url)
        assert is_official is expected_official


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
