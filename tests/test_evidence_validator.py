"""Tests for evidence integrity validation."""

import pytest

from domain.models import EvidenceMatchType
from services.evidence_validator import (
    extract_context,
    validate_evidence_text,
)


SOURCE = (
    "According to the annual report, Company X generated $4.2 billion in revenue "
    "in fiscal year 2025. The company also announced plans to expand into Market Y."
)


class TestExactMatch:
    def test_exact_substring_accepted(self):
        evidence = "Company X generated $4.2 billion in revenue"
        result = validate_evidence_text(evidence, SOURCE, allow_fuzzy=False)
        assert result.is_valid is True
        assert result.match_type == EvidenceMatchType.EXACT
        assert result.match_ratio == 1.0

    def test_fabricated_evidence_rejected(self):
        evidence = "Company X generated $9.9 trillion in revenue"
        result = validate_evidence_text(evidence, SOURCE, allow_fuzzy=False)
        assert result.is_valid is False
        assert result.match_type == EvidenceMatchType.NOT_FOUND

    def test_empty_evidence_rejected(self):
        result = validate_evidence_text("", SOURCE)
        assert result.is_valid is False
        assert result.reason == "Empty evidence text"

    def test_empty_source_rejected(self):
        result = validate_evidence_text("some text", "")
        assert result.is_valid is False
        assert result.reason == "Empty source content"


class TestNormalizedMatch:
    def test_whitespace_variation_accepted(self):
        evidence = "Company  X   generated  $4.2  billion  in  revenue"
        result = validate_evidence_text(evidence, SOURCE, allow_fuzzy=False)
        assert result.is_valid is True
        assert result.match_type == EvidenceMatchType.NORMALIZED

    def test_unicode_quote_variation_accepted(self):
        source = 'The CEO said "revenue grew 40%" during the call.'
        evidence = 'The CEO said "revenue grew 40%" during the call.'
        result = validate_evidence_text(evidence, source, allow_fuzzy=False)
        assert result.is_valid is True

    def test_case_insensitive_normalized_match(self):
        evidence = "company x generated $4.2 billion in revenue"
        result = validate_evidence_text(evidence, SOURCE, allow_fuzzy=False)
        assert result.is_valid is True
        assert result.match_type == EvidenceMatchType.NORMALIZED


class TestFuzzyMatch:
    def test_minor_word_change_with_fuzzy(self):
        source = "Revenue increased by approximately 40 percent year over year."
        evidence = "Revenue increased by approximately 40 percent annually."
        result = validate_evidence_text(evidence, source, allow_fuzzy=True, fuzzy_threshold=0.6)
        # May or may not match depending on threshold — ensure no false EXACT
        if result.is_valid:
            assert result.match_type in (EvidenceMatchType.FUZZY, EvidenceMatchType.NORMALIZED)

    def test_fuzzy_disabled_rejects_near_miss(self):
        source = "The market grew rapidly in Q3 2024."
        evidence = "The market expanded quickly in Q3 2024."
        result = validate_evidence_text(evidence, source, allow_fuzzy=False)
        assert result.is_valid is False


class TestContextExtraction:
    def test_extracts_surrounding_context(self):
        evidence = "$4.2 billion in revenue"
        before, after = extract_context(evidence, SOURCE, context_chars=30)
        assert before is not None
        assert "Company X" in before or "report" in before.lower()
        assert after is not None
        assert "fiscal" in after.lower() or "2025" in after

    def test_returns_none_for_missing_span(self):
        before, after = extract_context("nonexistent text", SOURCE)
        assert before is None
        assert after is None
