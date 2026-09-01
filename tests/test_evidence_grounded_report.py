"""Tests for evidence confidence and report consistency."""

import pytest

from domain.models import Evidence, EvidenceMatchType, EvidenceType, ExtractionMethod, Source, SourceQuality, SourceType
from services.evidence_confidence import compute_evidence_confidence, CONFIDENCE_NUMERIC
from services.evidence_context import evidence_ids_to_urls, extract_cited_evidence_ids
from services.report_consistency import check_internal_consistency, run_consistency_checks
from domain.models import EvidenceConfidence


def _make_evidence(text: str, source_id: int = 1, display_id: str = "E1") -> Evidence:
    return Evidence(
        id=1,
        source_id=source_id,
        research_run_id=1,
        exact_text=text,
        is_validated=True,
        match_type=EvidenceMatchType.EXACT,
        evidence_type=EvidenceType.DIRECT_QUOTE,
        extraction_method=ExtractionMethod.LLM,
        metadata={"display_id": display_id, "source_url": f"https://example.com/{source_id}"},
    )


def _make_source(source_id: int, url: str, quality: SourceQuality = SourceQuality.OFFICIAL) -> Source:
    return Source(
        id=source_id,
        research_run_id=1,
        url=url,
        title="Test Source",
        content="content",
        content_hash="hash",
        source_type=SourceType.OFFICIAL,
        source_quality=quality,
        metadata={"domain": "example.com"},
    )


class TestEvidenceConfidence:
    def test_no_evidence_is_low(self):
        level, numeric, _ = compute_evidence_confidence([], [])
        assert level == EvidenceConfidence.LOW
        assert numeric == CONFIDENCE_NUMERIC[EvidenceConfidence.LOW]

    def test_conflicts_prevent_high(self):
        ev = [_make_evidence("Fact A"), _make_evidence("Contradicting fact B", source_id=2, display_id="E2")]
        sources = [
            _make_source(1, "https://a.com"),
            _make_source(2, "https://b.com"),
        ]
        level, _, _ = compute_evidence_confidence(
            ev, sources, potential_conflicts=["Conflicting race counts"]
        )
        assert level != EvidenceConfidence.HIGH

    def test_quality_issues_reduce_confidence(self):
        ev = [_make_evidence("Tokyo is the capital of Japan.")]
        sources = [_make_source(1, "https://example.com")]
        level_high, _, _ = compute_evidence_confidence(ev, sources, critique_quality_score=0.9)
        level_low, _, _ = compute_evidence_confidence(
            ev, sources,
            critique_quality_score=0.3,
            unsupported_areas=["Missing historical context"],
            consistency_issues=["22 vs 23 races"],
        )
        assert level_low != EvidenceConfidence.HIGH
        # Low critique should not exceed high critique level
        assert CONFIDENCE_NUMERIC[level_low] <= CONFIDENCE_NUMERIC[level_high]


class TestReportConsistency:
    def test_detects_conflicting_race_counts(self):
        report = "The season had 23 races. Verstappen won 19 of 22 races."
        result = check_internal_consistency(report)
        assert not result.is_consistent
        assert any("race" in issue.lower() for issue in result.issues)

    def test_consistent_report_passes(self):
        report = "The season had 22 races. Verstappen won 19 of 22 races."
        result = check_internal_consistency(report)
        assert result.is_consistent

    def test_unsupported_number_warning(self):
        report = "Tokyo is the capital of Japan with 15 million residents."
        evidence_texts = ["Tokyo is the capital of Japan."]
        result = run_consistency_checks(report, evidence_texts)
        assert any("15" in w for w in result.warnings)


class TestCitationGrounding:
    def test_only_cited_evidence_sources_included(self):
        ev1 = _make_evidence("Tokyo is the capital.", source_id=1, display_id="E1")
        ev2 = _make_evidence("Osaka is a major city.", source_id=2, display_id="E2")
        sources = [
            _make_source(1, "https://japan.com/tokyo"),
            _make_source(2, "https://japan.com/osaka"),
        ]
        content = "Tokyo is the capital of Japan [E1]."
        cited = extract_cited_evidence_ids(content)
        urls = evidence_ids_to_urls(cited, [ev1, ev2], sources)
        assert urls == ["https://japan.com/tokyo"]
        assert "https://japan.com/osaka" not in urls
