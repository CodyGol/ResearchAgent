"""Tests for structured fact values and Phase 2B.7 fast fact engine."""

import pytest

from domain.models import Evidence, EvidenceType, ExtractionMethod, Source, SourceQuality
from services.fact_sufficiency import check_fact_sufficiency, detect_conflicting_values
from services.fact_target import extract_fact_target, FactDomain, FactFreshness
from services.fact_value import (
    FactValueType,
    build_canonical_claim_from_value,
    cache_key_for_target,
    classify_freshness,
    detect_value_conflicts,
    extract_fact_value,
    validate_fact_value_in_evidence,
)
from services.fast_evidence import _try_deterministic_evidence
from services.source_authority import is_source_adequate_for_domain


class TestStructuredFactValue:
    def test_capital_extraction(self):
        target = extract_fact_target("What is the capital of Japan?")
        fv = extract_fact_value("Tokyo is the capital of Japan.", target)
        assert fv is not None
        assert fv.value == "Tokyo"
        assert fv.value_type == FactValueType.PLACE
        assert fv.attribute == "capital"

    def test_standings_winner_extraction(self):
        target = extract_fact_target("Who won the 2023 F1 World Championship?")
        text = "2023 DRIVERS' STANDINGS POS DRIVER 1 Max Verstappen Red Bull 575"
        fv = extract_fact_value(text, target)
        assert fv is not None
        assert fv.value == "Max Verstappen"
        target = extract_fact_target("Who won the 2023 F1 World Championship?")
        text = (
            "Max Verstappen secured his third Formula 1 world championship "
            "in 2023 with Red Bull Racing."
        )
        fv = extract_fact_value(text, target)
        assert fv is not None
        assert fv.value == "Max Verstappen"
        assert fv.value_type == FactValueType.PERSON
        assert fv.temporal_scope == "2023"
        assert fv.category == "drivers_championship"

    def test_f1_championship_disambiguation(self):
        target = extract_fact_target("Who won the 2023 F1 World Championship?")
        assert "drivers" in target.entity.lower() or target.category == "drivers_championship"

    def test_revenue_extraction(self):
        target = extract_fact_target("What was Apple's revenue in fiscal 2025?")
        text = (
            "Apple reported total net sales of $391.0 billion for fiscal 2025."
        )
        fv = extract_fact_value(text, target)
        assert fv is not None
        assert fv.value_type == FactValueType.NUMBER
        assert "391" in fv.value
        assert fv.currency == "USD"
        assert "fiscal" in (fv.temporal_scope or "").lower()

    def test_ceo_freshness(self):
        target = extract_fact_target("Who is the CEO of Apple?")
        assert classify_freshness(target) == FactFreshness.TIME_SENSITIVE

    def test_capital_freshness(self):
        target = extract_fact_target("What is the capital of Japan?")
        assert classify_freshness(target) == FactFreshness.STRUCTURALLY_STABLE

    def test_date_extraction(self):
        target = extract_fact_target("When was Python first released?")
        text = "Python was first released in February 20, 1991."
        fv = extract_fact_value(text, target)
        assert fv is not None
        assert fv.value_type == FactValueType.DATE
        assert "1991" in fv.value


class TestCanonicalClaims:
    def test_capital_template(self):
        target = extract_fact_target("What is the capital of Japan?")
        fv = extract_fact_value("Tokyo is the capital of Japan.", target)
        claim = build_canonical_claim_from_value(fv)
        assert claim == "Tokyo is the capital of Japan."

    def test_standings_claim_template(self):
        target = extract_fact_target("Who won the 2023 F1 World Championship?")
        fv = extract_fact_value(
            "| 1 | Max VerstappenVER | NED | Red Bull | 575 |", target
        )
        claim = build_canonical_claim_from_value(
            fv, "| 1 | Max VerstappenVER | NED | Red Bull | 575 |"
        )
        assert "finished in first position" in claim
        assert "Max Verstappen" in claim
        target = extract_fact_target("Who won the 2023 F1 World Championship?")
        fv = extract_fact_value(
            "Max Verstappen won the 2023 Formula One World Drivers' Championship.",
            target,
        )
        claim = build_canonical_claim_from_value(fv)
        assert "Max Verstappen" in claim
        assert "2023" in claim

    def test_revenue_template(self):
        target = extract_fact_target("What was Apple's revenue in fiscal 2025?")
        fv = extract_fact_value(
            "Apple reported revenue of $391.0 billion for fiscal 2025.", target
        )
        claim = build_canonical_claim_from_value(fv)
        assert "Apple" in claim
        assert "391" in claim
        assert "fiscal 2025" in claim.lower()


class TestValueValidation:
    def test_person_present(self):
        target = extract_fact_target("Who won the 2023 F1 World Championship?")
        fv = extract_fact_value(
            "Max Verstappen won the championship in 2023.", target
        )
        ok, _ = validate_fact_value_in_evidence(fv, "Max Verstappen won the championship in 2023.")
        assert ok

    def test_person_missing_fails(self):
        target = extract_fact_target("Who won the 2023 F1 World Championship?")
        fv = extract_fact_value(
            "Max Verstappen won the championship in 2023.", target
        )
        ok, reason = validate_fact_value_in_evidence(fv, "Lewis Hamilton won in 2022.")
        assert not ok
        assert "not found" in reason.lower()

    def test_numeric_validation(self):
        target = extract_fact_target("What was Apple's revenue in fiscal 2025?")
        fv = extract_fact_value(
            "Revenue of $391.0 billion for fiscal 2025.", target
        )
        ok, _ = validate_fact_value_in_evidence(
            fv, "Revenue of $391.0 billion for fiscal 2025."
        )
        assert ok


class TestConflictEscalation:
    def test_revenue_conflict(self):
        target = extract_fact_target("What was Apple's revenue in fiscal 2025?")
        text1 = "Apple reported revenue of $391.0 billion for fiscal 2025."
        text2 = "Apple reported revenue of $394.0 billion for fiscal 2025."
        fv1 = extract_fact_value(text1, target)
        fv2 = extract_fact_value(text2, target)
        assert fv1 is not None and fv2 is not None
        conflict = detect_value_conflicts([fv1, fv2], target)
        assert conflict is not None

    def test_winner_conflict(self):
        target = extract_fact_target("Who won the 2023 F1 World Championship?")
        ev1 = Evidence(
            id=1, source_id=1, research_run_id=1,
            exact_text="Max Verstappen won the 2023 championship.",
            evidence_type=EvidenceType.DIRECT_QUOTE,
            extraction_method=ExtractionMethod.RULE,
            is_validated=True,
        )
        ev2 = Evidence(
            id=2, source_id=2, research_run_id=1,
            exact_text="Lewis Hamilton won the 2023 championship.",
            evidence_type=EvidenceType.DIRECT_QUOTE,
            extraction_method=ExtractionMethod.RULE,
            is_validated=True,
        )
        conflict = detect_conflicting_values([ev1, ev2], target)
        assert conflict is not None


class TestSourceAuthority:
    def test_formula1_official(self):
        source = Source(
            id=1, research_run_id=1,
            url="https://www.formula1.com/en/results/2023/drivers",
            title="F1 Results",
            content="Results",
            content_hash="x",
            source_quality=SourceQuality.GENERAL_SECONDARY,
        )
        assert is_source_adequate_for_domain(source, FactDomain.SPORTS)

    def test_apple_ir_official(self):
        source = Source(
            id=1, research_run_id=1,
            url="https://investor.apple.com/sec-filings",
            title="Apple IR",
            content="Filings",
            content_hash="x",
            source_quality=SourceQuality.GENERAL_SECONDARY,
        )
        assert is_source_adequate_for_domain(source, FactDomain.FINANCIAL)


class TestF1Sufficiency:
    def _f1_source(self) -> Source:
        return Source(
            id=1,
            research_run_id=1,
            url="https://www.formula1.com/en/results/2023/drivers",
            title="2023 Results",
            content=(
                "Max Verstappen secured his third Formula 1 world championship "
                "in 2023, dominating the season with Red Bull."
            ),
            content_hash="f1",
            source_quality=SourceQuality.GENERAL_SECONDARY,
        )

    def test_f1_standings_sufficiency(self):
        target = extract_fact_target("Who won the 2023 F1 World Championship?")
        source = Source(
            id=1,
            research_run_id=1,
            url="https://www.formula1.com/en/results/2023/drivers",
            title="2023 Results",
            content="2023 DRIVERS' STANDINGS 1 Max Verstappen Red Bull 575",
            content_hash="f1",
            source_quality=SourceQuality.OFFICIAL,
        )
        evidence = Evidence(
            id=1, source_id=1, research_run_id=1,
            exact_text="2023 DRIVERS' STANDINGS 1 Max Verstappen Red Bull 575",
            evidence_type=EvidenceType.DIRECT_QUOTE,
            extraction_method=ExtractionMethod.RULE,
            is_validated=True,
        )
        result = check_fact_sufficiency(target, evidence, source)
        assert result.is_sufficient
        assert result.fact_value.value == "Max Verstappen"

        target = extract_fact_target("Who won the 2023 F1 World Championship?")
        source = self._f1_source()
        candidate = _try_deterministic_evidence(source, target)
        assert candidate is not None
        evidence = Evidence(
            id=1, source_id=1, research_run_id=1,
            exact_text=candidate.text,
            evidence_type=EvidenceType.DIRECT_QUOTE,
            extraction_method=ExtractionMethod.RULE,
            is_validated=True,
        )
        result = check_fact_sufficiency(target, evidence, source)
        assert result.is_sufficient
        assert result.fact_value is not None
        assert result.fact_value.value == "Max Verstappen"


class TestCacheKey:
    def test_cache_key_stable(self):
        target = extract_fact_target("What is the capital of Japan?")
        key = cache_key_for_target(target)
        assert "capital" in key
        assert "japan" in key
