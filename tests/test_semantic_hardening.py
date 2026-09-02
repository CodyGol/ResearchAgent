"""Post-smoke semantic hardening tests."""

from datetime import date

import pytest

from domain.models import Claim, ClaimType
from services.decision_framing_schemas import DecisionCriterion, DecisionFrame, DecisionOption, DecisionType
from services.decision_research_coverage import build_coverage_subqueries, merge_decision_coverage_into_plan
from services.decision_synthesis import (
    build_oe_index,
    compute_status_ceiling,
    detect_comparative_coverage_gaps,
    run_pre_check,
    validate_and_build_synthesis,
)
from services.decision_synthesis_schemas import (
    ConstraintCompliance,
    CriterionReferenceLLM,
    DecisionSynthesisLLMOutput,
    RecommendationStatus,
)
from services.knowledge_state_schemas import KnowledgeState
from services.option_evaluation import build_claim_catalog, validate_and_build_evaluation
from services.option_evaluation_schemas import (
    ClaimCatalogEntry,
    CriterionAssessment,
    CriterionEvaluationLLM,
    KnowledgeCoverage,
    OptionEvaluationLLMOutput,
)
from state import ResearchPlan
from tests.test_decision_synthesis import (
    _build_oe,
    _entry,
    _ks,
    _llm_out,
    _oe_row,
    _vendor_frame,
)
from utils.runtime_date import classify_date_relative, set_reference_date_for_tests


@pytest.fixture(autouse=True)
def _reset_reference_date():
    set_reference_date_for_tests(None)
    yield
    set_reference_date_for_tests(None)


class TestComparativeCoverageGap:
    def test_primary_asymmetry_blocks_recommend(self):
        frame = _vendor_frame(criteria=[
            DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
        ])
        oe = _build_oe(
            "Which vendor?",
            [
                _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1]),
                _oe_row("Vendor B", "Cost", CriterionAssessment.INSUFFICIENT_INFORMATION, KnowledgeCoverage.INSUFFICIENT, []),
            ],
        )
        oe_index = build_oe_index(oe)
        gaps = detect_comparative_coverage_gaps(frame, oe_index)
        assert "Cost" in gaps

        pre = run_pre_check(frame, oe)
        ceiling = compute_status_ceiling(
            frame,
            pre,
            oe_index,
            recommended_option="Vendor A",
            constraint_assessments=[],
            supporting=[],
            limiting=[],
            critical_missing_context=[],
            comparative_coverage_gaps=gaps,
        )
        assert ceiling != RecommendationStatus.RECOMMEND
        assert ceiling in (
            RecommendationStatus.TENTATIVE_RECOMMENDATION,
            RecommendationStatus.INSUFFICIENT_BASIS,
        )

    def test_comparable_cost_evidence_allows_recommend(self):
        frame = _vendor_frame(criteria=[
            DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
        ])
        oe = _build_oe(
            "Which vendor?",
            [
                _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1]),
                _oe_row("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2]),
            ],
        )
        oe_index = build_oe_index(oe)
        gaps = detect_comparative_coverage_gaps(frame, oe_index)
        assert gaps == []

        pre = run_pre_check(frame, oe)
        ceiling = compute_status_ceiling(
            frame,
            pre,
            oe_index,
            recommended_option="Vendor A",
            constraint_assessments=[],
            supporting=[],
            limiting=[],
            critical_missing_context=[],
            comparative_coverage_gaps=gaps,
        )
        assert ceiling == RecommendationStatus.RECOMMEND

    def test_single_primary_gap_yields_insufficient_basis(self):
        frame = _vendor_frame(criteria=[
            DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
        ])
        oe = _build_oe(
            "Which vendor?",
            [
                _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1]),
                _oe_row("Vendor B", "Cost", CriterionAssessment.INSUFFICIENT_INFORMATION, KnowledgeCoverage.INSUFFICIENT, []),
            ],
        )
        oe_index = build_oe_index(oe)
        gaps = detect_comparative_coverage_gaps(frame, oe_index)
        pre = run_pre_check(frame, oe)
        ceiling = compute_status_ceiling(
            frame,
            pre,
            oe_index,
            recommended_option="Vendor A",
            constraint_assessments=[],
            supporting=[],
            limiting=[],
            critical_missing_context=[],
            comparative_coverage_gaps=gaps,
        )
        assert ceiling == RecommendationStatus.INSUFFICIENT_BASIS


class TestNonComparableEvidenceScope:
    def test_api_vs_subscription_downgrades_directional(self):
        frame = _vendor_frame()
        catalog = {
            1: ClaimCatalogEntry(
                claim_id=1,
                bucket="known",
                verification_id=10,
                verification_status="supported",
                knowledge_category="known",
                claim_text="OpenAI API pricing is $2.50 per million input tokens.",
            ),
            2: ClaimCatalogEntry(
                claim_id=2,
                bucket="known",
                verification_id=20,
                verification_status="supported",
                knowledge_category="known",
                claim_text="Anthropic Claude Code Pro subscription is $20 per month.",
            ),
        }
        llm_out = OptionEvaluationLLMOutput(
            evaluations=[
                CriterionEvaluationLLM(
                    option_label="Vendor A",
                    criterion_label="Cost",
                    assessment=CriterionAssessment.FAVORABLE,
                    claim_ids=[1],
                    reason="API pricing is documented.",
                ),
                CriterionEvaluationLLM(
                    option_label="Vendor B",
                    criterion_label="Cost",
                    assessment=CriterionAssessment.FAVORABLE,
                    claim_ids=[2],
                    reason="Subscription pricing mentioned.",
                ),
            ]
        )
        evaluation, metrics = validate_and_build_evaluation(llm_out, frame, catalog)
        assert evaluation is not None
        rows = {
            (opt.option_label, ce.criterion_label): ce
            for opt in evaluation.option_evaluations
            for ce in opt.criteria_evaluations
        }
        assert rows[("Vendor A", "Cost")].assessment == CriterionAssessment.UNCERTAIN
        assert rows[("Vendor B", "Cost")].assessment == CriterionAssessment.UNCERTAIN
        assert metrics.evaluation_failed is False


class TestDecisionResearchCoverage:
    def test_builds_option_primary_pairs(self):
        frame = DecisionFrame(
            decision="Which vendor?",
            decision_type=DecisionType.VENDOR_SELECTION,
            options=[
                DecisionOption(label="OpenAI", origin="explicit"),
                DecisionOption(label="Anthropic", origin="explicit"),
            ],
            criteria=[DecisionCriterion(label="Cost", origin="explicit", priority="primary")],
        )
        queries = build_coverage_subqueries(frame)
        assert len(queries) == 2
        assert any("OpenAI" in q and "pricing" in q.lower() and "official" in q.lower() for q in queries)
        assert any("Anthropic" in q and "pricing" in q.lower() and "official" in q.lower() for q in queries)

    def test_merge_prepends_coverage_queries(self):
        frame = DecisionFrame(
            decision="Which vendor?",
            decision_type=DecisionType.VENDOR_SELECTION,
            options=[
                DecisionOption(label="OpenAI", origin="explicit"),
                DecisionOption(label="Anthropic", origin="explicit"),
            ],
            criteria=[DecisionCriterion(label="Cost", origin="explicit", priority="primary")],
        )
        plan = ResearchPlan(
            query="vendor choice",
            sub_queries=["generic vendor overview"],
            search_terms=["vendor"],
        )
        merged = merge_decision_coverage_into_plan(plan, frame, max_queries=3)
        assert len(merged.sub_queries) == 3
        assert "OpenAI" in merged.sub_queries[0]
        assert "official" in merged.sub_queries[0].lower()
        assert "Anthropic" in merged.sub_queries[1]


class TestRuntimeDate:
    def test_july_30_2026_is_past_on_september_1_2026(self):
        set_reference_date_for_tests(date(2026, 9, 1))
        assert classify_date_relative(date(2026, 7, 30)) == "past"

    def test_october_1_2026_is_future_on_september_1_2026(self):
        set_reference_date_for_tests(date(2026, 9, 1))
        assert classify_date_relative(date(2026, 10, 1)) == "future"


class TestSynthesisIntegrationGap:
    def test_validate_caps_recommend_under_comparative_gap(self):
        frame = _vendor_frame(criteria=[
            DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
        ])
        oe = _build_oe(
            "Which vendor?",
            [
                _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1]),
                _oe_row("Vendor B", "Cost", CriterionAssessment.INSUFFICIENT_INFORMATION, KnowledgeCoverage.INSUFFICIENT, []),
            ],
        )
        claims = [
            Claim(id=1, research_run_id=1, text="Vendor A API pricing is lower.", claim_type=ClaimType.FACTUAL),
        ]
        catalog = build_claim_catalog(_ks(known=[_entry(1, "known")]), claims)

        pre = run_pre_check(frame, oe)
        llm_output = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor A",
            rationale="Vendor A is cheaper on cost.",
            supporting_criteria=[
                CriterionReferenceLLM(option_label="Vendor A", criterion_label="Cost")
            ],
        )
        synthesis, metrics, _ = validate_and_build_synthesis(
            llm_output, frame, oe, catalog, pre
        )
        assert synthesis is not None
        assert synthesis.recommendation_status != RecommendationStatus.RECOMMEND
