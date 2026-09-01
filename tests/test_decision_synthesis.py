"""Tests for Phase 3C evidence-grounded decision synthesis."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.models import Claim, ClaimType
from graph import create_graph
from nodes.decision_synthesizer import decision_synthesizer_node
from services.decision_framing_schemas import DecisionCriterion, DecisionFrame, DecisionOption, DecisionType
from services.decision_synthesis import (
    _cap_status,
    _downgrade_constraint_compliance,
    _has_fabricated_threshold,
    run_pre_check,
    synthesize_decision,
    validate_and_build_synthesis,
)
from services.decision_synthesis_schemas import (
    ChangeConditionLLM,
    ConstraintAssessmentLLM,
    ConstraintCompliance,
    CriterionReferenceLLM,
    DecisionSynthesisLLMOutput,
    RecommendationStatus,
)
from services.knowledge_state_schemas import KnowledgeState, KnowledgeStateEntry
from services.option_evaluation import build_claim_catalog
from services.option_evaluation_schemas import (
    CriterionAssessment,
    CriterionEvaluation,
    KnowledgeCoverage,
    OptionEvaluation,
    OptionEvaluationEntry,
)
from domain.models import EvidenceConfidence, KnowledgeCategory, VerificationStatus


def _claim(cid: int, text: str) -> Claim:
    return Claim(id=cid, research_run_id=1, text=text, claim_type=ClaimType.FACTUAL)


def _entry(cid: int, bucket: str) -> KnowledgeStateEntry:
    status_map = {
        "known": VerificationStatus.SUPPORTED,
        "disputed": VerificationStatus.UNCERTAIN,
        "contradicted": VerificationStatus.CONTRADICTED,
    }
    cat_map = {"known": KnowledgeCategory.KNOWN, "disputed": KnowledgeCategory.DISPUTED}
    return KnowledgeStateEntry(
        claim_id=cid,
        verification_id=cid * 10,
        knowledge_category=cat_map.get(bucket),
        verification_status=status_map.get(bucket, VerificationStatus.SUPPORTED),
        confidence=EvidenceConfidence.HIGH,
        evidence_ids=[cid * 100],
    )


def _ks(**buckets) -> KnowledgeState:
    return KnowledgeState(**{k: buckets.get(k, []) for k in [
        "known", "likely", "disputed", "unknown", "contradicted", "unverifiable"
    ]})


def _oe_row(opt: str, crit: str, assessment: CriterionAssessment, coverage: KnowledgeCoverage,
            claim_ids: list[int], *, origin="explicit", priority="standard") -> tuple:
    return (opt, crit, assessment, coverage, claim_ids, origin, priority)


def _build_oe(decision: str, rows: list[tuple], *, constraints=None, limitations=None) -> OptionEvaluation:
    by_opt: dict[str, OptionEvaluationEntry] = {}
    for opt, crit, assessment, coverage, claim_ids, origin, priority in rows:
        if opt not in by_opt:
            by_opt[opt] = OptionEvaluationEntry(option_label=opt, option_origin="explicit")
        by_opt[opt].criteria_evaluations.append(
            CriterionEvaluation(
                criterion_label=crit,
                criterion_origin=origin,
                criterion_priority=priority,
                assessment=assessment,
                knowledge_coverage=coverage,
                claim_ids=claim_ids,
                reason=f"Eval for {opt} on {crit}.",
            )
        )
    return OptionEvaluation(
        decision=decision,
        option_evaluations=list(by_opt.values()),
        constraints=constraints or [],
        decision_limitations=limitations or [],
    )


def _vendor_frame(**kwargs) -> DecisionFrame:
    return DecisionFrame(
        decision=kwargs.get("decision", "Which vendor should we use?"),
        decision_type=DecisionType.VENDOR_SELECTION,
        options=[
            DecisionOption(label="Vendor A", origin="explicit"),
            DecisionOption(label="Vendor B", origin="explicit"),
        ],
        criteria=kwargs.get("criteria", [
            DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
        ]),
        constraints=kwargs.get("constraints", []),
        missing_decision_context=kwargs.get("missing_decision_context", []),
        explicit_assumptions=kwargs.get("explicit_assumptions", []),
    )


def _llm_out(**kwargs) -> DecisionSynthesisLLMOutput:
    return DecisionSynthesisLLMOutput(**kwargs)


def _validate(frame, oe, llm_out, ks=None, claims=None):
    ks = ks or _ks()
    claims = claims or []
    catalog = build_claim_catalog(ks, claims)
    pre = run_pre_check(frame, oe)
    return validate_and_build_synthesis(llm_out, frame, oe, catalog, pre)


class TestClearRecommendation:
    def test_scenario_a_recommend_vendor_a(self):
        frame = _vendor_frame()
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], priority="primary"),
        ])
        llm = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor A",
            rationale="Vendor A is favorable on primary Cost criterion with grounded evidence.",
            supporting_criteria=[
                CriterionReferenceLLM(option_label="Vendor A", criterion_label="Cost"),
            ],
            limiting_criteria=[
                CriterionReferenceLLM(option_label="Vendor B", criterion_label="Cost"),
            ],
        )
        synthesis, metrics, _ = _validate(frame, oe, llm, _ks(known=[_entry(1, "known"), _entry(2, "known")]),
                                         [_claim(1, "A costs 5000"), _claim(2, "B costs 18000")])
        assert synthesis is not None
        assert synthesis.recommendation_status == RecommendationStatus.RECOMMEND
        assert synthesis.recommended_option == "Vendor A"
        assert synthesis.supporting_criteria[0].criterion_priority == "primary"
        assert synthesis.supporting_criteria[0].claim_ids == [1]


class TestSplitCriteria:
    def test_scenario_b_split_explicit_criteria(self):
        frame = _vendor_frame(criteria=[
            DecisionCriterion(label="Cost", origin="explicit"),
            DecisionCriterion(label="Reliability", origin="explicit"),
        ])
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1]),
            _oe_row("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2]),
            _oe_row("Vendor A", "Reliability", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [3]),
            _oe_row("Vendor B", "Reliability", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [4]),
        ])
        llm = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor A",
            rationale="Split strengths across explicit criteria.",
            supporting_criteria=[CriterionReferenceLLM(option_label="Vendor A", criterion_label="Cost")],
            limiting_criteria=[CriterionReferenceLLM(option_label="Vendor A", criterion_label="Reliability")],
        )
        synthesis, _, _ = _validate(frame, oe, llm)
        assert synthesis.recommendation_status in (
            RecommendationStatus.TENTATIVE_RECOMMENDATION,
            RecommendationStatus.INSUFFICIENT_BASIS,
        )


class TestPrimaryCriterion:
    def test_scenario_c_primary_cost_wins(self):
        frame = _vendor_frame(criteria=[
            DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
            DecisionCriterion(label="Reliability", origin="explicit", priority="standard"),
        ])
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], priority="primary"),
            _oe_row("Vendor A", "Reliability", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [3], priority="standard"),
            _oe_row("Vendor B", "Reliability", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [4], priority="standard"),
        ])
        llm = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor A",
            rationale="Primary Cost criterion favors Vendor A despite Reliability tradeoff.",
            supporting_criteria=[CriterionReferenceLLM(option_label="Vendor A", criterion_label="Cost")],
            limiting_criteria=[CriterionReferenceLLM(option_label="Vendor A", criterion_label="Reliability")],
        )
        synthesis, _, _ = _validate(frame, oe, llm)
        assert synthesis.recommended_option == "Vendor A"
        assert synthesis.recommendation_status == RecommendationStatus.RECOMMEND


class TestInferredCannotOverridePrimary:
    def test_scenario_d_inferred_does_not_override_primary(self):
        frame = _vendor_frame(criteria=[
            DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
            DecisionCriterion(label="Strategic fit", origin="inferred"),
        ])
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [1], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [2], priority="primary"),
            _oe_row("Vendor A", "Strategic fit", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [3], origin="inferred"),
            _oe_row("Vendor B", "Strategic fit", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [4], origin="inferred"),
        ])
        llm = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor A",
            rationale="Inferred strategic fit favors A.",
            supporting_criteria=[CriterionReferenceLLM(option_label="Vendor A", criterion_label="Strategic fit")],
            limiting_criteria=[CriterionReferenceLLM(option_label="Vendor A", criterion_label="Cost")],
        )
        synthesis, _, _ = _validate(frame, oe, llm)
        assert synthesis.recommendation_status != RecommendationStatus.RECOMMEND or synthesis.recommended_option != "Vendor A"


class TestConstraints:
    def test_scenario_e_constraint_violation_blocks_recommend(self):
        frame = _vendor_frame(constraints=["Budget must be under $20,000 per year"])
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], priority="primary"),
        ], constraints=frame.constraints)
        llm = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor A",
            rationale="Strong on cost.",
            constraint_assessments=[
                ConstraintAssessmentLLM(
                    option_label="Vendor A",
                    constraint="Budget must be under $20,000 per year",
                    compliance=ConstraintCompliance.VIOLATED,
                    claim_ids=[99],
                    reason="Vendor A costs $25,000 per year per claim 99.",
                ),
                ConstraintAssessmentLLM(
                    option_label="Vendor B",
                    constraint="Budget must be under $20,000 per year",
                    compliance=ConstraintCompliance.SATISFIED,
                    claim_ids=[2],
                    reason="Vendor B within budget.",
                ),
            ],
        )
        synthesis, metrics, _ = _validate(
            frame, oe, llm,
            _ks(known=[_entry(99, "known"), _entry(2, "known")]),
            [_claim(99, "Vendor A costs $25,000 per year"), _claim(2, "Vendor B costs $15,000 per year")],
        )
        assert synthesis.recommendation_status == RecommendationStatus.INSUFFICIENT_BASIS
        assert synthesis.recommended_option is None
        assert metrics.constraint_violation_count >= 1

    def test_scenario_f_constraint_not_established_caps_recommend(self):
        frame = _vendor_frame(constraints=["Must integrate with Salesforce"])
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], priority="primary"),
        ], constraints=frame.constraints)
        llm = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor A",
            rationale="Cost favors A.",
            constraint_assessments=[
                ConstraintAssessmentLLM(
                    option_label="Vendor A",
                    constraint="Must integrate with Salesforce",
                    compliance=ConstraintCompliance.NOT_ESTABLISHED,
                    claim_ids=[],
                    reason="No verified integration evidence.",
                ),
                ConstraintAssessmentLLM(
                    option_label="Vendor B",
                    constraint="Must integrate with Salesforce",
                    compliance=ConstraintCompliance.NOT_ESTABLISHED,
                    claim_ids=[],
                    reason="No verified integration evidence.",
                ),
            ],
        )
        synthesis, _, _ = _validate(frame, oe, llm)
        assert synthesis.recommendation_status == RecommendationStatus.TENTATIVE_RECOMMENDATION

    def test_scenario_g_constraint_claim_outside_oe(self):
        frame = _vendor_frame(constraints=["Budget must be under $20,000 per year"])
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], priority="primary"),
        ], constraints=frame.constraints)
        llm = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor A",
            rationale="Within budget per constraint-specific claim.",
            constraint_assessments=[
                ConstraintAssessmentLLM(
                    option_label="Vendor A",
                    constraint="Budget must be under $20,000 per year",
                    compliance=ConstraintCompliance.SATISFIED,
                    claim_ids=[500],
                    reason="Claim 500 establishes Vendor A at $12,000 per year.",
                ),
                ConstraintAssessmentLLM(
                    option_label="Vendor B",
                    constraint="Budget must be under $20,000 per year",
                    compliance=ConstraintCompliance.SATISFIED,
                    claim_ids=[501],
                    reason="Claim 501 establishes Vendor B at $15,000 per year.",
                ),
            ],
        )
        synthesis, _, _ = _validate(
            frame, oe, llm,
            _ks(known=[_entry(500, "known"), _entry(501, "known"), _entry(1, "known"), _entry(2, "known")]),
            [
                _claim(1, "A has good features"),
                _claim(2, "B is expensive on features"),
                _claim(500, "Vendor A annual cost is $12,000 per year"),
                _claim(501, "Vendor B annual cost is $15,000 per year"),
            ],
        )
        assert synthesis.recommendation_status == RecommendationStatus.RECOMMEND
        ca = next(c for c in synthesis.constraint_assessments if c.option_label == "Vendor A")
        assert 500 in ca.claim_ids
        assert 500 not in [cid for r in oe.option_evaluations for ce in r.criteria_evaluations for cid in ce.claim_ids]


class TestInsufficientAndUncertainty:
    def test_scenario_h_mostly_insufficient(self):
        frame = _vendor_frame()
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.INSUFFICIENT_INFORMATION, KnowledgeCoverage.INSUFFICIENT, [], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.INSUFFICIENT_INFORMATION, KnowledgeCoverage.INSUFFICIENT, [], priority="primary"),
        ])
        llm = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor A",
            rationale="Guess.",
        )
        synthesis, _, _ = _validate(frame, oe, llm)
        assert synthesis.recommendation_status == RecommendationStatus.INSUFFICIENT_BASIS
        assert synthesis.recommended_option is None

    def test_scenario_i_uncertain_primary_caps_recommend(self):
        frame = _vendor_frame()
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.UNCERTAIN, KnowledgeCoverage.PARTIAL, [1], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [2], priority="primary"),
        ])
        llm = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor A",
            rationale="Prefer A despite cost uncertainty.",
            supporting_criteria=[CriterionReferenceLLM(option_label="Vendor A", criterion_label="Cost")],
        )
        synthesis, _, _ = _validate(frame, oe, llm, _ks(disputed=[_entry(1, "disputed")]), [_claim(1, "disputed"), _claim(2, "known")])
        assert synthesis.recommendation_status == RecommendationStatus.TENTATIVE_RECOMMENDATION


class TestAssumptionsAndMissingContext:
    def test_scenario_j_assumption_dependency(self):
        frame = _vendor_frame(explicit_assumptions=["Interest rates stay above 4%"])
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], priority="primary"),
        ])
        llm = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor A",
            rationale="Depends on rate assumption.",
            assumptions_relied_on=["Interest rates stay above 4%"],
            change_conditions=[
                ChangeConditionLLM(
                    description="If interest rates fall below 4%, cost advantage may change.",
                    change_type="decision_context_change",
                    related_assumption="Interest rates stay above 4%",
                ),
            ],
        )
        synthesis, _, _ = _validate(frame, oe, llm)
        assert "Interest rates stay above 4%" in synthesis.assumptions_relied_on
        assert any("4%" in cc.description for cc in synthesis.change_conditions)

    def test_scenario_k_critical_missing_context_subset(self):
        frame = _vendor_frame(missing_decision_context=["Acceptable acquisition price"])
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], priority="primary"),
        ], limitations=frame.missing_decision_context)
        llm = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor A",
            rationale="Cost favors A but price context missing.",
            critical_missing_context=["Acceptable acquisition price"],
        )
        synthesis, _, _ = _validate(frame, oe, llm)
        assert synthesis.critical_missing_context == ["Acceptable acquisition price"]
        assert synthesis.recommendation_status == RecommendationStatus.TENTATIVE_RECOMMENDATION

    def test_scenario_l_invented_missing_context_rejected(self):
        frame = _vendor_frame(missing_decision_context=["Acceptable acquisition price"])
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], priority="primary"),
        ])
        llm = _llm_out(
            recommendation_status=RecommendationStatus.TENTATIVE_RECOMMENDATION,
            recommended_option="Vendor A",
            rationale="Lean.",
            critical_missing_context=["Invented strategic objective"],
        )
        synthesis, _, _ = _validate(frame, oe, llm)
        assert synthesis.critical_missing_context == []


class TestChangeConditionsAndValidation:
    def test_scenario_m_change_conditions_traceable(self):
        frame = _vendor_frame(constraints=["Must integrate with Salesforce"])
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], priority="primary"),
        ], constraints=frame.constraints)
        llm = _llm_out(
            recommendation_status=RecommendationStatus.TENTATIVE_RECOMMENDATION,
            recommended_option="Vendor A",
            rationale="Cost lean with integration uncertainty.",
            change_conditions=[
                ChangeConditionLLM(
                    description="If Salesforce integration cannot be established for Vendor A.",
                    change_type="evidence_change",
                    related_constraint="Must integrate with Salesforce",
                    related_claim_ids=[1],
                ),
            ],
            constraint_assessments=[
                ConstraintAssessmentLLM(option_label="Vendor A", constraint="Must integrate with Salesforce",
                                      compliance=ConstraintCompliance.NOT_ESTABLISHED, claim_ids=[], reason="Unknown"),
                ConstraintAssessmentLLM(option_label="Vendor B", constraint="Must integrate with Salesforce",
                                      compliance=ConstraintCompliance.NOT_ESTABLISHED, claim_ids=[], reason="Unknown"),
            ],
        )
        synthesis, _, _ = _validate(frame, oe, llm, _ks(known=[_entry(1, "known")]), [_claim(1, "A low cost")])
        assert synthesis.change_conditions
        assert synthesis.change_conditions[0].related_constraint == "Must integrate with Salesforce"

    def test_scenario_n_fabricated_threshold_rejected(self):
        trusted = {"$20,000", "4%"}
        assert _has_fabricated_threshold("If cost exceeds $50,000", trusted)
        assert not _has_fabricated_threshold("If cost exceeds $20,000", trusted)

    def test_scenario_o_incomplete_matrix_deterministic(self):
        frame = _vendor_frame(criteria=[
            DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
            DecisionCriterion(label="Reliability", origin="explicit"),
        ])
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], priority="primary"),
        ])
        pre = run_pre_check(frame, oe)
        assert not pre.matrix_complete

    @pytest.mark.asyncio
    async def test_scenario_o_incomplete_returns_insufficient_without_llm(self):
        frame = _vendor_frame(criteria=[
            DecisionCriterion(label="Cost", origin="explicit"),
            DecisionCriterion(label="Reliability", origin="explicit"),
        ])
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1]),
        ])
        synthesis, metrics = await synthesize_decision(frame, oe, _ks(), [], llm=MagicMock())
        assert synthesis.recommendation_status == RecommendationStatus.INSUFFICIENT_BASIS
        assert synthesis.recommended_option is None
        assert metrics.synthesis_llm_calls == 0

    @pytest.mark.asyncio
    async def test_scenario_p_skip_without_option_evaluation(self):
        state = {"decision_frame": {}, "option_evaluation": None, "cost_metrics": {}}
        result = await decision_synthesizer_node(state)
        assert result["decision_synthesis"] is None
        assert result["decision_synthesis_metrics"]["synthesis_skipped"]

    def test_scenario_q_invented_option_rejected(self):
        frame = _vendor_frame()
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], priority="primary"),
        ])
        llm = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor C",
            rationale="Invented.",
        )
        synthesis, _, _ = _validate(frame, oe, llm)
        assert synthesis.recommended_option is None

    def test_scenario_r_validator_never_upgrades_status(self):
        frame = _vendor_frame(constraints=["Must integrate with Salesforce"])
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], priority="primary"),
        ], constraints=frame.constraints)
        llm = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor A",
            rationale="Should be capped.",
            constraint_assessments=[
                ConstraintAssessmentLLM(option_label="Vendor A", constraint="Must integrate with Salesforce",
                                      compliance=ConstraintCompliance.NOT_ESTABLISHED, claim_ids=[], reason="Unknown"),
                ConstraintAssessmentLLM(option_label="Vendor B", constraint="Must integrate with Salesforce",
                                      compliance=ConstraintCompliance.NOT_ESTABLISHED, claim_ids=[], reason="Unknown"),
            ],
        )
        synthesis, _, _ = _validate(frame, oe, llm)
        assert synthesis.recommendation_status == RecommendationStatus.TENTATIVE_RECOMMENDATION


class TestIncompleteConstraintMatrix:
    def test_scenario_15_incomplete_constraint_matrix_forces_insufficient(self):
        frame = _vendor_frame(constraints=["Budget under $20,000", "Must integrate with Salesforce"])
        oe = _build_oe("Which vendor?", [
            _oe_row("Vendor A", "Cost", CriterionAssessment.FAVORABLE, KnowledgeCoverage.GROUNDED, [1], priority="primary"),
            _oe_row("Vendor B", "Cost", CriterionAssessment.UNFAVORABLE, KnowledgeCoverage.GROUNDED, [2], priority="primary"),
        ], constraints=frame.constraints)
        llm = _llm_out(
            recommendation_status=RecommendationStatus.RECOMMEND,
            recommended_option="Vendor A",
            rationale="Strong cost position.",
            constraint_assessments=[
                ConstraintAssessmentLLM(
                    option_label="Vendor A",
                    constraint="Budget under $20,000",
                    compliance=ConstraintCompliance.SATISFIED,
                    claim_ids=[1],
                    reason="Within budget.",
                ),
            ],
        )
        synthesis, _, errors = _validate(frame, oe, llm)
        assert synthesis.recommendation_status == RecommendationStatus.INSUFFICIENT_BASIS
        assert synthesis.recommended_option is None
        assert any("missing_constraint_pairs" in e for e in errors)


class TestConstraintEpistemics:
    def test_disputed_claim_cannot_establish_compliance(self):
        catalog = build_claim_catalog(_ks(disputed=[_entry(10, "disputed")]), [_claim(10, "maybe integrates")])
        result = _downgrade_constraint_compliance(
            ConstraintCompliance.SATISFIED, [10], catalog
        )
        assert result == ConstraintCompliance.NOT_ESTABLISHED


class TestGraphPlacement:
    def test_graph_has_decision_synthesizer_node(self):
        assert "decision_synthesizer" in create_graph().nodes

    def test_cap_status_never_upgrades(self):
        assert _cap_status(
            RecommendationStatus.TENTATIVE_RECOMMENDATION,
            RecommendationStatus.RECOMMEND,
        ) == RecommendationStatus.TENTATIVE_RECOMMENDATION
