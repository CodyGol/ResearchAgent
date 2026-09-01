"""Tests for Phase 3B evidence-grounded option evaluation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.models import (
    Claim,
    ClaimType,
    EvidenceConfidence,
    KnowledgeCategory,
    VerificationStatus,
)
from graph import create_graph
from nodes.option_evaluator import option_evaluator_node
from services.decision_framing_schemas import (
    DecisionCriterion,
    DecisionFrame,
    DecisionOption,
    DecisionType,
)
from services.knowledge_state_schemas import KnowledgeState, KnowledgeStateEntry
from services.option_evaluation import (
    _downgrade_assessment,
    _has_recommendation_leakage,
    build_claim_catalog,
    evaluate_options,
    validate_and_build_evaluation,
)
from services.option_evaluation_schemas import (
    ClaimCatalogEntry,
    CriterionAssessment,
    CriterionEvaluationLLM,
    KnowledgeCoverage,
    OptionEvaluationLLMOutput,
)


def _claim(cid: int, text: str) -> Claim:
    return Claim(id=cid, research_run_id=1, text=text, claim_type=ClaimType.FACTUAL)


def _entry(
    cid: int,
    *,
    bucket: KnowledgeCategory = KnowledgeCategory.KNOWN,
    status: VerificationStatus = VerificationStatus.SUPPORTED,
    vid: int | None = None,
) -> KnowledgeStateEntry:
    return KnowledgeStateEntry(
        claim_id=cid,
        verification_id=vid or cid * 10,
        knowledge_category=bucket,
        verification_status=status,
        confidence=EvidenceConfidence.HIGH,
        evidence_ids=[cid * 100],
    )


def _knowledge_state(**buckets: list[KnowledgeStateEntry]) -> KnowledgeState:
    return KnowledgeState(
        known=buckets.get("known", []),
        likely=buckets.get("likely", []),
        disputed=buckets.get("disputed", []),
        unknown=buckets.get("unknown", []),
        contradicted=buckets.get("contradicted", []),
        unverifiable=buckets.get("unverifiable", []),
    )


def _vendor_frame() -> DecisionFrame:
    return DecisionFrame(
        decision="Which LLM provider to use",
        decision_type=DecisionType.VENDOR_SELECTION,
        options=[
            DecisionOption(label="OpenAI", origin="explicit"),
            DecisionOption(label="Anthropic", origin="explicit"),
        ],
        criteria=[
            DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
            DecisionCriterion(label="Enterprise readiness", origin="inferred"),
        ],
    )


def _empty_options_frame() -> DecisionFrame:
    return DecisionFrame(
        decision="Which CRM to choose",
        decision_type=DecisionType.VENDOR_SELECTION,
        options=[],
        criteria=[DecisionCriterion(label="Fit for requirements", origin="inferred")],
        constraints=["Budget under $20,000 per year"],
    )


def _binary_frame() -> DecisionFrame:
    return DecisionFrame(
        decision="Whether to enter Market X",
        decision_type=DecisionType.MARKET_ENTRY,
        options=[
            DecisionOption(label="Enter Market X", origin="implied"),
            DecisionOption(label="Do not Enter Market X", origin="implied"),
        ],
        criteria=[DecisionCriterion(label="Market opportunity", origin="inferred")],
    )


def _mock_llm(output: OptionEvaluationLLMOutput) -> MagicMock:
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=output)
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


class TestEmptyOptions:
    @pytest.mark.asyncio
    async def test_skip_when_no_concrete_options(self):
        frame = _empty_options_frame()
        ks = _knowledge_state(known=[_entry(1)])
        claims = [_claim(1, "Some CRM fact")]

        evaluation, metrics = await evaluate_options(frame, ks, claims, llm=_mock_llm(
            OptionEvaluationLLMOutput()
        ))

        assert evaluation is None
        assert metrics.evaluation_skipped is True
        assert metrics.evaluation_skipped_reason == "no_concrete_options"
        assert metrics.evaluation_failed is False
        assert metrics.evaluation_llm_calls == 0

    @pytest.mark.asyncio
    async def test_node_skips_without_placeholder(self):
        frame = _empty_options_frame()
        ks = _knowledge_state(known=[_entry(1)])
        state = {
            "decision_frame": frame.model_dump(mode="json"),
            "knowledge_state": ks.model_dump(mode="json"),
            "material_claims": [_claim(1, "fact")],
            "cost_metrics": {},
        }
        result = await option_evaluator_node(state)
        assert result["option_evaluation"] is None
        assert result["option_evaluation_metrics"]["evaluation_skipped"] is True
        assert result["option_evaluation_metrics"]["evaluation_skipped_reason"] == "no_concrete_options"
        assert result["option_evaluation_metrics"]["evaluation_failed"] is False


class TestClaimCatalog:
    def test_full_epistemic_catalog_includes_all_buckets(self):
        ks = _knowledge_state(
            known=[_entry(1)],
            likely=[_entry(2)],
            disputed=[_entry(3)],
            unknown=[_entry(4)],
            contradicted=[_entry(5)],
            unverifiable=[_entry(6)],
        )
        claims = [_claim(i, f"claim {i}") for i in range(1, 7)]
        catalog = build_claim_catalog(ks, claims)

        assert len(catalog) == 6
        assert catalog[1].bucket == "known"
        assert catalog[3].bucket == "disputed"
        assert catalog[5].bucket == "contradicted"
        assert catalog[6].bucket == "unverifiable"
        assert catalog[1].claim_text == "claim 1"


class TestEpistemicGuardrails:
    def test_disputed_cannot_support_confident_directional(self):
        catalog = {
            3: ClaimCatalogEntry(
                claim_id=3,
                bucket="disputed",
                verification_id=30,
                verification_status="uncertain",
                knowledge_category="disputed",
                claim_text="Disputed pricing",
            )
        }
        result = _downgrade_assessment(
            CriterionAssessment.FAVORABLE, [3], catalog
        )
        assert result == CriterionAssessment.UNCERTAIN

    def test_contradicted_cannot_support_directional(self):
        catalog = {
            5: ClaimCatalogEntry(
                claim_id=5,
                bucket="contradicted",
                verification_id=50,
                verification_status="contradicted",
                knowledge_category=None,
                claim_text="Conflicting data",
            )
        }
        result = _downgrade_assessment(
            CriterionAssessment.UNFAVORABLE, [5], catalog
        )
        assert result == CriterionAssessment.INSUFFICIENT_INFORMATION

    def test_contradicted_only_uncertain_normalized_to_insufficient(self):
        """Regression: contradicted-only citations must not remain uncertain/partial."""
        frame = DecisionFrame(
            decision="Whether to enter Market X",
            decision_type=DecisionType.MARKET_ENTRY,
            options=[DecisionOption(label="Enter Market X", origin="implied")],
            criteria=[DecisionCriterion(label="Market growth", origin="inferred")],
        )
        ks = _knowledge_state(contradicted=[_entry(401, vid=4010)])
        catalog = build_claim_catalog(ks, [_claim(401, "Market X growth estimates conflict across sources.")])
        llm_out = OptionEvaluationLLMOutput(
            evaluations=[
                CriterionEvaluationLLM(
                    option_label="Enter Market X",
                    criterion_label="Market growth",
                    assessment=CriterionAssessment.UNCERTAIN,
                    claim_ids=[401],
                    reason="Growth estimates conflict across sources.",
                ),
            ]
        )
        evaluation, _ = validate_and_build_evaluation(llm_out, frame, catalog)
        assert evaluation is not None
        row = evaluation.option_evaluations[0].criteria_evaluations[0]
        assert row.assessment == CriterionAssessment.INSUFFICIENT_INFORMATION
        assert row.knowledge_coverage == KnowledgeCoverage.INSUFFICIENT
        assert row.claim_ids == [401]

    def test_known_can_support_directional(self):
        catalog = {
            1: ClaimCatalogEntry(
                claim_id=1,
                bucket="known",
                verification_id=10,
                verification_status="supported",
                knowledge_category="known",
                claim_text="Verified fact",
            )
        }
        result = _downgrade_assessment(
            CriterionAssessment.FAVORABLE, [1], catalog
        )
        assert result == CriterionAssessment.FAVORABLE

    def test_contradicted_in_catalog_but_downgraded_in_validation(self):
        frame = _vendor_frame()
        ks = _knowledge_state(
            known=[_entry(1)],
            contradicted=[_entry(5)],
        )
        catalog = build_claim_catalog(ks, [_claim(1, "known"), _claim(5, "contradicted")])

        llm_out = OptionEvaluationLLMOutput(
            evaluations=[
                CriterionEvaluationLLM(
                    option_label="OpenAI",
                    criterion_label="Cost",
                    assessment=CriterionAssessment.FAVORABLE,
                    claim_ids=[5],
                    reason="Contradicted claim alone suggests low cost.",
                ),
            ]
        )
        evaluation, metrics = validate_and_build_evaluation(llm_out, frame, catalog)
        assert evaluation is not None
        row = evaluation.option_evaluations[0].criteria_evaluations[0]
        assert row.assessment != CriterionAssessment.FAVORABLE
        assert 5 in catalog


class TestNoBinaryInversion:
    def test_validator_does_not_mirror_binary_assessments(self):
        frame = _binary_frame()
        catalog = build_claim_catalog(
            _knowledge_state(known=[_entry(1)]),
            [_claim(1, "Market growing")],
        )
        llm_out = OptionEvaluationLLMOutput(
            evaluations=[
                CriterionEvaluationLLM(
                    option_label="Enter Market X",
                    criterion_label="Market opportunity",
                    assessment=CriterionAssessment.FAVORABLE,
                    claim_ids=[1],
                    reason="Growth supports entry.",
                ),
            ]
        )
        evaluation, _ = validate_and_build_evaluation(llm_out, frame, catalog)
        assert evaluation is not None
        assert len(evaluation.option_evaluations) == 1
        assert evaluation.option_evaluations[0].option_label == "Enter Market X"
        labels = {e.option_label for e in evaluation.option_evaluations}
        assert "Do not Enter Market X" not in labels


class TestRecommendationLeakage:
    def test_detects_recommendation_language(self):
        assert _has_recommendation_leakage("We recommend OpenAI for cost reasons.")
        assert _has_recommendation_leakage("The best choice is Anthropic.")
        assert not _has_recommendation_leakage("Claims suggest lower API pricing.")

    def test_leakage_row_rejected_not_rewritten(self):
        frame = _vendor_frame()
        catalog = build_claim_catalog(
            _knowledge_state(known=[_entry(1)]),
            [_claim(1, "OpenAI pricing")],
        )
        original_reason = "We recommend OpenAI because it is the best choice."
        llm_out = OptionEvaluationLLMOutput(
            evaluations=[
                CriterionEvaluationLLM(
                    option_label="OpenAI",
                    criterion_label="Cost",
                    assessment=CriterionAssessment.FAVORABLE,
                    claim_ids=[1],
                    reason=original_reason,
                ),
            ]
        )
        evaluation, metrics = validate_and_build_evaluation(llm_out, frame, catalog)
        assert evaluation is None
        assert metrics.evaluation_failed is True
        assert metrics.failure_reason == "recommendation_leakage"
        assert metrics.rejected_row_count == 1

    def test_valid_row_reason_preserved_unchanged(self):
        frame = _vendor_frame()
        catalog = build_claim_catalog(
            _knowledge_state(known=[_entry(1)]),
            [_claim(1, "OpenAI pricing")],
        )
        reason = "Claim 1 indicates competitive API pricing for OpenAI."
        llm_out = OptionEvaluationLLMOutput(
            evaluations=[
                CriterionEvaluationLLM(
                    option_label="OpenAI",
                    criterion_label="Cost",
                    assessment=CriterionAssessment.FAVORABLE,
                    claim_ids=[1],
                    reason=reason,
                ),
            ]
        )
        evaluation, _ = validate_and_build_evaluation(llm_out, frame, catalog)
        assert evaluation is not None
        assert evaluation.option_evaluations[0].criteria_evaluations[0].reason == reason


class TestProvenanceAndLineage:
    def test_provenance_copied_from_decision_frame(self):
        frame = _vendor_frame()
        catalog = build_claim_catalog(
            _knowledge_state(known=[_entry(1, vid=101)]),
            [_claim(1, "fact")],
        )
        llm_out = OptionEvaluationLLMOutput(
            evaluations=[
                CriterionEvaluationLLM(
                    option_label="OpenAI",
                    criterion_label="Cost",
                    assessment=CriterionAssessment.FAVORABLE,
                    claim_ids=[1],
                    reason="Supported by verified pricing claim.",
                ),
            ]
        )
        evaluation, _ = validate_and_build_evaluation(llm_out, frame, catalog)
        assert evaluation is not None
        opt = evaluation.option_evaluations[0]
        crit = opt.criteria_evaluations[0]
        assert opt.option_origin == "explicit"
        assert crit.criterion_origin == "explicit"
        assert crit.criterion_priority == "primary"
        assert crit.verification_ids == [101]
        assert crit.knowledge_categories == ["known"]
        assert crit.knowledge_coverage == KnowledgeCoverage.GROUNDED

    def test_criterion_priority_copied_from_frame_not_llm(self):
        frame = DecisionFrame(
            decision="Which vendor",
            decision_type=DecisionType.VENDOR_SELECTION,
            options=[DecisionOption(label="Vendor A", origin="explicit")],
            criteria=[
                DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
            ],
        )
        catalog = build_claim_catalog(
            _knowledge_state(known=[_entry(1)]),
            [_claim(1, "Low cost")],
        )
        llm_out = OptionEvaluationLLMOutput(
            evaluations=[
                CriterionEvaluationLLM(
                    option_label="Vendor A",
                    criterion_label="Cost",
                    assessment=CriterionAssessment.FAVORABLE,
                    claim_ids=[1],
                    reason="Cost is favorable.",
                ),
            ]
        )
        evaluation, _ = validate_and_build_evaluation(llm_out, frame, catalog)
        assert evaluation is not None
        row = evaluation.option_evaluations[0].criteria_evaluations[0]
        assert row.criterion_priority == "primary"

    def test_invalid_claim_ids_stripped(self):
        frame = _vendor_frame()
        catalog = build_claim_catalog(
            _knowledge_state(known=[_entry(1)]),
            [_claim(1, "fact")],
        )
        llm_out = OptionEvaluationLLMOutput(
            evaluations=[
                CriterionEvaluationLLM(
                    option_label="OpenAI",
                    criterion_label="Cost",
                    assessment=CriterionAssessment.FAVORABLE,
                    claim_ids=[1, 999],
                    reason="Mixed valid and invalid references.",
                ),
            ]
        )
        _, metrics = validate_and_build_evaluation(llm_out, frame, catalog)
        assert metrics.invalid_reference_count == 1

    def test_unknown_option_or_criterion_rejected(self):
        frame = _vendor_frame()
        catalog = build_claim_catalog(
            _knowledge_state(known=[_entry(1)]),
            [_claim(1, "fact")],
        )
        llm_out = OptionEvaluationLLMOutput(
            evaluations=[
                CriterionEvaluationLLM(
                    option_label="Google",
                    criterion_label="Cost",
                    assessment=CriterionAssessment.NEUTRAL,
                    claim_ids=[1],
                    reason="Invented option.",
                ),
            ]
        )
        evaluation, metrics = validate_and_build_evaluation(llm_out, frame, catalog)
        assert evaluation is None
        assert metrics.rejected_row_count == 1


class TestEvaluateOptionsIntegration:
    @pytest.mark.asyncio
    async def test_successful_evaluation_mocked(self):
        frame = _vendor_frame()
        ks = _knowledge_state(known=[_entry(1), _entry(2, vid=20)])
        claims = [_claim(1, "OpenAI has enterprise SSO"), _claim(2, "Anthropic SOC2")]
        llm_out = OptionEvaluationLLMOutput(
            evaluations=[
                CriterionEvaluationLLM(
                    option_label="OpenAI",
                    criterion_label="Cost",
                    assessment=CriterionAssessment.FAVORABLE,
                    claim_ids=[1],
                    reason="Enterprise features verified.",
                ),
                CriterionEvaluationLLM(
                    option_label="Anthropic",
                    criterion_label="Enterprise readiness",
                    assessment=CriterionAssessment.FAVORABLE,
                    claim_ids=[2],
                    reason="SOC2 compliance verified.",
                ),
            ]
        )
        evaluation, metrics = await evaluate_options(
            frame, ks, claims, llm=_mock_llm(llm_out)
        )
        assert evaluation is not None
        assert evaluation.decision == frame.decision
        assert metrics.evaluation_failed is False
        assert metrics.evaluation_llm_calls == 1
        assert metrics.catalog_claim_count == 2
        assert "metrics" not in evaluation.model_dump()

    @pytest.mark.asyncio
    async def test_fail_open_on_llm_error(self):
        frame = _vendor_frame()
        ks = _knowledge_state(known=[_entry(1)])
        claims = [_claim(1, "fact")]
        llm = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
        llm.with_structured_output = MagicMock(return_value=structured)

        evaluation, metrics = await evaluate_options(frame, ks, claims, llm=llm)
        assert evaluation is None
        assert metrics.evaluation_failed is True
        assert "LLM down" in (metrics.failure_reason or "")


class TestGraphPlacement:
    def _route_after_knowledge_state(self, state: dict) -> str:
        frame = state.get("decision_frame")
        ks = state.get("knowledge_state")
        if not frame or not ks:
            return "writer"
        options = frame.get("options") or []
        if not options:
            return "writer"
        return "option_evaluator"

    def test_graph_contains_option_evaluator_node(self):
        graph = create_graph()
        assert "option_evaluator" in graph.nodes

    def test_route_to_evaluator_when_frame_and_options(self):
        state = {
            "decision_frame": {"options": [{"label": "A", "origin": "explicit"}]},
            "knowledge_state": {"known": []},
        }
        assert self._route_after_knowledge_state(state) == "option_evaluator"

    def test_route_skips_without_options(self):
        state = {
            "decision_frame": {"options": []},
            "knowledge_state": {"known": []},
        }
        assert self._route_after_knowledge_state(state) == "writer"

    def test_route_skips_without_decision_frame(self):
        state = {"knowledge_state": {"known": []}}
        assert self._route_after_knowledge_state(state) == "writer"

    def test_graph_routes_through_decision_synthesizer(self):
        graph = create_graph()
        assert "decision_synthesizer" in graph.nodes
        assert ("decision_synthesizer", "writer") in graph.edges
