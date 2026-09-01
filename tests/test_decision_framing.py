"""Tests for Phase 3A decision framing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graph import create_graph
from nodes.decision_framer import decision_framer_node
from services.decision_framing import _sanitize_time_horizon, _validate_frame, frame_decision_query
from services.decision_framing_schemas import (
    DecisionCriterion,
    DecisionFrame,
    DecisionFramingResult,
    DecisionOption,
    DecisionType,
)


def _mock_llm(result: DecisionFramingResult) -> MagicMock:
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=result)
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured)
    return llm


def _vendor_cost_fixture() -> DecisionFramingResult:
    return DecisionFramingResult(
        decision_oriented=True,
        decision_frame=DecisionFrame(
            decision="Which LLM provider to use for our enterprise support agent",
            decision_type=DecisionType.VENDOR_SELECTION,
            options=[
                DecisionOption(label="OpenAI", origin="explicit"),
                DecisionOption(label="Anthropic", origin="explicit"),
            ],
            criteria=[
                DecisionCriterion(label="Cost", origin="explicit", priority="primary"),
                DecisionCriterion(label="Enterprise readiness", origin="inferred"),
                DecisionCriterion(label="Integration complexity", origin="inferred"),
            ],
        ),
        detection_rationale="User asks which provider to use between named options",
    )


def _acquisition_fixture() -> DecisionFramingResult:
    return DecisionFramingResult(
        decision_oriented=True,
        decision_frame=DecisionFrame(
            decision="Whether to acquire Company X",
            decision_type=DecisionType.ACQUISITION,
            options=[
                DecisionOption(label="Acquire Company X", origin="implied"),
                DecisionOption(label="Do not acquire Company X", origin="implied"),
            ],
            criteria=[
                DecisionCriterion(label="Strategic fit", origin="inferred"),
                DecisionCriterion(label="Financial impact", origin="inferred"),
            ],
            constraints=[],
            missing_decision_context=[
                "Acceptable acquisition price",
                "Strategic objective",
                "Integration capacity",
            ],
            explicit_assumptions=[],
        ),
    )


def _market_entry_fixture() -> DecisionFramingResult:
    return DecisionFramingResult(
        decision_oriented=True,
        decision_frame=DecisionFrame(
            decision="Whether to enter the Japanese EV charging market",
            decision_type=DecisionType.MARKET_ENTRY,
            options=[
                DecisionOption(label="Enter the Japanese EV charging market", origin="implied"),
                DecisionOption(label="Do not enter the Japanese EV charging market", origin="implied"),
            ],
            criteria=[
                DecisionCriterion(label="Market opportunity", origin="inferred"),
                DecisionCriterion(label="Regulatory exposure", origin="inferred"),
                DecisionCriterion(label="Competitive landscape", origin="inferred"),
            ],
        ),
    )


def _crm_constraints_fixture() -> DecisionFramingResult:
    return DecisionFramingResult(
        decision_oriented=True,
        decision_frame=DecisionFrame(
            decision="Which CRM to choose",
            decision_type=DecisionType.VENDOR_SELECTION,
            options=[],
            criteria=[
                DecisionCriterion(label="Fit for requirements", origin="inferred"),
            ],
            constraints=[
                "Budget under $20,000 per year",
                "Must integrate with Salesforce",
            ],
        ),
    )


def _non_decision_fixture() -> DecisionFramingResult:
    return DecisionFramingResult(
        decision_oriented=False,
        decision_frame=None,
        detection_rationale="Comparison without a choice request",
    )


class TestProvenance:
    @pytest.mark.asyncio
    async def test_vendor_explicit_options_and_cost_criterion(self):
        frame, metrics = await frame_decision_query(
            "Should we use OpenAI or Anthropic? Cost is our most important consideration.",
            llm=_mock_llm(_vendor_cost_fixture()),
        )
        assert frame is not None
        assert [o.label for o in frame.options] == ["OpenAI", "Anthropic"]
        assert all(o.origin == "explicit" for o in frame.options)

        explicit_criteria = [c for c in frame.criteria if c.origin == "explicit"]
        inferred_criteria = [c for c in frame.criteria if c.origin == "inferred"]
        assert any(c.label == "Cost" for c in explicit_criteria)
        cost = next(c for c in frame.criteria if c.label == "Cost")
        assert cost.priority == "primary"
        assert all(c.priority == "standard" for c in inferred_criteria)
        assert len(inferred_criteria) >= 1
        assert metrics.explicit_criterion_count == 1
        assert metrics.inferred_criterion_count >= 1

    @pytest.mark.asyncio
    async def test_acquisition_implied_options_no_inferred_constraints(self):
        frame, metrics = await frame_decision_query(
            "Should we acquire Company X?",
            llm=_mock_llm(_acquisition_fixture()),
        )
        assert frame is not None
        assert all(o.origin == "implied" for o in frame.options)
        assert len(frame.options) == 2
        assert frame.constraints == []
        assert frame.explicit_assumptions == []
        assert len(frame.missing_decision_context) >= 1
        assert all(c.origin == "inferred" for c in frame.criteria)
        assert metrics.implied_option_count == 2
        assert metrics.constraint_count == 0

    def test_explicit_criteria_not_marked_inferred(self):
        frame = _validate_frame(_vendor_cost_fixture())
        assert frame is not None
        cost = next(c for c in frame.criteria if c.label == "Cost")
        assert cost.origin == "explicit"


class TestCriterionPriority:
    @pytest.mark.asyncio
    async def test_cost_most_important_is_primary(self):
        frame, _ = await frame_decision_query(
            "Should we use OpenAI or Anthropic? Cost is our most important consideration.",
            llm=_mock_llm(_vendor_cost_fixture()),
        )
        assert frame is not None
        cost = next(c for c in frame.criteria if c.label == "Cost")
        assert cost.origin == "explicit"
        assert cost.priority == "primary"
        assert all(c.priority == "standard" for c in frame.criteria if c.origin == "inferred")

    @pytest.mark.asyncio
    async def test_compare_without_priority_both_standard(self):
        fixture = DecisionFramingResult(
            decision_oriented=True,
            decision_frame=DecisionFrame(
                decision="Which LLM provider to use",
                decision_type=DecisionType.VENDOR_SELECTION,
                options=[
                    DecisionOption(label="OpenAI", origin="explicit"),
                    DecisionOption(label="Anthropic", origin="explicit"),
                ],
                criteria=[
                    DecisionCriterion(label="Cost", origin="explicit"),
                    DecisionCriterion(label="Reliability", origin="explicit"),
                ],
            ),
        )
        frame, _ = await frame_decision_query(
            "Compare them on cost and reliability.",
            llm=_mock_llm(fixture),
        )
        assert frame is not None
        for crit in frame.criteria:
            assert crit.origin == "explicit"
            assert crit.priority == "standard"

    def test_inferred_primary_forced_to_standard(self):
        result = DecisionFramingResult(
            decision_oriented=True,
            decision_frame=DecisionFrame(
                decision="Whether to enter Market X",
                decision_type=DecisionType.MARKET_ENTRY,
                criteria=[
                    DecisionCriterion(
                        label="Strategic fit",
                        origin="inferred",
                        priority="primary",
                    ),
                ],
            ),
        )
        frame = _validate_frame(result)
        assert frame is not None
        assert frame.criteria[0].priority == "standard"


class TestScenarios:
    @pytest.mark.asyncio
    async def test_market_entry(self):
        frame, metrics = await frame_decision_query(
            "Should our company enter the Japanese EV charging market?",
            llm=_mock_llm(_market_entry_fixture()),
        )
        assert frame is not None
        assert frame.decision_type == DecisionType.MARKET_ENTRY
        assert metrics.decision_detected is True
        assert "recommend" not in frame.decision.lower()

    @pytest.mark.asyncio
    async def test_crm_constraints_preserved(self):
        frame, _ = await frame_decision_query(
            "Which CRM should we choose if our budget is under $20,000 per year and we need Salesforce integration?",
            llm=_mock_llm(_crm_constraints_fixture()),
        )
        assert frame is not None
        assert any("$20,000" in c for c in frame.constraints)
        assert any("Salesforce" in c for c in frame.constraints)
        assert frame.time_horizon is None

    @pytest.mark.asyncio
    async def test_crm_budget_per_year_not_time_horizon(self):
        """Regression: budget cadence 'per year' is not a decision horizon."""
        llm_result = DecisionFramingResult(
            decision_oriented=True,
            decision_frame=DecisionFrame(
                decision="Which CRM system to choose",
                decision_type=DecisionType.VENDOR_SELECTION,
                constraints=[
                    "Budget must be under $20,000 per year",
                    "Must integrate with Salesforce",
                ],
                time_horizon="per year",
            ),
        )
        frame = _validate_frame(llm_result)
        assert frame is not None
        assert frame.time_horizon is None
        assert any("$20,000" in c for c in frame.constraints)
        assert any("Salesforce" in c for c in frame.constraints)

    @pytest.mark.asyncio
    async def test_europe_decision_horizon_preserved(self):
        llm_result = DecisionFramingResult(
            decision_oriented=True,
            decision_frame=DecisionFrame(
                decision="Whether to expand into Europe over the next three years",
                decision_type=DecisionType.MARKET_ENTRY,
                time_horizon="next three years",
                explicit_assumptions=["Interest rates stay above 4%"],
            ),
        )
        frame = _validate_frame(llm_result)
        assert frame is not None
        assert frame.time_horizon == "next three years"

    def test_sanitize_time_horizon_metric_cadence(self):
        assert _sanitize_time_horizon("per year") is None
        assert _sanitize_time_horizon("per month") is None
        assert _sanitize_time_horizon("annual subscription") is None
        assert _sanitize_time_horizon("next three years") == "next three years"

    @pytest.mark.asyncio
    async def test_non_decision_comparison(self):
        frame, metrics = await frame_decision_query(
            "Compare OpenAI and Anthropic.",
            llm=_mock_llm(_non_decision_fixture()),
        )
        assert frame is None
        assert metrics.decision_detected is False

    @pytest.mark.asyncio
    async def test_fail_open_on_llm_error(self):
        llm = MagicMock()
        llm.with_structured_output = MagicMock(side_effect=RuntimeError("llm down"))
        frame, metrics = await frame_decision_query("Should we acquire X?", llm=llm)
        assert frame is None
        assert metrics.framing_failed is True

    @pytest.mark.asyncio
    async def test_fail_open_on_invalid_frame(self):
        bad = DecisionFramingResult(
            decision_oriented=True,
            decision_frame=DecisionFrame(decision="   ", decision_type=DecisionType.OTHER),
        )
        frame, metrics = await frame_decision_query("Should we?", llm=_mock_llm(bad))
        assert frame is None
        assert metrics.framing_failed is True


class TestGraphRouting:
    def _route_after_router(self, state: dict) -> str:
        classification = state.get("query_classification") or {}
        route = classification.get("route", "standard")
        if route == "simple_fact" and not state.get("escalated_from_fast_path"):
            return "fast_path"
        return "decision_framer"

    def test_simple_fact_to_fast_path(self):
        state = {"query_classification": {"route": "simple_fact"}}
        assert self._route_after_router(state) == "fast_path"

    def test_standard_to_decision_framer(self):
        state = {"query_classification": {"route": "standard"}}
        assert self._route_after_router(state) == "decision_framer"

    def test_graph_has_decision_framer_node(self):
        graph = create_graph()
        assert "decision_framer" in graph.nodes


class TestDecisionFramerNode:
    @pytest.mark.asyncio
    async def test_node_writes_state_and_continues(self):
        frame = _validate_frame(_vendor_cost_fixture())
        from services.decision_framing_schemas import DecisionFramingMetrics

        m = DecisionFramingMetrics(
            decision_detected=True,
            decision_type="vendor_selection",
            option_count=2,
            criteria_count=3,
            framing_llm_calls=1,
            framing_time_ms=1.0,
        )
        state = {"user_query": "test", "cost_metrics": {}}
        with patch(
            "nodes.decision_framer.frame_decision_query",
            new=AsyncMock(return_value=(frame, m)),
        ):
            result = await decision_framer_node(state)

        assert result["decision_frame"] is not None
        assert result["decision_frame"]["options"][0]["origin"] == "explicit"
        assert result["current_node"] == "planner"
        assert result["decision_frame_metrics"]["decision_detected"] is True

    @pytest.mark.asyncio
    async def test_node_fail_open(self):
        from services.decision_framing_schemas import DecisionFramingMetrics

        m = DecisionFramingMetrics(framing_failed=True, failure_reason="err")
        state = {"user_query": "test", "cost_metrics": {}}
        with patch(
            "nodes.decision_framer.frame_decision_query",
            new=AsyncMock(return_value=(None, m)),
        ):
            result = await decision_framer_node(state)
        assert result["decision_frame"] is None
        assert result["current_node"] == "planner"
