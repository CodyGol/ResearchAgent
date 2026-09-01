"""Tests for query complexity classification and research budgets."""

import pytest

from services.query_router import (
    BUDGETS,
    QueryComplexity,
    classify_query,
)


class TestSimpleClassification:
    def test_capital_of_japan(self):
        result = classify_query("What is the capital of Japan?")
        assert result.complexity.value == "simple"
        assert result.route.value == "simple_fact"
        assert result.direct_answer_expected is True
        assert result.research_budget.max_search_queries == 1

    def test_f1_winner(self):
        result = classify_query("Who won the 2023 F1 World Championship?")
        assert result.route.value == "simple_fact"

    def test_apple_revenue(self):
        result = classify_query("What was Apple's revenue in fiscal 2025?")
        assert result.route.value == "simple_fact"


class TestDeepClassification:
    def test_acquisition_decision(self):
        result = classify_query("Should a company acquire Company X?")
        assert result.complexity == QueryComplexity.DEEP
        assert result.research_budget.max_iterations >= 2

    def test_strategic_risks(self):
        result = classify_query(
            "What are the strategic risks facing the semiconductor industry over five years?"
        )
        assert result.complexity == QueryComplexity.DEEP


class TestStandardClassification:
    def test_comparison(self):
        result = classify_query(
            "Compare OpenAI and Anthropic enterprise offerings."
        )
        assert result.complexity in (
            QueryComplexity.STANDARD,
            QueryComplexity.DEEP,
        )


class TestBudgets:
    def test_simple_budget_smaller_than_deep(self):
        simple = BUDGETS[QueryComplexity.SIMPLE]
        deep = BUDGETS[QueryComplexity.DEEP]
        assert simple.max_search_queries < deep.max_search_queries
        assert simple.target_sources < deep.target_sources
        assert simple.max_evidence_items < (deep.max_evidence_items or 999)

    def test_budget_has_claim_depth(self):
        assert BUDGETS[QueryComplexity.SIMPLE].claim_depth.value == "minimal"
