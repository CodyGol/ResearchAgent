"""Schemas for Phase 3A decision framing."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    MARKET_ENTRY = "market_entry"
    VENDOR_SELECTION = "vendor_selection"
    ACQUISITION = "acquisition"
    INVESTMENT = "investment"
    BUY_VS_NOT_BUY = "buy_vs_not_buy"
    HIRING = "hiring"
    PRIORITIZATION = "prioritization"
    PRODUCT_STRATEGY = "product_strategy"
    OTHER = "other"


class DecisionOption(BaseModel):
    """An alternative under consideration with provenance."""

    label: str = Field(..., min_length=1)
    origin: Literal["explicit", "implied"]


class DecisionCriterion(BaseModel):
    """An evaluation dimension with provenance."""

    label: str = Field(..., min_length=1)
    origin: Literal["explicit", "inferred"]


class DecisionFrame(BaseModel):
    """Run-scoped decision structure extracted from the user query."""

    decision: str = Field(..., min_length=1, description="What must be decided")
    decision_type: DecisionType = DecisionType.OTHER
    options: list[DecisionOption] = Field(default_factory=list)
    criteria: list[DecisionCriterion] = Field(default_factory=list)
    constraints: list[str] = Field(
        default_factory=list,
        description="Hard requirements explicitly stated by the user only",
    )
    time_horizon: str | None = Field(
        None,
        description=(
            "Period over which the user intends to make or evaluate the decision "
            "(e.g. 'over the next three years'). Not metric/pricing cadence such as "
            "'per year' or 'monthly subscription' — those belong in constraints."
        ),
    )
    missing_decision_context: list[str] = Field(
        default_factory=list,
        description="Unspecified decision inputs — not factual research unknowns",
    )
    explicit_assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions explicitly stated by the user only",
    )


class DecisionFramingResult(BaseModel):
    """Structured LLM output: detection + optional frame."""

    decision_oriented: bool
    decision_frame: DecisionFrame | None = None
    detection_rationale: str | None = None


@dataclass
class DecisionFramingMetrics:
    """Observability for decision framing."""

    decision_detected: bool = False
    decision_type: str | None = None
    option_count: int = 0
    explicit_option_count: int = 0
    implied_option_count: int = 0
    criteria_count: int = 0
    explicit_criterion_count: int = 0
    inferred_criterion_count: int = 0
    constraint_count: int = 0
    missing_context_count: int = 0
    explicit_assumption_count: int = 0
    framing_llm_calls: int = 0
    framing_time_ms: float = 0.0
    framing_failed: bool = False
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_detected": self.decision_detected,
            "decision_type": self.decision_type,
            "option_count": self.option_count,
            "explicit_option_count": self.explicit_option_count,
            "implied_option_count": self.implied_option_count,
            "criteria_count": self.criteria_count,
            "explicit_criterion_count": self.explicit_criterion_count,
            "inferred_criterion_count": self.inferred_criterion_count,
            "constraint_count": self.constraint_count,
            "missing_context_count": self.missing_context_count,
            "explicit_assumption_count": self.explicit_assumption_count,
            "framing_llm_calls": self.framing_llm_calls,
            "framing_time_ms": round(self.framing_time_ms, 2),
            "framing_failed": self.framing_failed,
            "failure_reason": self.failure_reason,
        }
