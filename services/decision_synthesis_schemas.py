"""Schemas for Phase 3C evidence-grounded decision synthesis."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from services.option_evaluation_schemas import CriterionAssessment, KnowledgeCoverage


class RecommendationStatus(str, Enum):
    RECOMMEND = "recommend"
    TENTATIVE_RECOMMENDATION = "tentative_recommendation"
    INSUFFICIENT_BASIS = "insufficient_basis"


class ConstraintCompliance(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    NOT_ESTABLISHED = "not_established"


class CriterionReference(BaseModel):
    option_label: str
    criterion_label: str
    criterion_origin: Literal["explicit", "inferred"]
    criterion_priority: Literal["primary", "standard"]
    assessment: CriterionAssessment
    knowledge_coverage: KnowledgeCoverage
    claim_ids: list[int] = Field(default_factory=list)


class ConstraintAssessment(BaseModel):
    option_label: str
    constraint: str
    compliance: ConstraintCompliance
    claim_ids: list[int] = Field(default_factory=list)
    reason: str


class ChangeCondition(BaseModel):
    description: str
    change_type: Literal["evidence_change", "decision_context_change"]
    related_option_label: str | None = None
    related_criterion_label: str | None = None
    related_constraint: str | None = None
    related_assumption: str | None = None
    related_missing_context: str | None = None
    related_claim_ids: list[int] = Field(default_factory=list)


class DecisionSynthesis(BaseModel):
    """Structured decision synthesis — domain data only (no metrics)."""

    decision: str
    recommendation_status: RecommendationStatus
    recommended_option: str | None = None
    rationale: str
    supporting_criteria: list[CriterionReference] = Field(default_factory=list)
    limiting_criteria: list[CriterionReference] = Field(default_factory=list)
    constraint_assessments: list[ConstraintAssessment] = Field(default_factory=list)
    key_uncertainties: list[str] = Field(default_factory=list)
    decision_limitations: list[str] = Field(default_factory=list)
    critical_missing_context: list[str] = Field(default_factory=list)
    assumptions_relied_on: list[str] = Field(default_factory=list)
    change_conditions: list[ChangeCondition] = Field(default_factory=list)


class CriterionReferenceLLM(BaseModel):
    option_label: str
    criterion_label: str


class ConstraintAssessmentLLM(BaseModel):
    option_label: str
    constraint: str
    compliance: ConstraintCompliance
    claim_ids: list[int] = Field(default_factory=list)
    reason: str


class ChangeConditionLLM(BaseModel):
    description: str
    change_type: Literal["evidence_change", "decision_context_change"]
    related_option_label: str | None = None
    related_criterion_label: str | None = None
    related_constraint: str | None = None
    related_assumption: str | None = None
    related_missing_context: str | None = None
    related_claim_ids: list[int] = Field(default_factory=list)


class DecisionSynthesisLLMOutput(BaseModel):
    recommendation_status: RecommendationStatus
    recommended_option: str | None = None
    rationale: str
    supporting_criteria: list[CriterionReferenceLLM] = Field(default_factory=list)
    limiting_criteria: list[CriterionReferenceLLM] = Field(default_factory=list)
    constraint_assessments: list[ConstraintAssessmentLLM] = Field(default_factory=list)
    key_uncertainties: list[str] = Field(default_factory=list)
    critical_missing_context: list[str] = Field(default_factory=list)
    assumptions_relied_on: list[str] = Field(default_factory=list)
    change_conditions: list[ChangeConditionLLM] = Field(default_factory=list)


@dataclass
class SynthesisPreCheck:
    matrix_complete: bool = False
    expected_pairs: int = 0
    actual_pairs: int = 0
    option_count: int = 0
    primary_criterion_count: int = 0
    explicit_criterion_count: int = 0
    inferred_criterion_count: int = 0
    constraint_count: int = 0
    missing_context_count: int = 0
    status_ceiling: RecommendationStatus = RecommendationStatus.RECOMMEND
    blockers: list[str] = field(default_factory=list)


@dataclass
class DecisionSynthesisMetrics:
    synthesis_llm_calls: int = 0
    synthesis_time_ms: float = 0.0
    recommendation_status: str | None = None
    recommendation_present: bool = False
    matrix_complete: bool = False
    expected_pairs: int = 0
    actual_pairs: int = 0
    primary_criterion_count: int = 0
    constraint_count: int = 0
    constraint_violation_count: int = 0
    constraint_not_established_count: int = 0
    critical_missing_context_count: int = 0
    assumptions_relied_on_count: int = 0
    supporting_criterion_count: int = 0
    limiting_criterion_count: int = 0
    change_condition_count: int = 0
    synthesis_failed: bool = False
    failure_reason: str | None = None
    synthesis_skipped: bool = False
    synthesis_skipped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "synthesis_llm_calls": self.synthesis_llm_calls,
            "synthesis_time_ms": round(self.synthesis_time_ms, 2),
            "recommendation_status": self.recommendation_status,
            "recommendation_present": self.recommendation_present,
            "matrix_complete": self.matrix_complete,
            "expected_pairs": self.expected_pairs,
            "actual_pairs": self.actual_pairs,
            "primary_criterion_count": self.primary_criterion_count,
            "constraint_count": self.constraint_count,
            "constraint_violation_count": self.constraint_violation_count,
            "constraint_not_established_count": self.constraint_not_established_count,
            "critical_missing_context_count": self.critical_missing_context_count,
            "assumptions_relied_on_count": self.assumptions_relied_on_count,
            "supporting_criterion_count": self.supporting_criterion_count,
            "limiting_criterion_count": self.limiting_criterion_count,
            "change_condition_count": self.change_condition_count,
            "synthesis_failed": self.synthesis_failed,
            "failure_reason": self.failure_reason,
            "synthesis_skipped": self.synthesis_skipped,
            "synthesis_skipped_reason": self.synthesis_skipped_reason,
        }
