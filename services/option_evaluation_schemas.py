"""Schemas for Phase 3B evidence-grounded option evaluation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class CriterionAssessment(str, Enum):
    FAVORABLE = "favorable"
    UNFAVORABLE = "unfavorable"
    MIXED = "mixed"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_INFORMATION = "insufficient_information"


class KnowledgeCoverage(str, Enum):
    GROUNDED = "grounded"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class CriterionEvaluation(BaseModel):
    """Evaluation of one option on one criterion."""

    criterion_label: str
    criterion_origin: Literal["explicit", "inferred"]
    criterion_priority: Literal["primary", "standard"] = "standard"
    assessment: CriterionAssessment
    knowledge_coverage: KnowledgeCoverage
    claim_ids: list[int] = Field(default_factory=list)
    verification_ids: list[int] = Field(default_factory=list)
    knowledge_categories: list[str] = Field(default_factory=list)
    reason: str


class OptionEvaluationEntry(BaseModel):
    option_label: str
    option_origin: Literal["explicit", "implied"]
    criteria_evaluations: list[CriterionEvaluation] = Field(default_factory=list)


class OptionEvaluation(BaseModel):
    """Structured option evaluation — domain data only (no metrics)."""

    decision: str
    option_evaluations: list[OptionEvaluationEntry] = Field(default_factory=list)
    decision_limitations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class CriterionEvaluationLLM(BaseModel):
    """LLM row: maps claims to option × criterion implication."""

    option_label: str
    criterion_label: str
    assessment: CriterionAssessment
    claim_ids: list[int] = Field(default_factory=list)
    reason: str


class OptionEvaluationLLMOutput(BaseModel):
    evaluations: list[CriterionEvaluationLLM] = Field(default_factory=list)


@dataclass
class ClaimCatalogEntry:
    claim_id: int
    bucket: str
    verification_id: int | None
    verification_status: str
    knowledge_category: str | None
    claim_text: str


@dataclass
class OptionEvaluationMetrics:
    option_count: int = 0
    criterion_count: int = 0
    catalog_claim_count: int = 0
    evaluations_generated: int = 0
    grounded_evaluation_count: int = 0
    partial_evaluation_count: int = 0
    insufficient_evaluation_count: int = 0
    referenced_claim_count: int = 0
    invalid_reference_count: int = 0
    rejected_row_count: int = 0
    evaluation_llm_calls: int = 0
    evaluation_time_ms: float = 0.0
    evaluation_failed: bool = False
    evaluation_skipped: bool = False
    evaluation_skipped_reason: str | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_count": self.option_count,
            "criterion_count": self.criterion_count,
            "catalog_claim_count": self.catalog_claim_count,
            "evaluations_generated": self.evaluations_generated,
            "grounded_evaluation_count": self.grounded_evaluation_count,
            "partial_evaluation_count": self.partial_evaluation_count,
            "insufficient_evaluation_count": self.insufficient_evaluation_count,
            "referenced_claim_count": self.referenced_claim_count,
            "invalid_reference_count": self.invalid_reference_count,
            "rejected_row_count": self.rejected_row_count,
            "evaluation_llm_calls": self.evaluation_llm_calls,
            "evaluation_time_ms": round(self.evaluation_time_ms, 2),
            "evaluation_failed": self.evaluation_failed,
            "evaluation_skipped": self.evaluation_skipped,
            "evaluation_skipped_reason": self.evaluation_skipped_reason,
            "failure_reason": self.failure_reason,
        }
