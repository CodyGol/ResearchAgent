"""Structured schemas for cross-source claim verification."""

from pydantic import BaseModel, Field


class ClaimEvidenceAssessment(BaseModel):
    """Relationship between one material claim and one evidence item."""

    evidence_id: int = Field(..., description="Evidence database ID")
    relationship: str = Field(
        ...,
        description="One of: supports, contradicts, qualifies",
    )
    reasoning: str = Field(default="", description="Why this relationship was assigned")
    classification_mode: str = Field(
        default="deterministic",
        description="deterministic or llm",
    )


class ClaimRelationshipBatchItem(BaseModel):
    """LLM classification for one (claim, evidence) pair."""

    evidence_id: int = Field(..., description="Evidence ID from the prompt")
    relationship: str = Field(
        ...,
        description="supports, contradicts, qualifies, or none",
    )
    reasoning: str = Field(default="")


class ClaimRelationshipBatchOutput(BaseModel):
    """Batch LLM output for ambiguous claim-evidence pairs."""

    assessments: list[ClaimRelationshipBatchItem] = Field(default_factory=list)
