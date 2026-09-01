"""Structured output schemas for atomic claim extraction."""

from pydantic import BaseModel, Field


class CandidateClaimItem(BaseModel):
    """A single atomic claim proposed from validated evidence."""

    claim_text: str = Field(
        ...,
        description="One atomic proposition directly supported by the evidence",
        min_length=1,
    )
    claim_type: str = Field(
        default="factual",
        description="One of: factual, statistical, comparative, causal, predictive, definitional, opinion",
    )
    importance: str = Field(
        default="medium",
        description="Materiality to research question: high, medium, low",
    )
    temporal_scope: str | None = Field(
        None,
        description="When the claim applies, e.g. 'as of December 31, 2025' or 'in 2023'",
    )
    geographic_scope: str | None = Field(
        None, description="Geographic applicability if relevant"
    )
    raw_value: str | None = Field(None, description="Preserved quantitative value if applicable")
    unit: str | None = Field(None, description="Unit of measurement")
    currency: str | None = Field(None, description="Currency if applicable")
    qualifiers: list[str] = Field(
        default_factory=list,
        description="Uncertainty, conditions, exceptions (e.g. 'excluding discontinued operations')",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Key entities mentioned in the claim",
    )
    support_basis: str = Field(
        default="direct",
        description="direct, inferred, or analytical — only direct claims are persisted",
    )


class ClaimExtractionOutput(BaseModel):
    """Structured LLM response for claim extraction from one evidence item."""

    claims: list[CandidateClaimItem] = Field(
        default_factory=list,
        description="Atomic claims directly supported by this evidence",
    )


class ClaimSupportValidationOutput(BaseModel):
    """Structured LLM response for claim-support entailment check."""

    is_supported: bool = Field(
        ...,
        description="Whether the evidence supports this exact proposition",
    )
    reason: str = Field(
        default="",
        description="Brief explanation of support or rejection",
    )


class ClaimBatchValidationItem(BaseModel):
    """Validation result for one claim in a batch."""

    claim_index: int = Field(..., ge=0, description="Index into the claims list")
    is_supported: bool = Field(
        ..., description="Whether evidence supports this exact proposition"
    )
    reason: str = Field(default="", description="Brief explanation")


class ClaimBatchValidationOutput(BaseModel):
    """Structured LLM response for batch claim-support validation."""

    results: list[ClaimBatchValidationItem] = Field(
        default_factory=list,
        description="One result per claim, indexed by claim_index",
    )
