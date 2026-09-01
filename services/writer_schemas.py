"""Structured output schemas for evidence-grounded report generation."""

from pydantic import BaseModel, Field


class EvidenceGroundedWriterOutput(BaseModel):
    """LLM structured output for evidence-grounded report synthesis."""

    content: str = Field(
        ...,
        description=(
            "Markdown report. Every material factual statement must include "
            "[E#] evidence references. Keep length proportional to question complexity."
        ),
    )
    evidence_ids_used: list[str] = Field(
        default_factory=list,
        description="List of evidence IDs cited in the report, e.g. ['E1', 'E2']",
    )
    factual_summary: str = Field(
        ...,
        description="One-paragraph summary of evidence-backed facts only",
    )
