"""LLM structured-output schemas for evidence extraction."""

from pydantic import BaseModel, Field


class CandidateEvidenceItem(BaseModel):
    """A single candidate evidence span proposed by the LLM."""

    text: str = Field(
        ...,
        description="Exact verbatim text from the source snippet. Must be copy-pasted, not paraphrased.",
        min_length=1,
    )
    evidence_type: str = Field(
        ...,
        description="One of: fact, statistic, quote, event, definition, observation, other",
    )
    relevance: str = Field(
        ...,
        description="Brief explanation of why this passage is relevant to the research question",
    )
    locator: str | None = Field(
        None,
        description="Position within the snippet if identifiable (e.g. 'sentence 2', 'opening paragraph'). Do not invent page numbers.",
    )
    context: str | None = Field(
        None,
        description="Brief surrounding context from the snippet that helps interpret the evidence",
    )


class EvidenceExtractionOutput(BaseModel):
    """Structured LLM response for evidence extraction from one source."""

    evidence: list[CandidateEvidenceItem] = Field(
        default_factory=list,
        description="List of useful evidence spans found in the source snippet",
    )
