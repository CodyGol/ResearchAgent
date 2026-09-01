"""Schemas for derived knowledge state (Phase 2D)."""

from typing import Literal

from pydantic import BaseModel, Field

from domain.models import (
    EvidenceConfidence,
    KnowledgeCategory,
    VerificationStatus,
)


class KnowledgeStateEntry(BaseModel):
    """Reference to an existing claim's derived knowledge position."""

    claim_id: int
    verification_id: int | None = None
    knowledge_category: KnowledgeCategory | None = None
    verification_status: VerificationStatus
    confidence: EvidenceConfidence
    relation_ids: list[int] = Field(default_factory=list)
    evidence_ids: list[int] = Field(default_factory=list)


class InformationGap(BaseModel):
    """Non-claim information gap hint from downstream signals."""

    description: str
    source: Literal["critic_unsupported_area"]


class KnowledgeState(BaseModel):
    """Run-level derived knowledge state for a full-pipeline research run."""

    known: list[KnowledgeStateEntry] = Field(default_factory=list)
    likely: list[KnowledgeStateEntry] = Field(default_factory=list)
    disputed: list[KnowledgeStateEntry] = Field(default_factory=list)
    unknown: list[KnowledgeStateEntry] = Field(default_factory=list)
    contradicted: list[KnowledgeStateEntry] = Field(default_factory=list)
    unverifiable: list[KnowledgeStateEntry] = Field(default_factory=list)
    information_gaps: list[InformationGap] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
