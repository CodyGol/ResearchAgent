"""Domain models for the evidence-backed intelligence system."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ResearchRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceType(str, Enum):
    WEB = "web"
    ACADEMIC = "academic"
    OFFICIAL = "official"
    NEWS = "news"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class SourceQuality(str, Enum):
    PRIMARY = "primary"
    OFFICIAL = "official"
    ACADEMIC = "academic"
    REPUTABLE_SECONDARY = "reputable_secondary"
    GENERAL_SECONDARY = "general_secondary"
    USER_GENERATED = "user_generated"
    UNKNOWN = "unknown"


class EvidenceType(str, Enum):
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    STATISTIC = "statistic"
    DEFINITION = "definition"
    OPINION = "opinion"
    OTHER = "other"


class ExtractionMethod(str, Enum):
    LLM = "llm"
    MANUAL = "manual"
    RULE = "rule"


class EvidenceMatchType(str, Enum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    FUZZY = "fuzzy"
    NOT_FOUND = "not_found"


class ClaimType(str, Enum):
    FACTUAL = "factual"
    STATISTICAL = "statistical"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    PREDICTIVE = "predictive"
    ANALYTICAL = "analytical"
    OPINION = "opinion"
    DEFINITIONAL = "definitional"


class ClaimEvidenceRelationship(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    QUALIFIES = "qualifies"
    CONTEXTUALIZES = "contextualizes"


class VerificationStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNVERIFIABLE = "unverifiable"


class EvidenceConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class KnowledgeCategory(str, Enum):
    KNOWN = "known"
    LIKELY = "likely"
    DISPUTED = "disputed"
    UNKNOWN = "unknown"
    ASSUMPTION = "assumption"


class ConflictType(str, Enum):
    CONTRADICTION = "contradiction"
    QUALIFICATION = "qualification"
    DIFFERENT_SCOPE = "different_scope"


class ClaimImportance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClaimSupportBasis(str, Enum):
    DIRECT = "direct"
    INFERRED = "inferred"
    ANALYTICAL = "analytical"


class ClaimSupportStatus(str, Enum):
    SUPPORTED_BY_ORIGIN_EVIDENCE = "supported_by_origin_evidence"
    NOT_SUPPORTED_BY_ORIGIN_EVIDENCE = "not_supported_by_origin_evidence"


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------


class ResearchRun(BaseModel):
    """A single research invocation with full audit trail."""

    id: int | None = Field(None, description="Database primary key")
    query: str = Field(..., description="Original research question")
    status: ResearchRunStatus = Field(
        default=ResearchRunStatus.PENDING,
        description="Current run status",
    )
    model_name: str | None = Field(None, description="LLM model used")
    iteration_count: int = Field(default=0, ge=0)
    sources_count: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    claims_count: int = Field(default=0, ge=0)
    failed_validations: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(None, description="Error message if failed")
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None


class Source(BaseModel):
    """A normalized research source discovered during a run."""

    id: int | None = Field(None, description="Database primary key")
    research_run_id: int = Field(..., description="Parent research run")
    url: str = Field(..., description="Source URL")
    title: str = Field(default="", description="Source title")
    publisher: str | None = Field(None, description="Publisher name")
    author: str | None = Field(None, description="Author name")
    published_at: datetime | None = Field(None, description="Publication date")
    accessed_at: datetime | None = Field(None, description="When source was accessed")
    source_type: SourceType = Field(default=SourceType.UNKNOWN)
    source_quality: SourceQuality = Field(default=SourceQuality.UNKNOWN)
    content: str = Field(default="", description="Full source text used for validation")
    content_hash: str = Field(..., description="SHA-256 hash of content for dedup")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    parent_source_id: int | None = Field(
        None, description="Lineage: upstream source if derived"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class Evidence(BaseModel):
    """A specific source passage relevant to a claim."""

    id: int | None = Field(None, description="Database primary key")
    source_id: int = Field(..., description="Parent source")
    research_run_id: int = Field(..., description="Parent research run")
    exact_text: str = Field(..., description="Verbatim text from source")
    normalized_text: str | None = Field(
        None, description="Normalized form used for matching"
    )
    locator: str | None = Field(
        None, description="Position hint (paragraph, section, offset)"
    )
    context_before: str | None = Field(None, description="Text before the evidence span")
    context_after: str | None = Field(None, description="Text after the evidence span")
    evidence_type: EvidenceType = Field(default=EvidenceType.DIRECT_QUOTE)
    extraction_method: ExtractionMethod = Field(default=ExtractionMethod.LLM)
    match_type: EvidenceMatchType | None = Field(
        None, description="How evidence was validated against source"
    )
    is_validated: bool = Field(default=False, description="Passed integrity check")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class Claim(BaseModel):
    """An atomic proposition extracted from evidence."""

    id: int | None = Field(None, description="Database primary key")
    research_run_id: int = Field(..., description="Parent research run")
    text: str = Field(..., description="The claim proposition")
    claim_type: ClaimType = Field(default=ClaimType.FACTUAL)
    temporal_scope: str | None = Field(
        None, description="When the claim applies (e.g. 'as of 2025')"
    )
    geographic_scope: str | None = Field(None, description="Geographic applicability")
    raw_value: str | None = Field(None, description="Preserved quantitative value")
    unit: str | None = Field(None, description="Unit of measurement")
    currency: str | None = Field(None, description="Currency if applicable")
    qualifiers: list[str] = Field(
        default_factory=list, description="Uncertainty or scope qualifiers"
    )
    duplicate_of_id: int | None = Field(
        None, description="If merged as duplicate, points to canonical claim"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class ClaimEvidenceRelation(BaseModel):
    """Explicit link between a claim and a piece of evidence."""

    id: int | None = Field(None, description="Database primary key")
    claim_id: int = Field(..., description="Linked claim")
    evidence_id: int = Field(..., description="Linked evidence")
    relationship: ClaimEvidenceRelationship = Field(
        ..., description="How evidence relates to claim"
    )
    reasoning: str | None = Field(
        None, description="Why this relationship was assigned"
    )
    created_at: datetime | None = None


class VerificationResult(BaseModel):
    """Verification state for a claim given its evidence."""

    id: int | None = Field(None, description="Database primary key")
    claim_id: int = Field(..., description="Verified claim")
    research_run_id: int = Field(..., description="Parent research run")
    status: VerificationStatus = Field(..., description="Verification outcome")
    confidence: EvidenceConfidence = Field(
        ..., description="Evidence confidence (not probability of truth)"
    )
    reasoning: str | None = Field(None, description="Explanation of verification")
    knowledge_category: KnowledgeCategory | None = Field(
        None, description="Derived knowledge state"
    )
    verified_at: datetime | None = None
    created_at: datetime | None = None
