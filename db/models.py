"""Database models for Supabase (Pydantic V2 schemas)."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ResearchPlanRecord(BaseModel):
    """Database record for research plans (caching)."""

    id: int | None = Field(None, description="Primary key")
    query_hash: str = Field(..., description="MD5 hash of the query")
    query: str = Field(..., description="Original query")
    plan_data: dict[str, Any] = Field(..., description="Serialized ResearchPlan")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(..., description="Cache expiration time")


class ResearchReportRecord(BaseModel):
    """Database record for research reports."""

    id: int | None = Field(None, description="Primary key")
    query: str = Field(..., description="Original research query")
    report_content: str = Field(..., description="Report markdown content")
    sources: list[str] = Field(default_factory=list, description="Source URLs")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    quality_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Final quality score from critic"
    )
    iteration_count: int = Field(
        default=0, ge=0, description="Number of research-critic cycles"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SearchResultRecord(BaseModel):
    """Database record for individual search results (for analytics)."""

    id: int | None = Field(None, description="Primary key")
    report_id: int | None = Field(None, description="Foreign key to research_report")
    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Source URL")
    content: str = Field(..., description="Result content snippet")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    created_at: datetime = Field(default_factory=datetime.utcnow)
