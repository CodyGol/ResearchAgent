"""Repositories for research runs and sources."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from config import settings
from db.client import get_client
from domain.models import ResearchRun, ResearchRunStatus, Source
from domain.models import (
    Claim,
    ClaimEvidenceRelation,
    ClaimEvidenceRelationship,
    Evidence,
    EvidenceMatchType,
    EvidenceType,
    ExtractionMethod,
)
from utils.serialization import serialize_for_db

logger = logging.getLogger(__name__)


def is_persistence_enabled() -> bool:
    """Check if Supabase persistence is available."""
    return bool(settings.supabase_url and settings.supabase_key)


class ResearchRunRepository:
    """Repository for research run lifecycle."""

    def __init__(self) -> None:
        self.client = get_client()
        self.table = "research_runs"

    async def create_run(
        self,
        query: str,
        model_name: str | None = None,
    ) -> ResearchRun:
        """Create a new research run in pending state."""
        now = datetime.now(timezone.utc)
        record = {
            "query": query,
            "status": ResearchRunStatus.RUNNING.value,
            "model_name": model_name or settings.model_name,
            "started_at": now.isoformat(),
            "metadata": {},
        }
        serialized = serialize_for_db(record)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.table(self.table).insert(serialized).execute(),
        )

        row = _extract_first_row(response)
        return _row_to_research_run(row)

    async def complete_run(
        self,
        run_id: int,
        *,
        status: ResearchRunStatus = ResearchRunStatus.COMPLETED,
        iteration_count: int = 0,
        sources_count: int = 0,
        evidence_count: int = 0,
        claims_count: int = 0,
        failed_validations: int = 0,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Mark a research run as completed or failed."""
        update: dict[str, Any] = {
            "status": status.value,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "iteration_count": iteration_count,
            "sources_count": sources_count,
            "evidence_count": evidence_count,
            "claims_count": claims_count,
            "failed_validations": failed_validations,
        }
        if metadata:
            update["metadata"] = metadata
        if error:
            update["error"] = error

        serialized = serialize_for_db(update)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: (
                self.client.table(self.table)
                .update(serialized)
                .eq("id", run_id)
                .execute()
            ),
        )

    async def get_run(self, run_id: int) -> ResearchRun | None:
        """Retrieve a research run by ID."""
        try:
            response = (
                self.client.table(self.table)
                .select("*")
                .eq("id", run_id)
                .limit(1)
                .execute()
            )
            if response.data:
                return _row_to_research_run(response.data[0])
        except Exception as e:
            logger.warning("Failed to retrieve research run %s: %s", run_id, e)
        return None


class SourceRepository:
    """Repository for normalized source persistence."""

    def __init__(self) -> None:
        self.client = get_client()
        self.table = "sources"

    async def save_sources(self, sources: list[Source]) -> list[Source]:
        """
        Persist sources, skipping duplicates via (research_run_id, url, content_hash).

        Returns sources with database IDs populated.
        """
        if not sources:
            return []

        records = []
        for source in sources:
            data = source.model_dump(
                exclude={"id", "created_at"},
                mode="json",
            )
            # Convert enums to values
            data["source_type"] = source.source_type.value
            data["source_quality"] = source.source_quality.value
            if source.accessed_at:
                data["accessed_at"] = source.accessed_at.isoformat()
            if source.published_at:
                data["published_at"] = source.published_at.isoformat()
            records.append(serialize_for_db(data))

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: (
                self.client.table(self.table)
                .upsert(records, on_conflict="research_run_id,url,content_hash")
                .execute()
            ),
        )

        saved: list[Source] = []
        rows = response.data if hasattr(response, "data") and response.data else []
        for row in rows:
            saved.append(_row_to_source(row))

        return saved

    async def get_sources_for_run(self, research_run_id: int) -> list[Source]:
        """Retrieve all sources for a research run."""
        try:
            response = (
                self.client.table(self.table)
                .select("*")
                .eq("research_run_id", research_run_id)
                .execute()
            )
            return [_row_to_source(row) for row in (response.data or [])]
        except Exception as e:
            logger.warning(
                "Failed to retrieve sources for run %s: %s", research_run_id, e
            )
            return []


class EvidenceRepository:
    """Repository for validated evidence persistence."""

    def __init__(self) -> None:
        self.client = get_client()
        self.table = "evidence"

    async def save_evidence(self, evidence_list: list[Evidence]) -> list[Evidence]:
        """
        Persist validated evidence records.

        Returns evidence with database IDs populated.
        """
        if not evidence_list:
            return []

        records = []
        for ev in evidence_list:
            data = ev.model_dump(exclude={"id", "created_at"}, mode="json")
            data["evidence_type"] = ev.evidence_type.value
            data["extraction_method"] = ev.extraction_method.value
            data["match_type"] = ev.match_type.value if ev.match_type else None
            records.append(serialize_for_db(data))

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.table(self.table).insert(records).execute(),
        )

        saved: list[Evidence] = []
        rows = response.data if hasattr(response, "data") and response.data else []
        for row in rows:
            saved.append(_row_to_evidence(row))
        return saved

    async def get_evidence_for_run(self, research_run_id: int) -> list[Evidence]:
        """Retrieve all evidence for a research run."""
        try:
            response = (
                self.client.table(self.table)
                .select("*")
                .eq("research_run_id", research_run_id)
                .eq("is_validated", True)
                .execute()
            )
            return [_row_to_evidence(row) for row in (response.data or [])]
        except Exception as e:
            logger.warning(
                "Failed to retrieve evidence for run %s: %s", research_run_id, e
            )
            return []


def _row_to_evidence(row: dict[str, Any]) -> Evidence:
    return Evidence(
        id=row["id"],
        source_id=row["source_id"],
        research_run_id=row["research_run_id"],
        exact_text=row["exact_text"],
        normalized_text=row.get("normalized_text"),
        locator=row.get("locator"),
        context_before=row.get("context_before"),
        context_after=row.get("context_after"),
        evidence_type=EvidenceType(row.get("evidence_type", "other")),
        extraction_method=ExtractionMethod(row.get("extraction_method", "llm")),
        match_type=(
            EvidenceMatchType(row["match_type"]) if row.get("match_type") else None
        ),
        is_validated=row.get("is_validated", False),
        metadata=row.get("metadata") or {},
        created_at=_parse_dt(row.get("created_at")),
    )


def _extract_first_row(response: Any) -> dict[str, Any]:
    data = response.data if hasattr(response, "data") else None
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unexpected database response: {response}")


def _row_to_research_run(row: dict[str, Any]) -> ResearchRun:
    return ResearchRun(
        id=row["id"],
        query=row["query"],
        status=ResearchRunStatus(row["status"]),
        model_name=row.get("model_name"),
        iteration_count=row.get("iteration_count", 0),
        sources_count=row.get("sources_count", 0),
        evidence_count=row.get("evidence_count", 0),
        claims_count=row.get("claims_count", 0),
        failed_validations=row.get("failed_validations", 0),
        metadata=row.get("metadata") or {},
        error=row.get("error"),
        started_at=_parse_dt(row.get("started_at")),
        completed_at=_parse_dt(row.get("completed_at")),
        created_at=_parse_dt(row.get("created_at")),
    )


def _row_to_source(row: dict[str, Any]) -> Source:
    from domain.models import SourceQuality, SourceType

    return Source(
        id=row["id"],
        research_run_id=row["research_run_id"],
        url=row["url"],
        title=row.get("title", ""),
        publisher=row.get("publisher"),
        author=row.get("author"),
        published_at=_parse_dt(row.get("published_at")),
        accessed_at=_parse_dt(row.get("accessed_at")),
        source_type=SourceType(row.get("source_type", "unknown")),
        source_quality=SourceQuality(row.get("source_quality", "unknown")),
        content=row.get("content", ""),
        content_hash=row["content_hash"],
        relevance_score=float(row.get("relevance_score", 0)),
        parent_source_id=row.get("parent_source_id"),
        metadata=row.get("metadata") or {},
        created_at=_parse_dt(row.get("created_at")),
    )


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


# Lazy singletons
_run_repo: ResearchRunRepository | None = None
_source_repo: SourceRepository | None = None
_evidence_repo: EvidenceRepository | None = None


def get_run_repo() -> ResearchRunRepository:
    global _run_repo
    if _run_repo is None:
        _run_repo = ResearchRunRepository()
    return _run_repo


def get_source_repo() -> SourceRepository:
    global _source_repo
    if _source_repo is None:
        _source_repo = SourceRepository()
    return _source_repo


def get_evidence_repo() -> EvidenceRepository:
    global _evidence_repo
    if _evidence_repo is None:
        _evidence_repo = EvidenceRepository()
    return _evidence_repo


class ClaimRepository:
    """Repository for claim and claim-evidence persistence."""

    def __init__(self) -> None:
        self.client = get_client()
        self.claims_table = "claims"
        self.relations_table = "claim_evidence"

    async def save_claims(self, claims: list[Claim]) -> list[Claim]:
        """Persist claims and return records with database IDs."""
        if not claims:
            return []

        records = []
        for claim in claims:
            data = claim.model_dump(exclude={"id", "created_at"}, mode="json")
            data["claim_type"] = claim.claim_type.value
            records.append(serialize_for_db(data))

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.table(self.claims_table).insert(records).execute(),
        )

        saved: list[Claim] = []
        rows = response.data if hasattr(response, "data") and response.data else []
        for row in rows:
            saved.append(_row_to_claim(row))
        return saved

    async def save_claim_evidence(
        self, relations: list[ClaimEvidenceRelation]
    ) -> list[ClaimEvidenceRelation]:
        """Persist claim-evidence relationships."""
        if not relations:
            return []

        records = []
        for rel in relations:
            data = rel.model_dump(exclude={"id", "created_at"}, mode="json")
            data["relationship"] = rel.relationship.value
            records.append(serialize_for_db(data))

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: (
                self.client.table(self.relations_table)
                .upsert(records, on_conflict="claim_id,evidence_id,relationship")
                .execute()
            ),
        )

        saved: list[ClaimEvidenceRelation] = []
        rows = response.data if hasattr(response, "data") and response.data else []
        for row in rows:
            saved.append(_row_to_claim_evidence(row))
        return saved

    async def get_relations_for_run(
        self, research_run_id: int
    ) -> list[ClaimEvidenceRelation]:
        """Retrieve claim-evidence relations for a research run via claims join."""
        try:
            claims = await self.get_claims_for_run(research_run_id)
            if not claims:
                return []
            claim_ids = [c.id for c in claims if c.id is not None]
            if not claim_ids:
                return []
            response = (
                self.client.table(self.relations_table)
                .select("*")
                .in_("claim_id", claim_ids)
                .execute()
            )
            return [
                _row_to_claim_evidence(row) for row in (response.data or [])
            ]
        except Exception as e:
            logger.warning(
                "Failed to retrieve relations for run %s: %s", research_run_id, e
            )
            return []

    async def get_claims_for_run(self, research_run_id: int) -> list[Claim]:
        """Retrieve all claims for a research run."""
        try:
            response = (
                self.client.table(self.claims_table)
                .select("*")
                .eq("research_run_id", research_run_id)
                .execute()
            )
            return [_row_to_claim(row) for row in (response.data or [])]
        except Exception as e:
            logger.warning(
                "Failed to retrieve claims for run %s: %s", research_run_id, e
            )
            return []


def _row_to_claim(row: dict[str, Any]) -> Claim:
    from domain.models import ClaimType

    return Claim(
        id=row["id"],
        research_run_id=row["research_run_id"],
        text=row["text"],
        claim_type=ClaimType(row.get("claim_type", "factual")),
        temporal_scope=row.get("temporal_scope"),
        geographic_scope=row.get("geographic_scope"),
        raw_value=row.get("raw_value"),
        unit=row.get("unit"),
        currency=row.get("currency"),
        qualifiers=row.get("qualifiers") or [],
        duplicate_of_id=row.get("duplicate_of_id"),
        metadata=row.get("metadata") or {},
        created_at=_parse_dt(row.get("created_at")),
    )


def _row_to_claim_evidence(row: dict[str, Any]) -> ClaimEvidenceRelation:
    return ClaimEvidenceRelation(
        id=row["id"],
        claim_id=row["claim_id"],
        evidence_id=row["evidence_id"],
        relationship=ClaimEvidenceRelationship(row["relationship"]),
        reasoning=row.get("reasoning"),
        created_at=_parse_dt(row.get("created_at")),
    )


_claim_repo: ClaimRepository | None = None


def get_claim_repo() -> ClaimRepository:
    global _claim_repo
    if _claim_repo is None:
        _claim_repo = ClaimRepository()
    return _claim_repo


class VerificationRepository:
    """Repository for claim verification results."""

    def __init__(self) -> None:
        self.client = get_client()
        self.table = "verifications"

    async def save_verifications(
        self, results: list["VerificationResult"]
    ) -> list["VerificationResult"]:
        from domain.models import VerificationResult

        if not results:
            return []

        records = []
        for vr in results:
            data = vr.model_dump(exclude={"id", "created_at", "verified_at"}, mode="json")
            data["status"] = vr.status.value
            data["confidence"] = vr.confidence.value
            if vr.knowledge_category:
                data["knowledge_category"] = vr.knowledge_category.value
            else:
                data["knowledge_category"] = None
            records.append(serialize_for_db(data))

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: (
                self.client.table(self.table)
                .upsert(records, on_conflict="claim_id,research_run_id")
                .execute()
            ),
        )

        saved: list[VerificationResult] = []
        rows = response.data if hasattr(response, "data") and response.data else []
        for row in rows:
            saved.append(_row_to_verification(row))
        return saved

    async def get_verifications_for_run(
        self, research_run_id: int
    ) -> list["VerificationResult"]:
        from domain.models import VerificationResult

        try:
            response = (
                self.client.table(self.table)
                .select("*")
                .eq("research_run_id", research_run_id)
                .execute()
            )
            return [_row_to_verification(row) for row in (response.data or [])]
        except Exception as e:
            logger.warning(
                "Failed to retrieve verifications for run %s: %s",
                research_run_id,
                e,
            )
            return []


def _row_to_verification(row: dict[str, Any]) -> "VerificationResult":
    from domain.models import EvidenceConfidence, KnowledgeCategory, VerificationResult, VerificationStatus

    kc = row.get("knowledge_category")
    return VerificationResult(
        id=row["id"],
        claim_id=row["claim_id"],
        research_run_id=row["research_run_id"],
        status=VerificationStatus(row["status"]),
        confidence=EvidenceConfidence(row["confidence"]),
        reasoning=row.get("reasoning"),
        knowledge_category=KnowledgeCategory(kc) if kc else None,
        verified_at=_parse_dt(row.get("verified_at")),
        created_at=_parse_dt(row.get("created_at")),
    )
