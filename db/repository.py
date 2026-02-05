"""Repository pattern for database operations (Rule 3.B: Idempotency)."""

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Any

from config import settings
from db.client import get_client
from db.models import ResearchPlanRecord, ResearchReportRecord, SearchResultRecord
from state import FinalReport, ResearchPlan
from utils.serialization import serialize_for_db


class ResearchPlanRepository:
    """Repository for research plan caching."""

    def __init__(self):
        self.client = get_client()
        self.table = "research_plans"

    def _hash_query(self, query: str) -> str:
        """Generate MD5 hash of query for caching."""
        return hashlib.md5(query.encode("utf-8")).hexdigest()

    async def get_cached_plan(self, query: str) -> ResearchPlan | None:
        """
        Retrieve cached research plan if available and not expired.

        Args:
            query: Original research query

        Returns:
            ResearchPlan if found and valid, None otherwise
        """
        if not settings.enable_caching:
            return None

        query_hash = self._hash_query(query)

        try:
            response = (
                self.client.table(self.table)
                .select("*")
                .eq("query_hash", query_hash)
                .gt("expires_at", datetime.utcnow().isoformat())
                .limit(1)
                .execute()
            )

            if response.data:
                record = ResearchPlanRecord(**response.data[0])
                return ResearchPlan(**record.plan_data)
        except Exception as e:
            # Fail silently for cache misses, log for other errors
            print(f"Cache lookup failed: {e}")
            return None

        return None

    async def save_plan(self, query: str, plan: ResearchPlan) -> None:
        """
        Save research plan to cache.

        Args:
            query: Original research query
            plan: ResearchPlan to cache
        """
        if not settings.enable_caching:
            return

        query_hash = self._hash_query(query)
        expires_at = datetime.utcnow() + timedelta(
            hours=settings.cache_ttl_hours
        )

        record = ResearchPlanRecord(
            query_hash=query_hash,
            query=query,
            plan_data=plan.model_dump(),
            expires_at=expires_at,
        )

        try:
            # Serialize with datetime handling for database storage
            record_dict = record.model_dump(exclude={"id", "created_at"})
            serialized = serialize_for_db(record_dict)
            
            # Upsert: update if exists, insert if not
            self.client.table(self.table).upsert(
                serialized,
                on_conflict="query_hash",
            ).execute()
        except Exception as e:
            # Fail silently for cache write failures (non-critical)
            print(f"Cache save failed: {e}")


class ResearchReportRepository:
    """Repository for research report persistence."""

    def __init__(self):
        self.client = get_client()
        self.reports_table = "research_reports"
        self.results_table = "search_results"

    async def save_report(
        self,
        query: str,
        report: FinalReport,
        quality_score: float | None = None,
        iteration_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Save research report to database.

        Args:
            query: Original research query
            report: FinalReport to save
            quality_score: Final quality score from critic
            iteration_count: Number of research-critic cycles
            metadata: Additional metadata

        Returns:
            Database ID of saved report
        """
        record = ResearchReportRecord(
            query=query,
            report_content=report.content,
            sources=report.sources,
            confidence=report.confidence,
            quality_score=quality_score,
            iteration_count=iteration_count,
            metadata=metadata or {},
        )

        try:
            # Prepare data for insert with datetime serialization
            data = record.model_dump(exclude={"id", "created_at"})
            serialized = serialize_for_db(data)
            print(f"📝 Attempting to save report to table '{self.reports_table}'...")
            print(f"   Data keys: {list(serialized.keys())}")
            
            # Execute insert (wrap sync call in executor for async compatibility)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: (
                    self.client.table(self.reports_table)
                    .insert(serialized)
                    .execute()
                ),
            )

            # Debug: Print response structure
            print(f"📦 Response type: {type(response)}")
            print(f"   Response attributes: {dir(response) if hasattr(response, '__dict__') else 'N/A'}")
            
            # Handle different response formats
            result_data = None
            if hasattr(response, "data"):
                result_data = response.data
                print(f"   Response.data type: {type(result_data)}")
            elif isinstance(response, dict):
                result_data = response.get("data")
                print(f"   Response is dict, data key: {result_data is not None}")
            elif hasattr(response, "__dict__"):
                # Check if response has data as attribute
                result_data = getattr(response, "data", None)
            
            if result_data:
                if isinstance(result_data, list) and len(result_data) > 0:
                    report_id = result_data[0].get("id", -1)
                    print(f"✅ Report saved successfully (ID: {report_id})")
                    return report_id
                elif isinstance(result_data, dict):
                    report_id = result_data.get("id", -1)
                    print(f"✅ Report saved successfully (ID: {report_id})")
                    return report_id
            
            # If we get here, response format is unexpected
            print(f"⚠️  Unexpected response format: {response}")
            print(f"   Full response: {response}")
            return -1
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Error saving report: {e}")
            print(f"   Traceback:\n{error_details}")
            raise Exception(f"Failed to save report: {e}\n{error_details}") from e

    async def get_report(self, report_id: int) -> ResearchReportRecord | None:
        """
        Retrieve research report by ID.

        Args:
            report_id: Database ID of report

        Returns:
            ResearchReportRecord if found, None otherwise
        """
        try:
            response = (
                self.client.table(self.reports_table)
                .select("*")
                .eq("id", report_id)
                .limit(1)
                .execute()
            )

            if response.data:
                return ResearchReportRecord(**response.data[0])
        except Exception as e:
            print(f"Failed to retrieve report: {e}")
            return None

        return None

    async def list_reports(
        self, limit: int = 10, offset: int = 0
    ) -> list[ResearchReportRecord]:
        """
        List recent research reports.

        Args:
            limit: Maximum number of reports to return
            offset: Offset for pagination

        Returns:
            List of ResearchReportRecord
        """
        try:
            response = (
                self.client.table(self.reports_table)
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .offset(offset)
                .execute()
            )

            return [ResearchReportRecord(**row) for row in response.data]
        except Exception as e:
            print(f"Failed to list reports: {e}")
            return []

    async def save_search_results(
        self, report_id: int, results: list[SearchResultRecord]
    ) -> None:
        """
        Save search results associated with a report.

        Args:
            report_id: Database ID of parent report
            results: List of SearchResultRecord to save
        """
        if not results:
            return

        try:
            records = [
                {**r.model_dump(exclude={"id", "created_at"}), "report_id": report_id}
                for r in results
            ]

            self.client.table(self.results_table).insert(records).execute()
        except Exception as e:
            # Non-critical, log but don't fail
            print(f"Failed to save search results: {e}")


# Global repository instances (lazy initialization to avoid import-time errors)
_plan_repo: ResearchPlanRepository | None = None
_report_repo: ResearchReportRepository | None = None


def _get_plan_repo() -> ResearchPlanRepository:
    """Get or create global plan repository instance."""
    global _plan_repo
    if _plan_repo is None:
        _plan_repo = ResearchPlanRepository()
    return _plan_repo


def _get_report_repo() -> ResearchReportRepository:
    """Get or create global report repository instance."""
    global _report_repo
    if _report_repo is None:
        _report_repo = ResearchReportRepository()
    return _report_repo


# Lazy module-level accessors using __getattr__ (Python 3.7+)
# This allows: from db.repository import plan_repo (lazy initialization)
def __getattr__(name: str):
    """Allow accessing plan_repo and report_repo as module attributes (lazy initialization)."""
    if name == "plan_repo":
        return _get_plan_repo()
    if name == "report_repo":
        return _get_report_repo()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
