"""Research run lifecycle management."""

import logging
from dataclasses import dataclass, field

from config import settings
from domain.models import ResearchRun, ResearchRunStatus, Source

logger = logging.getLogger(__name__)

_transient_run_counter = 0


@dataclass
class RunContext:
    """Holds research run state for the duration of a pipeline execution."""

    run: ResearchRun
    sources: list[Source] = field(default_factory=list)
    is_persisted: bool = False


async def start_research_run(query: str) -> RunContext:
    """
    Create a research run at pipeline start.

    Persists to Supabase when configured; otherwise uses a transient in-memory ID.
    """
    global _transient_run_counter

    from db.evidence_repositories import get_run_repo, is_persistence_enabled

    if is_persistence_enabled():
        try:
            repo = get_run_repo()
            run = await repo.create_run(query, model_name=settings.model_name)
            logger.info("Created research run %s for query: %s", run.id, query[:80])
            return RunContext(run=run, is_persisted=True)
        except Exception as e:
            logger.warning(
                "Failed to create persisted research run, using transient: %s", e
            )

    _transient_run_counter += 1
    run = ResearchRun(
        id=-_transient_run_counter,
        query=query,
        status=ResearchRunStatus.RUNNING,
        model_name=settings.model_name,
    )
    return RunContext(run=run, is_persisted=False)


async def persist_sources(ctx: RunContext, sources: list[Source]) -> list[Source]:
    """
    Persist normalized sources and update run context.

    When persistence is disabled, sources are kept in-memory only.
    """
    from db.evidence_repositories import get_run_repo, get_source_repo, is_persistence_enabled

    ctx.sources = sources

    if not is_persistence_enabled() or not ctx.is_persisted or ctx.run.id is None:
        logger.info("Sources kept in-memory (%d sources)", len(sources))
        return sources

    try:
        repo = get_source_repo()
        saved = await repo.save_sources(sources)
        ctx.sources = saved

        run_repo = get_run_repo()
        await run_repo.complete_run(
            ctx.run.id,
            status=ResearchRunStatus.RUNNING,
            sources_count=len(saved),
        )
        ctx.run.sources_count = len(saved)
        logger.info("Persisted %d sources for run %s", len(saved), ctx.run.id)
        return saved
    except Exception as e:
        logger.warning("Failed to persist sources: %s", e)
        return sources


async def persist_sources_for_run(
    run_id: int,
    query: str,
    sources: list[Source],
    *,
    is_persisted: bool,
) -> list[Source]:
    """Persist sources for a given run ID without a full RunContext."""
    ctx = RunContext(
        run=ResearchRun(id=run_id, query=query, status=ResearchRunStatus.RUNNING),
        is_persisted=is_persisted,
    )
    return await persist_sources(ctx, sources)


async def finalize_research_run(
    ctx: RunContext,
    *,
    status: ResearchRunStatus = ResearchRunStatus.COMPLETED,
    iteration_count: int = 0,
    evidence_count: int = 0,
    claims_count: int = 0,
    failed_validations: int = 0,
    metadata: dict | None = None,
    error: str | None = None,
) -> None:
    """Finalize a research run on pipeline completion."""
    from db.evidence_repositories import get_run_repo, is_persistence_enabled

    ctx.run.status = status
    ctx.run.iteration_count = iteration_count
    ctx.run.evidence_count = evidence_count
    ctx.run.claims_count = claims_count
    ctx.run.failed_validations = failed_validations
    if error:
        ctx.run.error = error

    if not is_persistence_enabled() or not ctx.is_persisted or ctx.run.id is None:
        return

    try:
        repo = get_run_repo()
        await repo.complete_run(
            ctx.run.id,
            status=status,
            iteration_count=iteration_count,
            sources_count=len(ctx.sources),
            evidence_count=evidence_count,
            claims_count=claims_count,
            failed_validations=failed_validations,
            metadata=metadata,
            error=error,
        )
    except Exception as e:
        logger.warning("Failed to finalize research run: %s", e)
