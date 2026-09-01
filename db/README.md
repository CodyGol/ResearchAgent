# Supabase Integration

The Oracle uses Supabase for persistence and (optionally) research-plan caching.

## What Is Persisted

| Layer | Tables | When |
|-------|--------|------|
| Plan cache | `research_plans` | Planner (if `ENABLE_CACHING=true`) |
| Research run | `research_runs` | Start/finalize via `services/research_run_service.py` |
| Sources | `sources` | Researcher / fast path |
| Evidence | `evidence` | Evidence extractor |
| Claims | `claims` | Claim extractor |
| Claim–evidence links | `claim_evidence` | Claim extractor + claim verifier |
| Verifications | `verifications` | Claim verifier; `knowledge_category` backfilled by Knowledge State node |
| Reports | `research_reports` | Writer (optional `research_run_id` link) |

Legacy analytics table: `search_results` (from `schema.sql`).

**There is no full caching system** for evidence or claims. `ENABLE_CACHING` applies to research plans only.

## Setup

### 1. Create Supabase Project

1. Go to https://supabase.com/
2. Create a new project
3. Note your project URL and service role key

### 2. Configure Environment

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
ENABLE_CACHING=true
CACHE_TTL_HOURS=24
```

Use the **Service Role Key** for backend services (bypasses RLS).

### 3. Create Database Schema

Run both scripts in Supabase SQL Editor:

1. `db/schema.sql` — plan cache, reports, search_results
2. `db/migrations/001_evidence_foundation.sql` — evidence-backed intelligence tables

### 4. Install Dependencies

```bash
uv sync
```

## Repositories

| Module | Purpose |
|--------|---------|
| `db/repository.py` | Plan cache, report storage (legacy) |
| `db/evidence_repositories.py` | Research runs, sources, evidence, claims, verifications |
| `db/client.py` | Supabase/postgrest client |

Persistence is skipped when `SUPABASE_URL` / `SUPABASE_KEY` are unset; the agent uses in-memory negative IDs.

## Knowledge State Persistence

Phase 2D writes:

- `verifications.knowledge_category` — `known`, `likely`, `disputed`, or `unknown` (null for contradicted/unverifiable)
- `research_runs.metadata` — compact snapshots at finalize (IDs and metrics only)

Decision artifacts (`decision_frame`, `option_evaluation`, `decision_synthesis`) may appear in `research_runs.metadata` when present in final state. There is no dedicated decision-persistence schema yet.

## Manual Usage (Legacy Reports)

```python
from db.repository import plan_repo, report_repo

cached_plan = await plan_repo.get_cached_plan("your query")
report_id = await report_repo.save_report(
    query="Your query",
    report=final_report,
    quality_score=0.85,
    iteration_count=2,
)
```

For evidence-backed runs, prefer repositories in `db/evidence_repositories.py`.

## Configuration

- `ENABLE_CACHING`: Plan cache only (default: `true` when Supabase configured)
- `CACHE_TTL_HOURS`: Plan cache TTL (default: `24`)

## Security

Row Level Security is enabled on evidence tables. Service role policies allow backend access. Adjust in migration SQL for production auth models.

## Maintenance

### Cleanup Expired Plan Cache

```sql
SELECT cleanup_expired_plans();
```

### Query Recent Runs

```sql
SELECT id, query, status, metadata->'knowledge_state_metrics' AS ks_metrics, created_at
FROM research_runs
ORDER BY created_at DESC
LIMIT 10;
```

### Query Verifications by Category

```sql
SELECT c.text, v.status, v.knowledge_category, v.confidence
FROM verifications v
JOIN claims c ON c.id = v.claim_id
WHERE v.research_run_id = :run_id;
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Table does not exist | Run both `schema.sql` and `001_evidence_foundation.sql` |
| Permission denied | Use service role key; check RLS policies |
| Cache not working | `ENABLE_CACHING=true`; verify `research_plans` exists |
| Verifications not saved | Check `SUPABASE_*` env vars; see logs for persistence warnings |

See [docs/architecture.md](../docs/architecture.md) for the full data model.
