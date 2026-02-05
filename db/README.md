# Supabase Integration

The Oracle uses Supabase for:
1. **Research Plan Caching** - Avoid redundant LLM calls for similar queries
2. **Report Persistence** - Store all research reports for later retrieval
3. **Analytics** - Track search results and research patterns

## Setup

### 1. Create Supabase Project

1. Go to https://supabase.com/
2. Create a new project
3. Note your project URL and API key

### 2. Configure Environment

Add to your `.env` file:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-or-service-role-key
ENABLE_CACHING=true
CACHE_TTL_HOURS=24
```

### 3. Create Database Schema

1. Open Supabase Dashboard → SQL Editor
2. Run the SQL from `schema.sql`:
   ```bash
   cat db/schema.sql
   ```
3. Copy and paste into SQL Editor, then execute

### 4. Install Dependencies

```bash
pip install supabase postgrest
# or
uv sync
```

## Usage

### Automatic Integration

The system automatically:
- **Caches research plans** in the planner node (if `ENABLE_CACHING=true`)
- **Saves all reports** to the database in the writer node

### Manual Usage

```python
from db.repository import plan_repo, report_repo

# Check cache
cached_plan = await plan_repo.get_cached_plan("your query")
if cached_plan:
    print("Found cached plan!")

# Save report
report_id = await report_repo.save_report(
    query="Your query",
    report=final_report,
    quality_score=0.85,
    iteration_count=2
)

# Retrieve report
report = await report_repo.get_report(report_id)

# List recent reports
reports = await report_repo.list_reports(limit=10)
```

## Configuration

### Cache Settings

- `ENABLE_CACHING`: Enable/disable plan caching (default: `true`)
- `CACHE_TTL_HOURS`: How long to cache plans (default: `24`)

### API Key Type

- **Anon Key**: Public, limited by RLS policies
- **Service Role Key**: Full access, bypasses RLS (use for backend)

For The Oracle, use **Service Role Key** since it's a backend service.

## Security

The schema includes Row Level Security (RLS). Adjust policies in `schema.sql` based on your needs:

- **Public access**: Current policies allow all (for service role)
- **Authenticated users**: Modify policies to check `auth.uid()`
- **Read-only**: Create separate read policies

## Maintenance

### Cleanup Expired Cache

The schema includes a cleanup function. Run manually:

```sql
SELECT cleanup_expired_plans();
```

Or schedule with pg_cron (if enabled):

```sql
SELECT cron.schedule('cleanup-expired-plans', '0 2 * * *', 'SELECT cleanup_expired_plans()');
```

### Query Reports

```sql
-- Recent reports
SELECT id, query, confidence, created_at 
FROM research_reports 
ORDER BY created_at DESC 
LIMIT 10;

-- Reports by quality
SELECT * FROM research_reports 
WHERE quality_score >= 0.8 
ORDER BY created_at DESC;

-- Cache hit rate (if tracking)
SELECT COUNT(*) as total, 
       COUNT(CASE WHEN expires_at > NOW() THEN 1 END) as active
FROM research_plans;
```

## Troubleshooting

### "Table does not exist"
- Run the schema SQL in Supabase SQL Editor

### "Permission denied"
- Check RLS policies
- Use service role key instead of anon key

### "Cache not working"
- Check `ENABLE_CACHING=true` in `.env`
- Verify table exists and has correct structure
- Check Supabase logs for errors
