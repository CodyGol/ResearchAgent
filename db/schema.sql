-- Supabase Database Schema for The Oracle
-- Run this SQL in your Supabase SQL Editor to create the required tables

-- Research Plans Cache Table
CREATE TABLE IF NOT EXISTS research_plans (
    id BIGSERIAL PRIMARY KEY,
    query_hash TEXT UNIQUE NOT NULL,
    query TEXT NOT NULL,
    plan_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

-- Index for fast cache lookups
CREATE INDEX IF NOT EXISTS idx_research_plans_query_hash ON research_plans(query_hash);
CREATE INDEX IF NOT EXISTS idx_research_plans_expires_at ON research_plans(expires_at);

-- Research Reports Table
CREATE TABLE IF NOT EXISTS research_reports (
    id BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    report_content TEXT NOT NULL,
    sources TEXT[] DEFAULT '{}',
    confidence DECIMAL(3, 2) CHECK (confidence >= 0 AND confidence <= 1),
    quality_score DECIMAL(3, 2) CHECK (quality_score >= 0 AND quality_score <= 1),
    iteration_count INTEGER DEFAULT 0 CHECK (iteration_count >= 0),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Index for querying reports
CREATE INDEX IF NOT EXISTS idx_research_reports_created_at ON research_reports(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_reports_query ON research_reports(query);

-- Search Results Table (for analytics)
CREATE TABLE IF NOT EXISTS search_results (
    id BIGSERIAL PRIMARY KEY,
    report_id BIGINT REFERENCES research_reports(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    content TEXT,
    score DECIMAL(3, 2) CHECK (score >= 0 AND score <= 1),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for report association
CREATE INDEX IF NOT EXISTS idx_search_results_report_id ON search_results(report_id);

-- Enable Row Level Security (RLS) - adjust policies based on your needs
ALTER TABLE research_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE search_results ENABLE ROW LEVEL SECURITY;

-- Example RLS policies (adjust based on your authentication setup)
-- For now, allow all operations with service role key
CREATE POLICY "Allow all operations for service role" ON research_plans
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow all operations for service role" ON research_reports
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow all operations for service role" ON search_results
    FOR ALL USING (true) WITH CHECK (true);

-- Optional: Function to clean up expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_plans()
RETURNS void AS $$
BEGIN
    DELETE FROM research_plans WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- Optional: Schedule cleanup (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-expired-plans', '0 2 * * *', 'SELECT cleanup_expired_plans()');