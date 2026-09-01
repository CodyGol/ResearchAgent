-- Migration 001: Evidence Foundation
-- Run in Supabase SQL Editor after existing schema.sql

-- Research Runs: one row per invocation
CREATE TABLE IF NOT EXISTS research_runs (
    id BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    model_name TEXT,
    iteration_count INTEGER DEFAULT 0 CHECK (iteration_count >= 0),
    sources_count INTEGER DEFAULT 0 CHECK (sources_count >= 0),
    evidence_count INTEGER DEFAULT 0 CHECK (evidence_count >= 0),
    claims_count INTEGER DEFAULT 0 CHECK (claims_count >= 0),
    failed_validations INTEGER DEFAULT 0 CHECK (failed_validations >= 0),
    metadata JSONB DEFAULT '{}',
    error TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_research_runs_status ON research_runs(status);
CREATE INDEX IF NOT EXISTS idx_research_runs_created_at ON research_runs(created_at DESC);

-- Sources: normalized from search results, scoped to a research run
CREATE TABLE IF NOT EXISTS sources (
    id BIGSERIAL PRIMARY KEY,
    research_run_id BIGINT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT DEFAULT '',
    publisher TEXT,
    author TEXT,
    published_at TIMESTAMPTZ,
    accessed_at TIMESTAMPTZ DEFAULT NOW(),
    source_type TEXT DEFAULT 'unknown'
        CHECK (source_type IN ('web', 'academic', 'official', 'news', 'document', 'unknown')),
    source_quality TEXT DEFAULT 'unknown'
        CHECK (source_quality IN (
            'primary', 'official', 'academic', 'reputable_secondary',
            'general_secondary', 'user_generated', 'unknown'
        )),
    content TEXT DEFAULT '',
    content_hash TEXT NOT NULL,
    relevance_score DECIMAL(3, 2) DEFAULT 0 CHECK (relevance_score >= 0 AND relevance_score <= 1),
    parent_source_id BIGINT REFERENCES sources(id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (research_run_id, url, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_sources_research_run_id ON sources(research_run_id);
CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(url);
CREATE INDEX IF NOT EXISTS idx_sources_content_hash ON sources(content_hash);

-- Evidence: validated source passages
CREATE TABLE IF NOT EXISTS evidence (
    id BIGSERIAL PRIMARY KEY,
    source_id BIGINT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    research_run_id BIGINT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    exact_text TEXT NOT NULL,
    normalized_text TEXT,
    locator TEXT,
    context_before TEXT,
    context_after TEXT,
    evidence_type TEXT DEFAULT 'direct_quote'
        CHECK (evidence_type IN (
            'direct_quote', 'paraphrase', 'statistic', 'definition', 'opinion', 'other'
        )),
    extraction_method TEXT DEFAULT 'llm'
        CHECK (extraction_method IN ('llm', 'manual', 'rule')),
    match_type TEXT
        CHECK (match_type IS NULL OR match_type IN ('exact', 'normalized', 'fuzzy', 'not_found')),
    is_validated BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evidence_source_id ON evidence(source_id);
CREATE INDEX IF NOT EXISTS idx_evidence_research_run_id ON evidence(research_run_id);
CREATE INDEX IF NOT EXISTS idx_evidence_is_validated ON evidence(is_validated);

-- Claims: atomic propositions
CREATE TABLE IF NOT EXISTS claims (
    id BIGSERIAL PRIMARY KEY,
    research_run_id BIGINT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    claim_type TEXT DEFAULT 'factual'
        CHECK (claim_type IN (
            'factual', 'statistical', 'comparative', 'causal',
            'predictive', 'analytical', 'opinion', 'definitional'
        )),
    temporal_scope TEXT,
    geographic_scope TEXT,
    raw_value TEXT,
    unit TEXT,
    currency TEXT,
    qualifiers TEXT[] DEFAULT '{}',
    duplicate_of_id BIGINT REFERENCES claims(id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_claims_research_run_id ON claims(research_run_id);
CREATE INDEX IF NOT EXISTS idx_claims_text ON claims USING gin(to_tsvector('english', text));

-- Claim ↔ Evidence relationships
CREATE TABLE IF NOT EXISTS claim_evidence (
    id BIGSERIAL PRIMARY KEY,
    claim_id BIGINT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    evidence_id BIGINT NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
    relationship TEXT NOT NULL
        CHECK (relationship IN ('supports', 'contradicts', 'qualifies', 'contextualizes')),
    reasoning TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (claim_id, evidence_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_claim_evidence_claim_id ON claim_evidence(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_evidence_evidence_id ON claim_evidence(evidence_id);

-- Verification results per claim
CREATE TABLE IF NOT EXISTS verifications (
    id BIGSERIAL PRIMARY KEY,
    claim_id BIGINT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    research_run_id BIGINT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    status TEXT NOT NULL
        CHECK (status IN (
            'supported', 'partially_supported', 'contradicted',
            'uncertain', 'insufficient_evidence', 'unverifiable'
        )),
    confidence TEXT NOT NULL
        CHECK (confidence IN ('high', 'medium', 'low')),
    reasoning TEXT,
    knowledge_category TEXT
        CHECK (knowledge_category IS NULL OR knowledge_category IN (
            'known', 'likely', 'disputed', 'unknown', 'assumption'
        )),
    verified_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (claim_id, research_run_id)
);

CREATE INDEX IF NOT EXISTS idx_verifications_claim_id ON verifications(claim_id);
CREATE INDEX IF NOT EXISTS idx_verifications_research_run_id ON verifications(research_run_id);
CREATE INDEX IF NOT EXISTS idx_verifications_status ON verifications(status);

-- Link existing reports to research runs (optional, backward compatible)
ALTER TABLE research_reports
    ADD COLUMN IF NOT EXISTS research_run_id BIGINT REFERENCES research_runs(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_research_reports_research_run_id ON research_reports(research_run_id);

-- RLS policies (match existing pattern)
ALTER TABLE research_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE claims ENABLE ROW LEVEL SECURITY;
ALTER TABLE claim_evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE verifications ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all operations for service role" ON research_runs
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all operations for service role" ON sources
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all operations for service role" ON evidence
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all operations for service role" ON claims
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all operations for service role" ON claim_evidence
    FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all operations for service role" ON verifications
    FOR ALL USING (true) WITH CHECK (true);
