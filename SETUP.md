# Setup Guide

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install langgraph langchain-anthropic tavily-python pydantic pydantic-settings python-dotenv postgrest httpx
   # or
   uv sync
   ```

2. **Create `.env` file** with your API keys (see below)

3. **Run The Oracle:**
   ```bash
   python graph.py
   ```

## Step 1: Create `.env` File

Create a `.env` file in the project root with this template:

```env
# Anthropic API Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

# Tavily API Configuration
TAVILY_API_KEY=your_tavily_api_key_here

# Supabase Configuration (Optional - for caching and persistence)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-service-role-key-here
ENABLE_CACHING=true
CACHE_TTL_HOURS=24

# Research Configuration (Optional - defaults shown)
MAX_RESEARCH_ITERATIONS=3
QUALITY_THRESHOLD=0.7

# LangSmith Observability (Optional - for tracing and debugging)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT=ResearchAgentv2
ENVIRONMENT=local-dev
```

## Step 2: Add Your API Keys

### Get Your Anthropic API Key

1. Go to https://console.anthropic.com/
2. Sign in or create an account
3. Navigate to **API Keys** section
4. Click **Create Key**
5. Copy the key (starts with `sk-ant-...`)
6. Replace `your_anthropic_api_key_here` in `.env` with your actual key

### Get Your Tavily API Key

1. Go to https://tavily.com/
2. Sign in or create an account
3. Navigate to your dashboard
4. Find your API key (starts with `tvly-...`)
5. Replace `your_tavily_api_key_here` in `.env` with your actual key

### Get Your Supabase Credentials

1. Go to https://supabase.com/
2. Sign in or create an account
3. Create a new project (or use existing)
4. Navigate to **Settings** → **API**
5. Copy:
   - **Project URL** (e.g., `https://xxxxx.supabase.co`)
   - **Service Role Key** (for backend - found under service_role secret)
6. Replace `your_supabase_url_here` and `your_supabase_key_here` in `.env`

**Important**: Use the **Service Role Key** (not anon key) for backend services.

### Create Database Tables

1. In Supabase Dashboard, go to **SQL Editor**
2. Click **New Query**
3. Copy the contents of `db/schema.sql` and paste it
4. Click **Run** to execute

This creates the tables needed for caching and report storage.

## Step 3: Supabase Setup (Optional)

Supabase is optional but recommended for:
- **Caching research plans** (saves LLM calls)
- **Persisting reports** (all research saved to database)

### Setup Supabase

1. **Get credentials** from https://supabase.com/dashboard → Settings → API
2. **Add to `.env`** (see template above)
3. **Create database tables:**
   - Open Supabase Dashboard → SQL Editor
   - Run the SQL from `db/schema.sql`

**Note:** If you skip Supabase, the system will work without caching/persistence.

### Get Your LangSmith API Key (Optional - for Observability)

1. Go to https://smith.langchain.com/
2. Sign in or create an account
3. Navigate to **Settings** → **API Keys**
4. Click **Create API Key**
5. Copy the key
6. Add to `.env`:
   ```env
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your-langsmith-api-key-here
   LANGCHAIN_PROJECT=ResearchAgentv2
   ENVIRONMENT=local-dev
   ```
   
   **Note:** `LANGCHAIN_PROJECT` should match your LangSmith project name. If the project doesn't exist, LangSmith will create it automatically.

**Benefits of LangSmith:**
- **Real-time tracing**: Watch agent execution in real-time
- **Debugging**: See all LLM calls, prompts, and responses
- **Performance metrics**: Track latency, token usage, costs
- **Eval tracking**: All evaluation runs are automatically tagged

**Note:** LangSmith is optional but highly recommended for debugging and monitoring.

## Step 4: Verify Setup

Test your configuration:

```bash
# Test Supabase connection (if configured)
python test_supabase.py

# Run The Oracle
python graph.py
```

## Troubleshooting

### `.env` File Issues

If you see "python-dotenv could not parse statement", check:
- No spaces around `=`: Use `KEY=value` (not `KEY = value`)
- One key-value pair per line
- No special characters that need escaping

Run `python fix_env.py` to diagnose and fix issues.

### Supabase Installation

The full `supabase` package doesn't work on Python 3.14. The system automatically uses `postgrest` (which you already have) - no action needed!

## Security

⚠️ **Important**: The `.env` file is in `.gitignore`. Never commit API keys to git!
