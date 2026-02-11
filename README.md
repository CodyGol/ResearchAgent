# ResearchAgentv2

A production-grade recursive deep-research agent system with a Generative UI frontend. Plans research paths, executes searches, critiques findings, and synthesizes comprehensive reports.

## 🚀 Quick Start

### Option 1: Web UI (Recommended)

1. **Start the frontend**:
   ```bash
   cd research-client
   npm install
   npm run dev
   ```

2. **Configure backend URL** in `research-client/.env.local`:
   ```env
   NEXT_PUBLIC_BACKEND_URL=https://research-agent-v2-69957378560.us-central1.run.app
   ```

3. **Open browser**: http://localhost:3000

### Option 2: Command Line

```bash
python run_research.py "Your research query here"
```

### Option 3: REST API (Streaming)

The API uses **NDJSON streaming** for real-time progress:

```bash
curl -X POST https://research-agent-v2-69957378560.us-central1.run.app/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest developments in quantum computing?"}' \
  --no-buffer
```

**Response**: Streams NDJSON events (`log`, `result`, `done`) for real-time updates.

## Architecture

ResearchAgentv2 uses a LangGraph state machine with four core nodes:

1. **Planner**: Analyzes user query and generates structured research plan with domain filters
2. **Researcher**: Executes web searches via Tavily API with retry logic and spam filtering
3. **Critic**: Evaluates research quality and decides if refinement is needed
4. **Writer**: Synthesizes final report from approved research

The Critic node implements a recursive loop: if quality is insufficient, it routes back to Researcher for refinement (up to `MAX_RESEARCH_ITERATIONS`).

**System Components:**
- **Backend**: FastAPI service (`api.py`) with queue-based streaming, deployed on Google Cloud Run
- **Frontend**: Next.js Generative UI (`research-client/`) with Edge runtime and NDJSON streaming
- **Agent**: LangGraph state machine with recursive refinement
- **Streaming**: NDJSON event stream with shielded event generator (prevents GeneratorExit)
- **Observability**: LangSmith tracing (always finalizes) and structured logging
- **Evaluation**: Automated testing with LLM-as-a-Judge (`run_eval.py`)

See [docs/architecture.md](docs/architecture.md) for the full architecture diagram.

## Setup

### Prerequisites

- Python 3.12+
- `uv` package manager (or `pip`)
- Node.js 18+ (for frontend)

### Installation

1. **Clone the repository**

2. **Create `.env` file** with your API keys:
   ```env
   ANTHROPIC_API_KEY=sk-ant-...
   TAVILY_API_KEY=tvly-...
   SUPABASE_URL=https://xxx.supabase.co (optional)
   SUPABASE_KEY=xxx (optional)
   LANGCHAIN_TRACING_V2=true (optional)
   LANGCHAIN_API_KEY=ls-... (optional)
   LANGCHAIN_PROJECT=ResearchAgentv2 (optional)
   ```

3. **Install Python dependencies**:
   ```bash
   uv sync
   # or
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies** (if using web UI):
   ```bash
   cd research-client
   npm install
   ```

See [SETUP.md](SETUP.md) for detailed setup instructions.

## Usage

### Web UI

The **Deep Research Console** provides a terminal-style interface:
- Non-blocking long-running requests (up to 10 minutes)
- Real-time execution timer using `requestAnimationFrame`
- Smooth fade-in animations for results
- React Markdown rendering with error boundaries

### Command Line

```bash
# Run with default query
python run_research.py

# Run with custom query
python run_research.py "Your research question here"
```

### Python API

```python
from graph import create_graph
import asyncio

async def research(query: str):
    graph = create_graph()
    app = graph.compile()
    
    initial_state = {
        "user_query": query,
        "research_plan": None,
        "research_results": None,
        "critique": None,
        "final_report": None,
        "current_node": "planner",
        "iteration_count": 0,
        "error": None,
    }
    
    result = await app.ainvoke(initial_state)
    return result["final_report"]

# Run it
report = asyncio.run(research("Your research query here"))
print(report.content)
```

### Evaluation System

Run automated evaluation against golden dataset:

```bash
python run_eval.py
```

This evaluates the agent against test cases in `tests/golden_dataset.json` using LLM-as-a-Judge, providing accuracy metrics and performance analysis.

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for comprehensive usage examples, integration patterns, and best practices.

## Project Structure

```
.
├── api.py                   # Production FastAPI service (Cloud Run)
├── config.py                # Pydantic-settings configuration
├── state.py                 # AgentState and Pydantic models
├── graph.py                 # LangGraph StateGraph definition
├── run_research.py          # CLI script for research
├── run_eval.py              # Evaluation system with LLM-as-a-Judge
├── server.py                # Legacy FastAPI server (use api.py for production)
├── Dockerfile               # Production Docker image for Cloud Run
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project configuration (uv)
│
├── nodes/                   # Agent node implementations
│   ├── planner.py           # Research plan generation
│   ├── researcher.py        # Web search execution
│   ├── critic.py            # Quality evaluation
│   └── writer.py            # Report synthesis
│
├── tools/                   # Utility tools
│   └── search.py            # Tavily search with retry & spam filtering
│
├── db/                      # Supabase integration
│   ├── client.py            # Database client
│   ├── models.py            # Database models
│   ├── repository.py        # Data access layer
│   └── schema.sql           # Database schema
│
├── utils/                   # Utilities
│   ├── observability.py     # Tracing and logging
│   ├── pii_redaction.py     # PII redaction
│   └── serialization.py     # JSON serialization helpers
│
├── tests/                   # Test suite
│   ├── test_logic.py        # Unit tests (blacklist, serialization, state)
│   └── golden_dataset.json  # Evaluation test cases
│
├── research-client/         # Next.js frontend
│   ├── app/
│   │   ├── page.tsx         # Main UI component
│   │   ├── api/
│   │   │   └── research/
│   │   │       └── route.ts # API proxy with timeout
│   │   └── components/
│   │       └── ErrorBoundary.tsx
│   └── package.json
│
├── docs/
│   └── architecture.md     # Architecture diagram
│
├── test_supabase.py         # Supabase connection test
├── fix_env.py               # .env file diagnostic tool
└── pytest.ini               # Pytest configuration
```

## Features

### Core Agent
- ✅ **Recursive Refinement**: Self-correcting research with quality checks
- ✅ **Domain Filtering**: Prioritizes primary sources (arxiv.org, github.com, etc.)
- ✅ **Spam Filtering**: Blacklists SEO blogs (Medium, LinkedIn, etc.)
- ✅ **Structured Output**: Pydantic V2 models for all data
- ✅ **Error Handling**: Retryable vs Fatal error categorization

### Production Features
- ✅ **FastAPI Service**: Production-ready API with health checks and CORS
- ✅ **Streaming Architecture**: NDJSON event streaming with queue-based shielding
- ✅ **Docker Support**: Optimized container for Cloud Run (Python 3.12-slim)
- ✅ **Next.js Frontend**: Generative UI with Edge runtime and streaming support
- ✅ **LangSmith Tracing**: Full observability with guaranteed trace finalization
- ✅ **Supabase Integration**: Caching and persistence
- ✅ **Evaluation System**: Automated testing with LLM-as-a-Judge

### Observability
- ✅ **Structured Logging**: JSON logs with PII redaction
- ✅ **LangSmith Integration**: Real-time trace viewing
- ✅ **Performance Metrics**: Latency tracking per node
- ✅ **Error Tracking**: Categorized errors with tracebacks

## Development Status

**Current Phase**: Production Ready ✅

- ✅ State machine scaffolding
- ✅ Real LLM integration (Claude 4.5 Sonnet)
- ✅ Real Tavily API integration
- ✅ Supabase integration (caching & persistence)
- ✅ Retry logic with structured error handling
- ✅ Observability/tracing with PII redaction
- ✅ LangSmith integration
- ✅ Production FastAPI service
- ✅ Next.js Generative UI
- ✅ Evaluation system
- ✅ Docker/Cloud Run deployment
- ✅ Unit tests (logic verification)
- ✅ CI/CD pipeline

## Deployment

### Backend (Google Cloud Run)

```bash
# Build and push
docker build -t gcr.io/YOUR_PROJECT/research-agent:latest .
docker push gcr.io/YOUR_PROJECT/research-agent:latest

# Deploy
gcloud run deploy research-agent \
  --image gcr.io/YOUR_PROJECT/research-agent:latest \
  --platform managed \
  --region us-central1 \
  --set-env-vars TAVILY_API_KEY=xxx,ANTHROPIC_API_KEY=xxx \
  --timeout 600 \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated
```

**Health Check**: `/health` endpoint returns `{"status": "ok"}` for Cloud Run probes.

### Frontend (Vercel)

```bash
cd research-client
vercel deploy
```

Set environment variable: `NEXT_PUBLIC_BACKEND_URL`

## Design Principles

This project follows strict architectural principles:

- **Composition > Inheritance**: Node-based architecture
- **Schema > Guesswork**: Pydantic V2 models for all data
- **Tracing > Logging**: Structured observability with PII redaction
- **Idempotency**: All operations are retryable
- **Fail Loudly**: Structured error handling (Retryable vs Fatal)
- **Observability First**: LangSmith tracing for all LLM calls
- **Production Ready**: Docker, health checks, error boundaries

## Documentation

- **[SETUP.md](SETUP.md)**: Detailed setup instructions
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)**: Comprehensive usage examples
- **[docs/architecture.md](docs/architecture.md)**: System architecture diagrams
- **[research-client/SETUP.md](research-client/SETUP.md)**: Frontend setup

## Testing

Run the test suite:

```bash
# Logic tests (no API calls)
uv run pytest tests/test_logic.py -v

# Full evaluation (requires API keys)
python run_eval.py
```

