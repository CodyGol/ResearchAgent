# ResearchAgentv2

A production-grade evidence-grounded research agent with a Next.js frontend. Routes questions by complexity, runs a trusted factual pipeline (sources → evidence → claims → verification → knowledge state), and synthesizes cited reports.

**Current implementation:** Phase 2D (Knowledge State). **178 tests passing.**

## Quick Start

### Web UI

```bash
cd research-client && npm install && npm run dev
```

Set `NEXT_PUBLIC_BACKEND_URL` in `research-client/.env.local`, then open http://localhost:3000.

### Command Line

```bash
uv sync
uv run python run_research.py "Your research query here"
```

### REST API (NDJSON streaming)

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest developments in quantum computing?"}' \
  --no-buffer
```

## Architecture (Phase 2D)

```
User Query → Router

SIMPLE_FACT:
  Fast Path → Fact Target → Targeted Search → Decisive Evidence
           → Structured Fact Value → Validation → Canonical Claim → Concise Answer → END

STANDARD / DEEP:
  Planner → Researcher → Evidence Extractor → Claim Extractor → Claim Verifier → Critic
    ├── insufficient / under budget → Researcher (loop)
    └── sufficient / max iterations → Knowledge State → Writer → END
```

**Trusted chain:** SOURCE → VALIDATED EVIDENCE → DIRECT ATOMIC CLAIM → MATERIAL CLAIM → CROSS-SOURCE VERIFICATION → KNOWLEDGE STATE → REPORT

| Route | Purpose |
|-------|---------|
| `simple_fact` | Frozen fast-fact path; escalates to full pipeline on failure |
| `standard` / `deep` | Full pipeline with configurable research budgets |

Full diagrams and node details: **[docs/architecture.md](docs/architecture.md)**  
Phase status and future work: **[docs/roadmap.md](docs/roadmap.md)**

## Key Capabilities

- **Adaptive routing** — SIMPLE_FACT / STANDARD / DEEP with per-route research budgets
- **Evidence integrity** — verbatim spans validated against source content
- **Atomic claims** — extraction, materiality filter, claim–evidence links (`supports`, `contradicts`, `qualifies`)
- **Cross-source verification** — independent publisher-domain check (approximation)
- **Knowledge State** — deterministic buckets: known, likely, disputed, unknown, contradicted, unverifiable + critic gap hints
- **Recursive refinement** — Critic loop with iteration budget
- **Supabase persistence** — research runs, sources, evidence, claims, verifications (optional)
- **Observability** — LangSmith tracing, per-stage metrics

## Known Limitations

- Successful **SIMPLE_FACT** runs do not receive Knowledge State (no verification step)
- **Writer** does not yet consume verification or Knowledge State
- **Critic** does not consume verification results
- Publisher-domain independence is an approximation, not true source lineage
- Cross-domain verification quality depends on retrieval diversity
- No full caching system for evidence/claims; plan cache only (`ENABLE_CACHING`)
- **Decision Engine**, monitoring, and change detection are **not implemented**

## Setup

**Prerequisites:** Python 3.12+, `uv` (or pip), Node.js 18+ (for UI)

```env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
SUPABASE_URL=https://xxx.supabase.co          # optional
SUPABASE_KEY=xxx                              # optional
LANGCHAIN_TRACING_V2=true                     # optional
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_PROJECT=ResearchAgentv2
```

```bash
uv sync
# Supabase: run db/schema.sql then db/migrations/001_evidence_foundation.sql
```

See **[SETUP.md](SETUP.md)** and **[USAGE_GUIDE.md](USAGE_GUIDE.md)** for details.

## Python API

Use `create_initial_state` so routing, research runs, and metrics initialize correctly:

```python
import asyncio
from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state, finalize_from_state

async def research(query: str):
    graph = create_graph()
    app = graph.compile()
    state, ctx = await create_initial_state(query)
    result = await app.ainvoke(state, config=create_run_config())
    await finalize_from_state(result, ctx)
    return result

report = asyncio.run(research("Who won the 2023 Formula 1 World Championship?"))["final_report"]
```

Inspect Knowledge State on full-pipeline runs:

```python
ks = result.get("knowledge_state")  # None for successful fast-path runs
```

## Testing

```bash
uv run pytest                    # 178 tests (no live LLM for core logic)
uv run python run_eval.py        # golden dataset + LLM-as-Judge (requires API keys)
```

Phase validation scripts (manual inspection):

```bash
uv run python scripts/validate_phase_2c_e2e.py "your query"
uv run python scripts/validate_phase_2d_e2e.py "your query"
```

## Project Structure

```
├── graph.py                 # LangGraph StateGraph
├── state.py                 # AgentState
├── api.py                   # Production FastAPI (NDJSON streaming)
├── run_research.py          # CLI entry point
├── nodes/                   # router, fast_path, planner, researcher, evidence_extractor,
│                            # claim_extractor, claim_verifier, critic, knowledge_state, writer
├── services/                # query_router, evidence/claim/verification/knowledge_state pipelines
├── domain/models.py         # Source, Evidence, Claim, VerificationResult, ...
├── db/                      # Supabase client, repositories, migrations
├── tests/                   # 178 unit/integration tests
├── scripts/validate_phase_*  # Phase E2E inspection scripts
├── docs/architecture.md
└── docs/roadmap.md
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/architecture.md](docs/architecture.md) | Engineering architecture, data flow, state, persistence |
| [docs/roadmap.md](docs/roadmap.md) | Implemented phases vs Phase 3+ |
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | Usage patterns, streaming, observability |
| [SETUP.md](SETUP.md) | Environment and Supabase setup |
| [db/README.md](db/README.md) | Database tables and repositories |

## Deployment

Backend: Docker → Google Cloud Run (`Dockerfile`, `DOCKER.md`). Frontend: Vercel (`research-client/`).

Health check: `GET /health` → `{"status": "ok"}`

## Design Principles

- **Composition > Inheritance** — node-based pipeline
- **Schema > Guesswork** — Pydantic V2 everywhere
- **Tracing > Logging** — LangSmith + structured metrics
- **Fail loudly** — retryable vs fatal errors; no silent drops
