# ResearchAgentv2

A production-grade evidence-grounded research agent with a Next.js frontend. Routes questions by complexity, runs a trusted research pipeline (sources → evidence → claims → verification → knowledge state), and—for decision-oriented queries—structured option evaluation and decision synthesis before report generation.

**Current implementation:** Phase 3C (Decision Synthesis). **244 tests passing.**

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

## Architecture (Phase 3C)

```
User Query → Router

SIMPLE_FACT:
  Fast Path → Concise Answer → END
  (escalation joins full pipeline at Decision Framer)

STANDARD / DEEP:
  Decision Framer → Planner → Researcher → Evidence → Claims → Verification → Critic
    → (loop | Knowledge State)
    → [Option Evaluation → Decision Synthesis]  (decision runs only)
    → Writer → END
```

**Trusted research chain:** SOURCE → VALIDATED EVIDENCE → MATERIAL CLAIM → CROSS-SOURCE VERIFICATION → KNOWLEDGE STATE

**Trusted decision chain:** DECISION FRAME + KNOWLEDGE STATE → OPTION EVALUATION → DECISION SYNTHESIS

Full diagrams: **[docs/architecture.md](docs/architecture.md)**

| Route | Purpose |
|-------|---------|
| `simple_fact` | Frozen fast-fact path; escalates to full pipeline at Decision Framer |
| `standard` / `deep` | Full pipeline with decision framing, research, and optional decision synthesis |

Phase status and future work: **[docs/roadmap.md](docs/roadmap.md)**

## Key Capabilities

- **Adaptive routing** — SIMPLE_FACT / STANDARD / DEEP with per-route research budgets
- **Evidence integrity** — verbatim spans validated against source content
- **Atomic claims** — extraction, materiality filter, claim–evidence links (`supports`, `contradicts`, `qualifies`)
- **Cross-source verification** — independent publisher-domain check (approximation)
- **Knowledge State** — deterministic buckets: known, likely, disputed, unknown, contradicted, unverifiable + critic gap hints
- **Decision Framing (3A)** — DecisionFrame with options, criteria, priority, constraints, assumptions
- **Option Evaluation (3B)** — evidence-grounded option×criterion assessments with claim lineage
- **Decision Synthesis (3C)** — recommend / tentative / insufficient_basis with constraint assessments and change conditions
- **Recursive refinement** — Critic loop with iteration budget
- **Supabase persistence** — research runs, sources, evidence, claims, verifications (optional)
- **Observability** — LangSmith tracing, per-stage metrics

## Known Limitations

- **Writer does not consume DecisionSynthesis** — the user-facing report does not yet present the structured recommendation
- Decision artifacts (`decision_frame`, `option_evaluation`, `decision_synthesis`) are inspectable via state / metadata / validation scripts only
- Successful **SIMPLE_FACT** runs do not receive Knowledge State or decision artifacts
- **Critic** does not consume verification results
- **Planner / research** is not yet guided by DecisionFrame
- No persistent decision workspace, monitoring, change detection, or automatic re-evaluation
- No actions / execution layer; no numerical utility or weighting system
- Constraint mapping relies on constrained LLM semantic judgment against trusted claims
- Full KnowledgeState claim catalog is used (no relevance-aware truncation)
- Publisher-domain independence is an approximation; cross-domain verification depends on retrieval diversity
- No full caching system for evidence/claims; plan cache only (`ENABLE_CACHING`)

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

Inspect decision artifacts on full decision-oriented runs:

```python
ks = result.get("knowledge_state")           # None for successful fast-path runs
frame = result.get("decision_frame")         # None for SIMPLE_FACT
eval_ = result.get("option_evaluation")      # None when no concrete options
synth = result.get("decision_synthesis")     # None when option evaluation skipped
```

## Testing

```bash
uv run pytest                    # 244 tests (no live LLM for core logic)
uv run python run_eval.py        # golden dataset + LLM-as-Judge (requires API keys)
```

Phase validation scripts (isolated live LLM inspection):

```bash
uv run python scripts/validate_phase_3a_e2e.py "your query"
uv run python scripts/validate_phase_3b_live.py
uv run python scripts/validate_phase_3c_live.py
```

## Project Structure

```
├── graph.py                 # LangGraph StateGraph
├── state.py                 # AgentState
├── api.py                   # Production FastAPI (NDJSON streaming)
├── run_research.py          # CLI entry point
├── nodes/                   # router, fast_path, decision_framer, planner, researcher,
│                            # evidence_extractor, claim_extractor, claim_verifier, critic,
│                            # knowledge_state, option_evaluator, decision_synthesizer, writer
├── services/                # query_router, evidence/claim/verification/knowledge_state,
│                            # decision_framing, option_evaluation, decision_synthesis
├── domain/models.py         # Source, Evidence, Claim, VerificationResult, ...
├── db/                      # Supabase client, repositories, migrations
├── tests/                   # 244 unit/integration tests
├── scripts/validate_phase_*  # Phase E2E inspection scripts
├── docs/architecture.md
└── docs/roadmap.md
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/architecture.md](docs/architecture.md) | Engineering architecture, data flow, state, persistence |
| [docs/roadmap.md](docs/roadmap.md) | Implemented phases vs future productization |
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
