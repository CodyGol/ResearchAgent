# The Oracle

A recursive deep-research agent system that plans research paths, executes searches, critiques findings, and synthesizes comprehensive reports.

## Architecture

The Oracle uses a LangGraph state machine with four core nodes:

1. **Planner**: Analyzes user query and generates structured research plan
2. **Researcher**: Executes web searches via Tavily API with retry logic
3. **Critic**: Evaluates research quality and decides if refinement is needed
4. **Writer**: Synthesizes final report from approved research

The Critic node implements a recursive loop: if quality is insufficient, it routes back to Researcher for refinement (up to `MAX_RESEARCH_ITERATIONS`).

See [docs/architecture.md](docs/architecture.md) for the full architecture diagram.

## Setup

### Prerequisites

- Python 3.12+
- `uv` package manager

### Installation

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

3. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

4. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   ```

## Usage

### Quick Start

Run with the default test query:

```bash
python graph.py
```

### Programmatic Usage

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

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for comprehensive usage examples, integration patterns, and best practices.

See [SETUP.md](SETUP.md) for detailed setup instructions including API keys and Supabase configuration.

## Project Structure

```
.
├── config.py              # Pydantic-settings configuration
├── state.py               # AgentState and Pydantic models
├── graph.py               # LangGraph StateGraph definition
├── nodes/                 # Node implementations
│   ├── planner.py
│   ├── researcher.py
│   ├── critic.py
│   └── writer.py
├── tools/                 # Utility tools
│   └── search.py          # Tavily search with retry logic
├── db/                    # Supabase integration
│   ├── client.py          # Database client
│   ├── models.py          # Database models
│   ├── repository.py      # Data access layer
│   └── schema.sql         # Database schema
├── utils/                 # Utilities
│   ├── observability.py   # Tracing and logging
│   └── pii_redaction.py   # PII redaction
├── docs/
│   └── architecture.md    # Architecture diagram
├── test_supabase.py       # Supabase connection test
├── fix_env.py             # .env file diagnostic tool
└── pyproject.toml         # Project dependencies
```

## Development Status

**Current Phase**: Production Ready
- ✅ State machine scaffolding
- ✅ Real LLM integration (Claude 4.5 Sonnet)
- ✅ Real Tavily API integration
- ✅ Supabase integration (caching & persistence)
- ✅ Retry logic with structured error handling
- ✅ Observability/tracing with PII redaction
- ⏳ Unit tests
- ⏳ Langfuse/Arize integration (optional enhancement)

## Design Principles

This project follows strict architectural principles:

- **Composition > Inheritance**: Node-based architecture
- **Schema > Guesswork**: Pydantic V2 models for all data
- **Tracing > Logging**: Structured observability with PII redaction
- **Idempotency**: All operations are retryable
- **Fail Loudly**: Structured error handling (Retryable vs Fatal)

## License

MIT
