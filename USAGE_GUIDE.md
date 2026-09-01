# ResearchAgentv2 - Complete Usage Guide

## System Overview

ResearchAgentv2 is an evidence-grounded research agent (Phase 3C) with:

- **Backend**: FastAPI service (`api.py`) with NDJSON streaming
- **Frontend**: Next.js Deep Research Console (`research-client/`)
- **Agent**: LangGraph pipeline — Router → (Fast Path | Decision Framer → full pipeline) → Knowledge State → (Option Evaluation → Decision Synthesis) → Writer
- **Persistence**: Optional Supabase (research runs, sources, evidence, claims, verifications)
- **Observability**: LangSmith tracing and per-stage metrics
- **Tests**: 244 passing (`uv run pytest`)

**Full pipeline (STANDARD / DEEP):**

```
Decision Framer → Planner → Researcher → Evidence → Claims → Verification → Critic
  → (loop | Knowledge State)
  → [Option Evaluation → Decision Synthesis]  (when decision_frame present with options)
  → Writer
```

**Trusted research chain:** SOURCE → VALIDATED EVIDENCE → MATERIAL CLAIM → VERIFICATION → KNOWLEDGE STATE

**Trusted decision chain:** DECISION FRAME + KNOWLEDGE STATE → OPTION EVALUATION → DECISION SYNTHESIS

See [docs/architecture.md](docs/architecture.md) for diagrams. See [docs/roadmap.md](docs/roadmap.md) for implemented vs future work.

---

## 🚀 Quick Start

### Option 1: Web UI (Recommended)

1. **Start the frontend**:
   ```bash
   cd research-client
   npm install
   npm run dev
   ```

2. **Configure backend URL**:
   Create `research-client/.env.local`:
   ```env
   NEXT_PUBLIC_BACKEND_URL=https://research-agent-v2-69957378560.us-central1.run.app
   ```

3. **Open browser**: http://localhost:3000

4. **Enter query** and click "RESEARCH"

### Option 2: Command Line

```bash
python run_research.py "Your research query here"
```

### Option 3: Python API

Always bootstrap state with `create_initial_state` (routing, research run, metrics):

```python
from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state, finalize_from_state
import asyncio

async def research(query: str):
    graph = create_graph()
    app = graph.compile()
    state, ctx = await create_initial_state(query)
    result = await app.ainvoke(state, config=create_run_config())
    await finalize_from_state(result, ctx)
    return result

result = asyncio.run(research("Latest AI safety research"))
report = result["final_report"]
knowledge_state = result.get("knowledge_state")       # full pipeline only
decision_frame = result.get("decision_frame")         # STANDARD/DEEP only
option_evaluation = result.get("option_evaluation")   # when options present
decision_synthesis = result.get("decision_synthesis")   # when option eval ran
print(report.content)
if knowledge_state:
    print(knowledge_state["metrics"])
if decision_synthesis:
    print(decision_synthesis["recommendation_status"])
```

> **Note:** Use `create_initial_state` for all programmatic runs so routing, persistence, and metrics initialize correctly.

### Option 4: REST API (Streaming)

The API uses **NDJSON streaming** for real-time progress updates:

```bash
curl -X POST https://research-agent-v2-69957378560.us-central1.run.app/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest developments in quantum computing?"}' \
  --no-buffer
```

**Response Format (NDJSON)**:
- `{"type": "log", "content": "Step completed: planner", "node": "planner"}\n`
- `{"type": "log", "content": "Step completed: researcher", "node": "researcher"}\n`
- `{"type": "result", "report": {...}}\n`
- `{"type": "done"}\n`

**Python Streaming Client**:
```python
import requests
import json

def stream_research(query: str):
    response = requests.post(
        "https://research-agent-v2-69957378560.us-central1.run.app/research",
        json={"query": query},
        stream=True,
        timeout=600
    )
    
    for line in response.iter_lines():
        if line:
            event = json.loads(line)
            if event["type"] == "log":
                print(f"Status: {event['content']}")
            elif event["type"] == "result":
                return event["report"]
            elif event["type"] == "error":
                raise Exception(event["error"])
```

---

## 🎨 Frontend Usage (Deep Research Console)

### Features

- **Non-blocking UI**: Handles requests up to 10 minutes without freezing
- **Accurate Timer**: Uses `requestAnimationFrame` to prevent browser throttling
- **Real-time Updates**: Shows execution time (`T+ [seconds]`) with blinking cursor
- **Smooth Animations**: Framer Motion fade-in when results arrive
- **Error Handling**: Robust error boundaries and timeout handling
- **Terminal Aesthetic**: Black background, green monospace font

### Usage Flow

1. Enter your research query in the input field
2. Click "RESEARCH" button
3. Watch the timer (`T+ X.Xs`) and blinking cursor
4. Results fade in when complete
5. View report, sources, confidence score, and metadata
6. Click "New Query" to start over

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file in project root:

```env
# Required
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...

# Optional - Supabase (persistence + plan cache)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=xxx
ENABLE_CACHING=true
CACHE_TTL_HOURS=24
# Also run db/migrations/001_evidence_foundation.sql for evidence/claim tables

# Optional - LangSmith (for observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_PROJECT=ResearchAgentv2
ENVIRONMENT=local-dev

# Optional - Research Settings
MAX_RESEARCH_ITERATIONS=3
QUALITY_THRESHOLD=0.7
```

### Adjust Quality Thresholds

```env
QUALITY_THRESHOLD=0.8  # Higher = stricter quality requirements
MAX_RESEARCH_ITERATIONS=5  # More refinement cycles
```

---

## Testing & Phase Validation

### Unit / integration tests (no live LLM for core logic)

```bash
uv run pytest              # 244 tests
uv run pytest tests/test_knowledge_state.py -v
uv run pytest tests/test_claim_verification.py -v
uv run pytest tests/test_decision_synthesis.py -v
```

### Evaluation (requires API keys)

```bash
uv run python run_eval.py
```

### Phase inspection scripts (constrained / manual)

| Script | Inspects |
|--------|----------|
| `scripts/validate_phase_2a5_e2e.py` | Evidence extraction |
| `scripts/validate_phase_2b_e2e.py` | Claim extraction |
| `scripts/validate_phase_2b7_e2e.py` | SIMPLE_FACT fast path |
| `scripts/validate_phase_2c_e2e.py` | Cross-source verification |
| `scripts/validate_phase_2d_e2e.py` | Knowledge State buckets |
| `scripts/validate_phase_3a_e2e.py` | Decision Framing |
| `scripts/validate_phase_3b_live.py` | Option Evaluation (isolated live) |
| `scripts/validate_phase_3c_live.py` | Decision Synthesis (isolated live) |

```bash
uv run python scripts/validate_phase_3c_live.py
uv run python scripts/validate_phase_3a_e2e.py "Should we adopt Kubernetes or serverless?"
```

Isolated live validation has been completed for Decision Framing, Option Evaluation, and Decision Synthesis. The Writer does not yet surface `DecisionSynthesis` in the user-facing report.

---

## Advanced Usage

### 1. Evaluation System

Run automated evaluation against golden dataset:

```bash
python run_eval.py
```

This will:
- Load test cases from `tests/golden_dataset.json`
- Run all queries in parallel (max 5 concurrent)
- Use LLM-as-a-Judge to grade responses
- Generate comprehensive report with accuracy metrics

### 2. Streaming Results (Direct LangGraph)

For direct LangGraph streaming (bypassing the API):

```python
from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state

async def stream_research(query: str):
    graph = create_graph()
    app = graph.compile()
    state, _ctx = await create_initial_state(query)
    run_config = create_run_config()  # For LangSmith tracing

    async for chunk in app.astream(state, config=run_config):
        node_name = list(chunk.keys())[0]
        state = chunk[node_name]
        
        if "research_plan" in state and state["research_plan"]:
            print(f"📋 Plan: {state['research_plan'].sub_queries}")
        
        if "research_results" in state and state["research_results"]:
            print(f"🔍 Found {state['research_results'].total_count} results")
        
        if "final_report" in state and state["final_report"]:
            yield state["final_report"]
```

**Note**: The production API (`api.py`) uses a **queue-based event generator** that shields LangGraph from client disconnects, ensuring LangSmith traces always finalize properly.

### 3. Error Recovery & Retry

```python
from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state, finalize_from_state

async def robust_research(query: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            graph = create_graph()
            app = graph.compile()
            state, ctx = await create_initial_state(query)
            result = await app.ainvoke(state, config=create_run_config())
            await finalize_from_state(result, ctx)

            if result.get("error"):
                raise Exception(result["error"])

            return result

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}, retrying...")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### 4. State Inspection & Debugging

```python
from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state

async def debug_research(query: str):
    graph = create_graph()
    app = graph.compile()
    state, _ctx = await create_initial_state(query)

    async for chunk in app.astream(state, config=create_run_config()):
        node_name = list(chunk.keys())[0]
        node_state = chunk[node_name]
        
        print(f"\n{'='*50}")
        print(f"Node: {node_name}")
        print(f"{'='*50}")
        
        if node_name == "planner" and node_state.get("research_plan"):
            plan = node_state["research_plan"]
            print(f"Sub-queries: {plan.sub_queries}")
            print(f"Search terms: {plan.search_terms}")
            print(f"Domains: {plan.domains}")
        
        if node_name == "researcher" and node_state.get("research_results"):
            results = node_state["research_results"]
            print(f"Total results: {results.total_count}")
            for i, r in enumerate(results.results[:3], 1):
                print(f"  {i}. {r.title[:60]}... (score: {r.score:.2f})")
        
        if node_name == "critic" and node_state.get("critique"):
            critique = node_state["critique"]
            print(f"Quality score: {critique.quality_score:.2f}")
            print(f"Sufficient: {critique.is_sufficient}")
            if critique.unsupported_areas:
                print(f"Unsupported areas: {critique.unsupported_areas}")
            if critique.issues:
                print(f"Issues: {critique.issues}")

        if node_name == "knowledge_state" and node_state.get("knowledge_state"):
            ks = node_state["knowledge_state"]
            print(f"Knowledge metrics: {ks.get('metrics', {})}")

        if node_name == "decision_framer" and node_state.get("decision_frame"):
            df = node_state["decision_frame"]
            print(f"Decision: {df.get('decision', '')[:80]}")
            print(f"Options: {len(df.get('options', []))}, Criteria: {len(df.get('criteria', []))}")

        if node_name == "option_evaluator" and node_state.get("option_evaluation"):
            oe = node_state["option_evaluation"]
            print(f"Option evaluations: {len(oe.get('evaluations', []))}")

        if node_name == "decision_synthesizer" and node_state.get("decision_synthesis"):
            ds = node_state["decision_synthesis"]
            print(f"Recommendation status: {ds.get('recommendation_status')}")
            print(f"Recommended option: {ds.get('recommended_option')}")

        if node_name == "writer" and node_state.get("final_report"):
            report = node_state["final_report"]
            print(f"Report length: {len(report.content)} chars")
            print(f"Sources: {len(report.sources)}")
            print(f"Confidence: {report.confidence:.2f}")
```

---

## 🎨 Integration Patterns

### 1. Web API Integration (Streaming)

The production FastAPI service (`api.py`) uses **NDJSON streaming** for real-time updates:

```python
import requests
import json

def research_via_api(query: str):
    """Stream research results from the API."""
    response = requests.post(
        "https://research-agent-v2-69957378560.us-central1.run.app/research",
        json={"query": query},
        stream=True,
        timeout=600  # 10 minutes
    )
    
    for line in response.iter_lines():
        if line:
            event = json.loads(line)
            if event["type"] == "log":
                print(f"Progress: {event['content']}")
            elif event["type"] == "result":
                return event["report"]
            elif event["type"] == "error":
                raise Exception(event["error"])
            elif event["type"] == "done":
                break
    
    return None
```

**Key Features**:
- **Queue-based shielding**: LangGraph runs in a background task, isolated from HTTP stream
- **Graceful disconnects**: Client disconnects don't interrupt LangGraph execution
- **LangSmith compatibility**: Traces always finalize (green checkmark)
- **Real-time progress**: `log` events show current node execution

### 2. Batch Processing

```python
from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state, finalize_from_state

async def batch_research(queries: list[str]):
    graph = create_graph()
    app = graph.compile()

    results = []
    for query in queries:
        state, ctx = await create_initial_state(query)
        result = await app.ainvoke(state, config=create_run_config())
        await finalize_from_state(result, ctx)
        results.append(result)

    return results
```

### 3. Scheduled Research Jobs

```python
import asyncio
from datetime import datetime
from graph import create_graph, create_run_config
from services.pipeline_init import create_initial_state, finalize_from_state

async def scheduled_research(query: str, schedule_name: str):
    """Run research and save results with timestamp."""
    graph = create_graph()
    app = graph.compile()
    state, ctx = await create_initial_state(query)
    result = await app.ainvoke(state, config=create_run_config())
    await finalize_from_state(result, ctx)
    report = result["final_report"]
    
    # Save with timestamp
    filename = f"research_{schedule_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w") as f:
        f.write(f"# Research Report: {query}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(report.content)
        f.write(f"\n\n## Sources\n")
        for source in report.sources:
            f.write(f"- {source}\n")
    
    return filename
```

---

## 📊 Observability & Monitoring

### LangSmith Tracing

1. **Enable tracing** in `.env`:
   ```env
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=ls-...
   LANGCHAIN_PROJECT=ResearchAgentv2
   ```

2. **View traces**: When you run research, the system prints:
   ```
   🛠️  View Trace: https://smith.langchain.com/o/<org-id>/projects/p/ResearchAgentv2
   ```

3. **Filter traces**: Use tags like `eval-run`, `eval-factual`, etc. for evaluation runs

### Structured Logging

All LLM calls are logged with:
- Input prompts (PII redacted)
- Output responses
- Latency metrics
- Error details

---

## 🎓 Best Practices

### 1. Query Formulation
- **Be specific**: "Latest developments in AI safety" > "AI safety"
- **Include context**: "Quantum error correction in 2024" > "Quantum computing"
- **Set scope**: "Enterprise AI adoption trends" > "AI"

### 2. Quality Threshold Tuning
- **High-stakes research**: `QUALITY_THRESHOLD=0.8` (stricter)
- **Quick research**: `QUALITY_THRESHOLD=0.6` (faster)
- **Balanced**: `QUALITY_THRESHOLD=0.7` (default)

### 3. Iteration Limits
- **Deep research**: `MAX_RESEARCH_ITERATIONS=5`
- **Quick reports**: `MAX_RESEARCH_ITERATIONS=2`
- **Balanced**: `MAX_RESEARCH_ITERATIONS=3` (default)

### 4. Error Handling
Always check for errors:
```python
state, ctx = await create_initial_state(query)
result = await app.ainvoke(state, config=create_run_config())
await finalize_from_state(result, ctx)
if result.get("error"):
    raise Exception(f"Research failed: {result['error']}")
```

---

## 🚨 Common Pitfalls & Solutions

### Issue: Too Many API Calls
**Solution**: Reduce `MAX_RESEARCH_ITERATIONS` or increase `QUALITY_THRESHOLD`

### Issue: Reports Too Generic
**Solution**: Make queries more specific, adjust planner prompts

### Issue: Slow Execution
**Solution**: 
- Use SIMPLE_FACT phrasing for narrow factual questions (fast path)
- Reduce `max_iterations` via query complexity (router assigns budgets)
- Reduce search result limits in researcher budget

### Issue: Quality Always Insufficient
**Solution**: Lower `QUALITY_THRESHOLD` or improve search query quality

### Issue: Frontend Timeout
**Solution**: The API uses streaming (NDJSON) which prevents timeouts. The frontend Edge runtime supports long-running streams. If you need longer than 10 minutes, increase Cloud Run timeout: `--timeout 1200` (20 minutes)

### Issue: LangSmith Traces Stay "Pending"
**Solution**: The queue-based event generator ensures LangGraph always completes even if the client disconnects. Traces will finalize automatically. Check logs for "LangGraph internal stream finished."

---

## 📈 Deployment

### Backend (Google Cloud Run)

1. **Build Docker image**:
   ```bash
   docker build -t gcr.io/YOUR_PROJECT/research-agent:latest .
   ```

2. **Push to GCR**:
   ```bash
   docker push gcr.io/YOUR_PROJECT/research-agent:latest
   ```

3. **Deploy**:
   ```bash
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

**Health Check**: The service exposes `/health` endpoint for Cloud Run probes:
```bash
curl https://your-service.run.app/health
# Returns: {"status": "ok"}
```

### Frontend (Vercel)

1. **Deploy to Vercel**:
   ```bash
   cd research-client
   vercel deploy
   ```

2. **Set environment variable**:
   - `NEXT_PUBLIC_BACKEND_URL`: Your Cloud Run URL

---

## 🎯 Pro Tips

1. **Combine with other tools**: Use ResearchAgentv2 for research, then feed results to other AI systems
2. **Iterative refinement**: Use the output as input for follow-up queries
3. **Multi-angle research**: Run multiple queries on the same topic from different angles
4. **Source verification**: Use the source URLs for manual verification when needed
5. **Confidence scores**: Use `report.confidence` to gauge reliability
6. **Evaluation**: Run `python run_eval.py` regularly to track system performance

---

## 📚 Example Workflows

### Academic Research Workflow
```python
# 1. Broad research
broad = await research("Quantum computing applications")

# 2. Deep dive on specific aspect
deep = await research(f"Error correction in {broad['final_report'].content[:100]}")

# 3. Compare perspectives
compare = await research("Quantum vs classical computing advantages")
```

### Competitive Analysis Workflow
```python
topics = [
    "Company X product strategy",
    "Company X market position",
    "Company X recent announcements"
]

reports = await batch_research(topics)
# Combine and analyze reports
```

### Evaluation Workflow
```bash
# Run evaluation suite
python run_eval.py

# Review results in terminal and LangSmith
# Check accuracy, latency, and category breakdowns
```

---

ResearchAgentv2 is a **self-correcting, evidence-grounded research system**. The full pipeline produces validated evidence, verified material claims, a deterministic Knowledge State, and—for decision-oriented queries—structured option evaluation and decision synthesis. The Writer produces a cited report but **does not yet consume DecisionSynthesis**. See [docs/architecture.md](docs/architecture.md) for current limitations.
