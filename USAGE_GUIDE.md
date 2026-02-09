# ResearchAgentv2 - Complete Usage Guide

## 🎯 System Overview

ResearchAgentv2 is a production-grade recursive deep-research agent system with:
- **Backend**: FastAPI service deployed on Google Cloud Run
- **Frontend**: Next.js Generative UI (Terminal-style interface)
- **Agent**: LangGraph state machine with 4 nodes (Planner → Researcher → Critic → Writer)
- **Observability**: LangSmith tracing and structured logging
- **Evaluation**: Automated testing with LLM-as-a-Judge

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
report = asyncio.run(research("Latest AI safety research"))
print(report.content)
```

### Option 4: REST API

```bash
curl -X POST https://research-agent-v2-69957378560.us-central1.run.app/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest developments in quantum computing?"}'
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

# Optional - Supabase (for caching)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=xxx
ENABLE_CACHING=true
CACHE_TTL_HOURS=24

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

## 🔧 Advanced Usage

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

### 2. Streaming Results

```python
async def stream_research(query: str):
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
    
    async for chunk in app.astream(initial_state):
        node_name = list(chunk.keys())[0]
        state = chunk[node_name]
        
        if "research_plan" in state and state["research_plan"]:
            print(f"📋 Plan: {state['research_plan'].sub_queries}")
        
        if "research_results" in state and state["research_results"]:
            print(f"🔍 Found {state['research_results'].total_count} results")
        
        if "final_report" in state and state["final_report"]:
            yield state["final_report"]
```

### 3. Error Recovery & Retry

```python
async def robust_research(query: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            graph = create_graph()
            app = graph.compile()
            
            state = {
                "user_query": query,
                "research_plan": None,
                "research_results": None,
                "critique": None,
                "final_report": None,
                "current_node": "planner",
                "iteration_count": 0,
                "error": None,
            }
            
            result = await app.ainvoke(state)
            
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
async def debug_research(query: str):
    graph = create_graph()
    app = graph.compile()
    
    state = {
        "user_query": query,
        "research_plan": None,
        "research_results": None,
        "critique": None,
        "final_report": None,
        "current_node": "planner",
        "iteration_count": 0,
        "error": None,
    }
    
    async for chunk in app.astream(state):
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
            if critique.issues:
                print(f"Issues: {critique.issues}")
        
        if node_name == "writer" and node_state.get("final_report"):
            report = node_state["final_report"]
            print(f"Report length: {len(report.content)} chars")
            print(f"Sources: {len(report.sources)}")
            print(f"Confidence: {report.confidence:.2f}")
```

---

## 🎨 Integration Patterns

### 1. Web API Integration

The production FastAPI service (`api.py`) is deployed on Google Cloud Run:

```python
import requests

def research_via_api(query: str):
    response = requests.post(
        "https://research-agent-v2-69957378560.us-central1.run.app/research",
        json={"query": query},
        timeout=600  # 10 minutes
    )
    return response.json()
```

### 2. Batch Processing

```python
async def batch_research(queries: list[str]):
    graph = create_graph()
    app = graph.compile()
    
    results = []
    for query in queries:
        state = {
            "user_query": query,
            "research_plan": None,
            "research_results": None,
            "critique": None,
            "final_report": None,
            "current_node": "planner",
            "iteration_count": 0,
            "error": None,
        }
        result = await app.ainvoke(state)
        results.append(result)
    
    return results
```

### 3. Scheduled Research Jobs

```python
import asyncio
from datetime import datetime
from graph import create_graph

async def scheduled_research(query: str, schedule_name: str):
    """Run research and save results with timestamp."""
    graph = create_graph()
    app = graph.compile()
    
    state = {
        "user_query": query,
        "research_plan": None,
        "research_results": None,
        "critique": None,
        "final_report": None,
        "current_node": "planner",
        "iteration_count": 0,
        "error": None,
    }
    
    result = await app.ainvoke(state)
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
result = await app.ainvoke(state)
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
- Process sub-queries in parallel
- Reduce `max_results` in searches
- Cache research plans (Supabase)

### Issue: Quality Always Insufficient
**Solution**: Lower `QUALITY_THRESHOLD` or improve search query quality

### Issue: Frontend Timeout
**Solution**: The API route has a 10-minute timeout. For longer research, increase `TIMEOUT_MS` in `research-client/app/api/research/route.ts`

---

## 📈 Deployment

### Backend (Google Cloud Run)

1. **Build Docker image**:
   ```bash
   docker build -f Dockerfile -t gcr.io/YOUR_PROJECT/research-agent:latest .
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
     --cpu 2
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

ResearchAgentv2 is designed to be a **self-correcting, quality-assured research system**. Leverage its recursive refinement and quality checks to get comprehensive, well-sourced research reports automatically.
