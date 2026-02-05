# The Oracle - Usage & Leverage Guide

## 🎯 Core Use Cases

### 1. Deep Research Reports
Perfect for comprehensive research on complex topics that require:
- Multiple angles and perspectives
- Recent developments verification
- Source diversity and credibility checks
- Self-correcting quality assurance

**Example:**
```python
query = "What are the latest developments in quantum computing error correction?"
```

### 2. Competitive Intelligence
Research competitors, market trends, or industry analysis with automatic quality checks.

### 3. Technical Deep Dives
Get comprehensive technical reports on emerging technologies, frameworks, or methodologies.

### 4. Academic Research Support
Generate well-sourced research reports with citations for academic or professional work.

---

## 🚀 Basic Usage

### Simple Execution

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
    return result

# Run it
result = asyncio.run(research("Latest AI safety research"))
print(result["final_report"].content)
```

### Batch Processing

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

---

## ⚙️ Customization & Configuration

### 1. Adjust Quality Thresholds

Edit `.env`:
```env
QUALITY_THRESHOLD=0.8  # Higher = stricter quality requirements
MAX_RESEARCH_ITERATIONS=5  # More refinement cycles
```

### 2. Customize Node Behavior

#### Modify Planner Prompts
Edit `nodes/planner.py` to change how research plans are generated:
- Add domain-specific instructions
- Change sub-query generation strategy
- Adjust search term extraction

#### Customize Critic Evaluation
Edit `nodes/critic.py` to:
- Add custom quality criteria
- Change evaluation weights
- Modify freshness requirements

#### Tune Writer Output
Edit `nodes/writer.py` to:
- Change report structure
- Adjust synthesis style
- Modify citation format

### 3. Add Custom Search Filters

Modify `tools/search.py` to:
- Filter by date ranges
- Add domain restrictions
- Customize result ranking

---

## 🔧 Advanced Patterns

### 1. Streaming Results

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
        # Process intermediate states
        node_name = list(chunk.keys())[0]
        state = chunk[node_name]
        
        if "research_plan" in state and state["research_plan"]:
            print(f"📋 Plan: {state['research_plan'].sub_queries}")
        
        if "research_results" in state and state["research_results"]:
            print(f"🔍 Found {state['research_results'].total_count} results")
        
        if "final_report" in state and state["final_report"]:
            yield state["final_report"]
```

### 2. Error Recovery & Retry

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

### 3. State Inspection & Debugging

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

### 1. Web API Wrapper

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ResearchRequest(BaseModel):
    query: str

class ResearchResponse(BaseModel):
    report: str
    sources: list[str]
    confidence: float

@app.post("/research", response_model=ResearchResponse)
async def research_endpoint(request: ResearchRequest):
    from graph import create_graph
    
    graph = create_graph()
    app_graph = graph.compile()
    
    state = {
        "user_query": request.query,
        "research_plan": None,
        "research_results": None,
        "critique": None,
        "final_report": None,
        "current_node": "planner",
        "iteration_count": 0,
        "error": None,
    }
    
    result = await app_graph.ainvoke(state)
    
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    
    report = result["final_report"]
    return ResearchResponse(
        report=report.content,
        sources=report.sources,
        confidence=report.confidence
    )
```

### 2. CLI Tool

```python
# cli.py
import argparse
import asyncio
from graph import create_graph

async def main():
    parser = argparse.ArgumentParser(description="The Oracle Research Agent")
    parser.add_argument("query", help="Research query")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    graph = create_graph()
    app = graph.compile()
    
    state = {
        "user_query": args.query,
        "research_plan": None,
        "research_results": None,
        "critique": None,
        "final_report": None,
        "current_node": "planner",
        "iteration_count": 0,
        "error": None,
    }
    
    if args.verbose:
        async for chunk in app.astream(state):
            print(f"Processing: {list(chunk.keys())[0]}")
    else:
        result = await app.ainvoke(state)
        report = result["final_report"]
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(report.content)
            print(f"Report saved to {args.output}")
        else:
            print(report.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Scheduled Research Jobs

```python
# scheduler.py
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

## 📊 Performance Optimization

### 1. Parallel Sub-Query Processing

Modify `nodes/researcher.py` to process sub-queries in parallel:

```python
import asyncio

# In researcher_node:
all_results = await asyncio.gather(*[
    search_tavily_with_retry(query=sq, max_results=5)
    for sq in plan.sub_queries
], return_exceptions=True)

# Flatten and handle errors
results = []
for r in all_results:
    if isinstance(r, Exception):
        # Handle error
        continue
    results.extend(r)
```

### 2. Cache Research Plans

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def get_cached_plan(query_hash: str):
    # Cache logic here
    pass

# In planner_node:
query_hash = hashlib.md5(query.encode()).hexdigest()
if cached := get_cached_plan(query_hash):
    return cached
```

### 3. Rate Limiting

Add rate limiting to avoid API quota issues:

```python
from asyncio import Semaphore

# In config.py or separate module
search_semaphore = Semaphore(5)  # Max 5 concurrent searches

# In search_tavily_with_retry:
async with search_semaphore:
    return await search_tavily(query, max_results)
```

---

## 🔍 Monitoring & Observability

### 1. Track Execution Metrics

```python
import time
from collections import defaultdict

metrics = defaultdict(list)

async def tracked_research(query: str):
    start_time = time.time()
    
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
        node_start = time.time()
        # ... process ...
        node_duration = time.time() - node_start
        metrics[node_name].append(node_duration)
    
    total_time = time.time() - start_time
    metrics["total"].append(total_time)
    
    return metrics
```

### 2. Export Traces

The system already logs traces. To export them:

```python
import json
import logging

class JSONFileHandler(logging.Handler):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
    
    def emit(self, record):
        with open(self.filename, "a") as f:
            f.write(record.getMessage() + "\n")

# Add to utils/observability.py
logger.addHandler(JSONFileHandler("traces.jsonl"))
```

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
    # Handle error appropriately
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
- Cache research plans

### Issue: Quality Always Insufficient
**Solution**: Lower `QUALITY_THRESHOLD` or improve search query quality

---

## 📈 Scaling Strategies

1. **Horizontal Scaling**: Run multiple instances with different queries
2. **Queue System**: Use Celery/RQ for background processing
3. **Database Storage**: Save reports and plans for reuse
4. **API Gateway**: Rate limit and authenticate requests

---

## 🎯 Pro Tips

1. **Combine with other tools**: Use The Oracle for research, then feed results to other AI systems
2. **Iterative refinement**: Use The Oracle's output as input for follow-up queries
3. **Multi-angle research**: Run multiple queries on the same topic from different angles
4. **Source verification**: Use the source URLs for manual verification when needed
5. **Confidence scores**: Use `report.confidence` to gauge reliability

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

---

The Oracle is designed to be a **self-correcting, quality-assured research system**. Leverage its recursive refinement and quality checks to get comprehensive, well-sourced research reports automatically.
