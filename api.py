"""Production FastAPI service for ResearchAgentv2 on Google Cloud Run."""

import asyncio
import json
import logging
import os
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from graph import create_graph, create_run_config
from services.decision_brief import build_decision_brief_payload
from services.pipeline_init import create_initial_state, finalize_from_state
from state import AgentState

# Configure structured logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("research_agent")

# Initialize FastAPI app
app = FastAPI(
    title="ResearchAgentv2",
    description="Production research agent service",
    version="0.1.0",
)

# Enable CORS for frontend (e.g., Vercel deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    """Request model for research queries."""

    query: str = Field(..., description="Research question to investigate", min_length=1)


class ResearchResponse(BaseModel):
    """Response model for research results."""

    query: str
    report: str
    sources: list[str]
    confidence: float
    iteration_count: int
    quality_score: float | None = None
    error: str | None = None


@app.on_event("startup")
async def startup_event() -> None:
    """Validate critical environment variables on startup."""
    missing_keys = []
    
    if not os.environ.get("TAVILY_API_KEY"):
        missing_keys.append("TAVILY_API_KEY")
        logger.critical("⚠️  TAVILY_API_KEY not found in environment variables")
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing_keys.append("ANTHROPIC_API_KEY")
        logger.critical("⚠️  ANTHROPIC_API_KEY not found in environment variables")
    
    if missing_keys:
        logger.critical(
            f"Missing critical API keys: {', '.join(missing_keys)}. "
            "Service will start but research requests will fail."
        )
    else:
        logger.info("✅ All critical API keys found")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint for Cloud Run probes.
    
    Returns:
        {"status": "ok"} if service is healthy
    """
    return {"status": "ok"}


async def event_generator(query: str):
    """
    Generate streaming events from the research agent execution.
    Uses a queue to decouple LangGraph from the HTTP stream, preventing GeneratorExit
    from reaching LangGraph's generator and ensuring traces finalize properly.
    
    Args:
        query: Research query string
        
    Yields:
        NDJSON formatted events with type and content
    """
    # We use a queue to decouple the Graph from the HTTP Stream
    # This prevents the 'GeneratorExit' from ever reaching the LangGraph engine
    queue = asyncio.Queue()

    async def run_graph():
        ctx = None
        current_state: AgentState | None = None
        try:
            logger.info(f"Received query: {query}")

            graph = create_graph()
            app_instance = graph.compile()

            initial_state, ctx = await create_initial_state(query)
            current_state = initial_state

            run_config = create_run_config()

            async for output in app_instance.astream(initial_state, config=run_config):
                await queue.put(output)
                node_name = list(output.keys())[0]
                current_state = {**current_state, **output[node_name]}
            await queue.put("DONE")
        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            await queue.put(f"ERROR: {str(e)}")
        finally:
            if ctx is not None and current_state is not None:
                await finalize_from_state(current_state, ctx)
            logger.info("LangGraph internal stream finished.")

    # Start the graph as a background task that CANNOT be killed by a disconnect
    graph_task = asyncio.create_task(run_graph())

    try:
        final_state = None
        while True:
            # We get data from our local queue, not directly from LangGraph
            item = await queue.get()
            
            if item == "DONE":
                # Verify we got a report
                if not final_state or not final_state.get("final_report"):
                    error_msg = "No report generated"
                    logger.error(error_msg)
                    yield json.dumps({
                        "type": "error",
                        "error": error_msg
                    }) + "\n"
                else:
                    yield json.dumps({"type": "done"}) + "\n"
                break
            elif isinstance(item, str) and item.startswith("ERROR"):
                yield json.dumps({"type": "error", "error": item}) + "\n"
                break
            
            # Standard yielding logic
            # Get the node name from the event
            node_name = list(item.keys())[0] if item else None
            node_state = item[node_name] if node_name else {}
            
            # Yield status update
            yield json.dumps({
                "type": "log",
                "content": f"Step completed: {node_name}",
                "node": node_name
            }) + "\n"
            
            # Check for final report
            if node_state.get("final_report"):
                report = node_state["final_report"]
                critique = node_state.get("critique")
                quality_score = critique.quality_score if critique else None
                
                # Convert Pydantic model to dict for JSON serialization
                # Match ResearchResponse structure expected by frontend
                report_dict = {
                    "query": query,
                    "report": report.content,
                    "sources": report.sources,
                    "confidence": report.confidence,
                    "iteration_count": node_state.get("iteration_count", 0),
                    "quality_score": quality_score,
                    "research_run_id": node_state.get("research_run_id"),
                    "sources_persisted": len(node_state.get("normalized_sources") or []),
                    "evidence_count": len(node_state.get("validated_evidence") or []),
                    "evidence_metrics": node_state.get("evidence_metrics"),
                    "confidence_level": (
                        node_state.get("final_report").confidence_level
                        if node_state.get("final_report")
                        and hasattr(node_state.get("final_report"), "confidence_level")
                        else None
                    ),
                    "report_metrics": node_state.get("report_metrics"),
                    "error": None,
                }

                # Decision Brief: deterministic presentation payload (fail-open)
                try:
                    brief = build_decision_brief_payload(node_state)
                    if brief:
                        report_dict["decision_brief"] = brief
                except Exception as brief_err:
                    logger.warning(
                        "Decision Brief payload assembly failed (omitting): %s",
                        brief_err,
                        exc_info=True,
                    )
                
                # Yield final result
                yield json.dumps({
                    "type": "result",
                    "report": report_dict
                }) + "\n"
                
                final_state = node_state
                # Don't break here - let the graph finish completely for LangSmith
            
            # Check for errors
            if node_state.get("error"):
                error_msg = f"Research failed: {node_state['error']}"
                logger.error(error_msg)
                yield json.dumps({
                    "type": "error",
                    "error": error_msg
                }) + "\n"
                break
                
    except GeneratorExit:
        logger.info("Frontend disconnected. Shielding LangGraph task so it can finish for LangSmith...")
        # CRITICAL: We do NOT cancel the graph_task. We let it finish in the background.
        # This is what turns the checkmark GREEN.
        await graph_task
    except Exception as e:
        # Log full traceback for debugging
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error during streaming: {str(e)}\n{error_trace}")
        # Still await the graph task to ensure it finishes
        await graph_task


@app.post("/research")
async def research(request: ResearchRequest):
    """
    Execute a research query using the research agent with streaming response.
    
    Args:
        request: Research request with query
        
    Returns:
        StreamingResponse with NDJSON events
    """
    return StreamingResponse(
        event_generator(request.query),
        media_type="application/x-ndjson"
    )


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
