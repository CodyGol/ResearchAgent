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
    
    Args:
        query: Research query string
        
    Yields:
        NDJSON formatted events with type and content
    """
    sent_done = False
    try:
        logger.info(f"Received query: {query}")
        
        # Instantiate the graph
        graph = create_graph()
        app_instance = graph.compile()
        
        # Initialize state
        initial_state: AgentState = {
            "user_query": query,
            "research_plan": None,
            "research_results": None,
            "critique": None,
            "final_report": None,
            "current_node": "planner",
            "iteration_count": 0,
            "error": None,
        }
        
        # Configure LangSmith tracing
        run_config = create_run_config()
        
        # Stream graph execution - wrap in try/except to handle disconnects gracefully
        # Get the stream object first
        stream = app_instance.astream(initial_state, config=run_config)
        
        # Track if client disconnected to prevent yielding after GeneratorExit
        client_disconnected = False
        
        try:
            final_state = None
            # Wrap async for loop to catch GeneratorExit and convert to StopAsyncIteration
            try:
                async for event in stream:
                    # Get the node name from the event
                    node_name = list(event.keys())[0] if event else None
                    node_state = event[node_name] if node_name else {}
                    
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
                            "report": report.content,  # Frontend expects result.report as string
                            "sources": report.sources,
                            "confidence": report.confidence,
                            "iteration_count": node_state.get("iteration_count", 0),
                            "quality_score": quality_score,
                            "error": None,
                        }
                        
                        # Yield final result
                        yield json.dumps({
                            "type": "result",
                            "report": report_dict
                        }) + "\n"
                        
                        final_state = node_state
                        break
                    
                    # Check for errors
                    if node_state.get("error"):
                        error_msg = f"Research failed: {node_state['error']}"
                        logger.error(error_msg)
                        yield json.dumps({
                            "type": "error",
                            "error": error_msg
                        }) + "\n"
                        return
                        
            except GeneratorExit:
                # Convert GeneratorExit to StopAsyncIteration to prevent propagation to LangGraph
                # This allows LangGraph's generator to handle it gracefully and finalize the trace
                logger.info("Client disconnected during stream. Converting GeneratorExit to StopAsyncIteration for graceful LangGraph finalization.")
                client_disconnected = True
                raise StopAsyncIteration  # This won't propagate to LangGraph's generator
                
        except StopAsyncIteration:
            # Normal completion or converted GeneratorExit - LangGraph handles this gracefully
            pass
        except Exception as e:
            # Handle other exceptions during streaming
            logger.error(f"Streaming error: {type(e).__name__}: {str(e)}")
            # Re-raise to be caught by outer handler if needed
            raise
        
        # Verify we got a report (only if client didn't disconnect and we didn't exit early)
        if not client_disconnected:
            if not final_state or not final_state.get("final_report"):
                error_msg = "No report generated"
                logger.error(error_msg)
                if not sent_done:
                    yield json.dumps({
                        "type": "error",
                        "error": error_msg
                    }) + "\n"
            else:
                # Yield explicit completion signal
                if not sent_done:
                    yield json.dumps({"type": "done"}) + "\n"
                    sent_done = True
            
    except Exception as e:
        # Log full traceback for debugging (outer exception handler)
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error during research: {str(e)}\n{error_trace}")
        
        # Only yield error if we haven't already sent done and it's not a GeneratorExit
        if not sent_done and not isinstance(e, GeneratorExit):
            try:
                yield json.dumps({
                    "type": "error",
                    "error": f"Internal server error: {str(e)}"
                }) + "\n"
            except GeneratorExit:
                # Can't yield on GeneratorExit, just return
                return
    finally:
        # Ensure generator closes cleanly to finalize LangSmith trace
        logger.info("LangGraph execution finalized. Check LangSmith for Green Checkmark.")


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
