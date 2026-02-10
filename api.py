"""Production FastAPI service for ResearchAgentv2 on Google Cloud Run."""

import logging
import os
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    """
    Execute a research query using the research agent.
    
    Args:
        request: Research request with query
        
    Returns:
        ResearchResponse with report, sources, and metadata
        
    Raises:
        HTTPException: If research fails
    """
    logger.info(f"Received query: {request.query}")
    
    try:
        # Instantiate the graph
        graph = create_graph()
        app_instance = graph.compile()
        
        # Initialize state
        initial_state: AgentState = {
            "user_query": request.query,
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
        
        # Execute graph
        final_state = await app_instance.ainvoke(initial_state, config=run_config)
        
        # Check for errors in state
        if final_state.get("error"):
            error_msg = f"Research failed: {final_state['error']}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Extract report
        report = final_state.get("final_report")
        if not report:
            error_msg = "No report generated"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Extract quality score from critique if available
        critique = final_state.get("critique")
        quality_score = critique.quality_score if critique else None
        
        # Log completion
        logger.info(
            f"Research complete. Quality score: {quality_score}, "
            f"Iterations: {final_state.get('iteration_count', 0)}"
        )
        
        return ResearchResponse(
            query=request.query,
            report=report.content,
            sources=report.sources,
            confidence=report.confidence,
            iteration_count=final_state.get("iteration_count", 0),
            quality_score=quality_score,
            error=None,
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log full traceback for debugging
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error during research: {str(e)}\n{error_trace}")
        
        # Return 500 with error message
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}",
        ) from e


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
