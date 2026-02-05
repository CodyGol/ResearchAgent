"""FastAPI server for The Oracle research agent API."""

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from graph import create_graph
from state import AgentState, FinalReport

app = FastAPI(
    title="The Oracle",
    description="A recursive deep-research agent system",
    version="0.1.0",
)


class ResearchRequest(BaseModel):
    """Request model for research queries."""

    query: str = Field(..., description="Research question to investigate", min_length=1)
    max_iterations: int = Field(
        default=3, ge=1, le=10, description="Maximum research-critic cycles"
    )


class ResearchResponse(BaseModel):
    """Response model for research results."""

    query: str
    report: str
    sources: list[str]
    confidence: float
    iteration_count: int
    error: str | None = None


@app.get("/")
async def root() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "The Oracle"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    """
    Execute a research query using The Oracle agent.
    
    Args:
        request: Research request with query and optional max_iterations
        
    Returns:
        ResearchResponse with report, sources, and metadata
        
    Raises:
        HTTPException: If research fails
    """
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

    try:
        # Execute graph
        final_state = await app_instance.ainvoke(initial_state)

        # Check for errors
        if final_state.get("error"):
            raise HTTPException(
                status_code=500,
                detail=f"Research failed: {final_state['error']}",
            )

        report = final_state.get("final_report")
        if not report:
            raise HTTPException(
                status_code=500,
                detail="No report generated",
            )

        return ResearchResponse(
            query=request.query,
            report=report.content,
            sources=report.sources,
            confidence=report.confidence,
            iteration_count=final_state.get("iteration_count", 0),
            error=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
