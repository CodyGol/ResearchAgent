# Multi-stage Dockerfile for The Oracle research agent
# Using Python 3.12 slim for production deployment

# ============================================================================
# Builder Stage: Install dependencies with uv
# ============================================================================
FROM python:3.12-slim-bookworm AS builder

# Install system dependencies for uv and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
# uv sync should create .venv and install all dependencies
RUN uv venv && \
    uv sync --frozen --no-dev

# Verify uvicorn is installed (check for uvicorn script in bin/)
RUN if [ -f .venv/bin/uvicorn ]; then \
        echo "✓ uvicorn found at .venv/bin/uvicorn"; \
    else \
        echo "ERROR: uvicorn not found!" && \
        echo "Checking .venv/bin/:" && \
        ls -la .venv/bin/ | head -20 && \
        echo "Installed packages:" && \
        uv pip list && \
        exit 1; \
    fi

# ============================================================================
# Final Stage: Production image
# ============================================================================
FROM python:3.12-slim-bookworm

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY config.py graph.py state.py server.py run_research.py ./
COPY nodes/ ./nodes/
COPY tools/ ./tools/
COPY utils/ ./utils/
COPY db/ ./db/

# Set Python path to use virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server using uvicorn directly from venv bin
CMD ["/app/.venv/bin/uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
