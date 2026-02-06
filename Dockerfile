# Production Dockerfile for Google Cloud Run
# Optimized for minimal size and fast startup

FROM python:3.12-slim

# Set environment variables for immediate log output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
# Remove local file reference if present and install from requirements
RUN sed -i '/^-e file:/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py graph.py state.py api.py ./
COPY nodes/ ./nodes/
COPY tools/ ./tools/
COPY utils/ ./utils/
COPY db/ ./db/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port (Cloud Run will set PORT env var)
EXPOSE 8080

# Health check (Cloud Run handles this via /health endpoint)
# Note: HEALTHCHECK is optional for Cloud Run as it uses the /health endpoint directly

# Run the API server
CMD ["python", "api.py"]
