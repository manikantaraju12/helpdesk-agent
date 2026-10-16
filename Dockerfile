# ─────────────────────────────────────────────────────────────
# Dockerfile — IT Helpdesk Agent
# Base: Python 3.11-slim for a lean, production-ready image
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="IT Helpdesk Agent"
LABEL description="Context-Aware IT Helpdesk Agent with Token Budgets and Pinned Memory"

# Set working directory
WORKDIR /app

# Install system dependencies (curl needed for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download tiktoken encoding to avoid runtime download in air-gapped envs
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Copy source code
COPY src/ ./src/

# Create logs directory with correct permissions
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Expose the API port
EXPOSE 8000

# Health check — polls the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application with uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
