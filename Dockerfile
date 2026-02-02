# Multi-stage build for efficiency
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

# Install make and libgomp (required by LightGBM) in the final image
RUN apt-get update && apt-get install -y --no-install-recommends make libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mluser
USER mluser
WORKDIR /home/mluser/app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/mluser/.local

# Copy application code
COPY --chown=mluser:mluser . .

# Add local bin to PATH
ENV PATH=/home/mluser/.local/bin:$PATH

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"

# Default command
CMD ["make", "score"]
