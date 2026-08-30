# Multi-stage build for efficiency
FROM python:3.11-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies.
#
# requirements.txt stays the single source of truth; the image derives a runtime
# subset from it so the two cannot drift:
#
#   * xgboost -> xgboost-cpu. The Linux xgboost wheel hard-depends on
#     nvidia-nccl-cu12, which is 457MB of CUDA this image never executes. The
#     macOS wheel has no such dependency, which is why requirements.txt does not
#     list it. Same package, same version, same `import xgboost`.
#   * Notebook, debugger and test-only packages are dropped: nothing in
#     src/telco_churn imports them, and notebooks/ is excluded by .dockerignore.
#
# Anything genuinely needed at runtime must NOT be listed here — Pygments for
# instance is a rich dependency and stays.
COPY requirements.txt .
RUN set -eux; \
    sed 's/^xgboost==/xgboost-cpu==/I' requirements.txt \
      | grep -vEi '^(ipykernel|ipython|ipython[-_]pygments[-_]lexers|jupyter[-_]client|jupyter[-_]core|matplotlib[-_]inline|debugpy|nest[-_]asyncio|pyzmq|tornado|traitlets|comm|jedi|parso|pexpect|ptyprocess|prompt[-_]toolkit|pure[-_]eval|stack[-_]data|asttokens|executing|pytest|iniconfig|pluggy|seaborn)==' \
      > requirements-runtime.txt; \
    pip install --no-cache-dir --user -r requirements-runtime.txt

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
