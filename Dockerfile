# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Stateless GBDT Ensemble Matrix Calculator — production container.
# Single-stage, slim, non-root. Zero storage layer, zero external dependencies
# at runtime: the image only needs CPU + RAM to serve inference.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# Fail fast, no .pyc clutter, unbuffered logs for container-friendly stdout.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first to maximise Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application package.
COPY app ./app

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Container-native health check hitting the stateless /health probe.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# Launch the ASGI server.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
