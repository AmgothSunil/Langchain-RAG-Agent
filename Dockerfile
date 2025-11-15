# ----------> Stage 1: Builder <----------
FROM python:3.11-slim AS builder

WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# ----------> Dependency Layer (cached if requirements.txt unchanged)
COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.4.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# ----------> Code Layer (rebuilds if app/config changes)
COPY app/ ./app/
COPY config/ ./config/

# ----------> Stage 2: Runtime <----------
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY app/ ./app/
COPY config/ ./config/

RUN useradd -m appuser && mkdir -p /app/logs && \
    chown -R appuser /app && chmod -R 755 /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "app.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
