# ── Stage 1: Builder ──────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Model artifacts ─────────────────────────────────────────
# Generates the synthetic data and trains and evaluates the models (make data,
# make train, make evaluate), so the image builds from a fresh checkout
FROM python:3.11-slim AS artifacts

WORKDIR /app

# XGBoost and LightGBM load the OpenMP runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY src/ src/
RUN python -m src.data.generate_data && \
    python -m src.features.feature_engineering && \
    python -m src.models.train && \
    python -m src.models.evaluate

# ── Stage 3: Runtime ─────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# XGBoost and LightGBM load the OpenMP runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source and the trained artifacts
COPY src/ src/
COPY --from=artifacts /app/models/ models/
COPY --from=artifacts /app/results/ results/
COPY --from=artifacts /app/data/processed/ data/processed/

# Non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --no-create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
