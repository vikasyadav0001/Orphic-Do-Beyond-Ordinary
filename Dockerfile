# ==============================================================================
# STAGE 1: BUILDER
# ==============================================================================
 
FROM python:3.12-slim AS builder
 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
 
WORKDIR /build
 
# Install compilation tools needed for C-extensions (e.g. psycopg, bcrypt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
 
# Copy only requirements to leverage Docker layer cache
COPY requirements.txt .
 
# Install dependencies into isolated prefix (not user-local, avoids home dir confusion)
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
 
 
# ==============================================================================
# STAGE 2: RUNNER
# ==============================================================================
 
FROM python:3.12-slim AS runner
 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PATH=/install/bin:$PATH \
    PYTHONPATH=/install/lib/python3.12/site-packages
 
WORKDIR /app
 
# Install only runtime lib (not build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*
 
# Non-root user with home directory /home/orphic
RUN groupadd -g 10001 orphic && \
    useradd -u 10001 -g orphic -m -s /sbin/nologin -d /home/orphic orphic && \
    chown -R orphic:orphic /home/orphic
 
# Copy compiled packages from builder stage
COPY --from=builder /install /install

ENV HF_HOME=/home/orphic/.cache/huggingface

RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=orphic:orphic . .

RUN chown -R orphic:orphic /home/orphic /app && chmod +x entrypoint.sh

USER orphic
 
EXPOSE 8000
 
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
 
CMD ["./entrypoint.sh"]