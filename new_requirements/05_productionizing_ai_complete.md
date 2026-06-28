# PRODUCTIONIZING AI SYSTEMS — Complete Guide
## Docker, FastAPI, CI/CD, Monitoring, Logging, Automated Testing for AI
## The biggest gap identified: You hand this off to MLOps. Here, YOU own all of it.

---

## SECTION 1: THE MINDSET SHIFT

### From Enterprise DS to AI Engineer at Consulting Firm

**At Chubb (what you know):**
- You build the model + API
- MLOps team handles Docker, K8s deployment, monitoring
- Large infra team, well-defined roles

**At this consulting firm (what's needed):**
- You own EVERYTHING: model → API → Docker → deployment → monitoring → iteration
- No MLOps team — you ARE the MLOps engineer for SME projects
- The client has no DevOps — you may be deploying to their Windows server, their VPS, or their cloud account

**The key mantra:**
> "I don't consider a feature done until it has: working tests, structured logging, error handling, health endpoint, and cost monitoring. A model that works in my Jupyter notebook is not done."

---

## SECTION 2: FASTAPI — THE PRODUCTION STANDARD FOR AI APIS

### Complete FastAPI Production Template

```python
# main.py — Production-ready FastAPI app for AI service

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional
import httpx
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog  # Structured logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# ═══════════════════════════════════════════════
# LOGGING SETUP — Structured logging with context
# ═══════════════════════════════════════════════
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()  # JSON logs for log aggregation
    ]
)
logger = structlog.get_logger()

# ═══════════════════════════════════════════════
# METRICS — Prometheus counters and histograms
# ═══════════════════════════════════════════════
REQUEST_COUNT = Counter("api_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_duration_seconds", "Request duration", ["endpoint"])
LLM_COST = Counter("llm_cost_usd_total", "Total LLM cost in USD", ["client_id", "model"])
LLM_TOKENS = Counter("llm_tokens_total", "Total LLM tokens", ["client_id", "type"])

# ═══════════════════════════════════════════════
# STARTUP/SHUTDOWN — Load models once at startup
# ═══════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Load models into memory (done once per pod start)
    logger.info("Starting up: loading models and connections")
    
    app.state.embed_model = load_embedding_model()
    app.state.vector_db = connect_to_qdrant()
    app.state.llm_client = create_openai_client()
    app.state.reranker = load_reranker()
    
    logger.info("Startup complete", models_loaded=True)
    
    yield  # App runs here
    
    # SHUTDOWN: Clean up resources
    logger.info("Shutting down: closing connections")
    await app.state.vector_db.close()

# ═══════════════════════════════════════════════
# APP CREATION
# ═══════════════════════════════════════════════
app = FastAPI(
    title="SME AI API",
    description="AI-powered document assistant for SME clients",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://client-dashboard.example.com"],  # Specific origins only
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"]
)

# ═══════════════════════════════════════════════
# MIDDLEWARE — Request ID, Logging, Timing
# ═══════════════════════════════════════════════
@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    
    # Log incoming request
    logger.info("Request received",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_id=request.headers.get("X-Client-ID", "unknown")
    )
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
        REQUEST_COUNT.labels(endpoint=request.url.path, status=response.status_code).inc()
        
        logger.info("Request completed",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2)
        )
        
        response.headers["X-Request-ID"] = request_id
        return response
        
    except Exception as e:
        logger.error("Request failed", request_id=request_id, error=str(e), exc_info=True)
        REQUEST_COUNT.labels(endpoint=request.url.path, status=500).inc()
        raise

# ═══════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════
async def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Verify API key and return client_id"""
    client_id = api_key_store.get(x_api_key)
    if not client_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return client_id

# ═══════════════════════════════════════════════
# MODELS (Pydantic)
# ═══════════════════════════════════════════════
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="User's question")
    doc_type: Optional[str] = Field(None, description="Filter by doc type: invoice, contract, email")
    language: Optional[str] = Field("it", description="Language: 'it' for Italian, 'en' for English")

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float
    cost_usd: float
    latency_ms: float
    request_id: str

class ErrorResponse(BaseModel):
    error: str
    error_code: str
    request_id: str

# ═══════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════
@app.post("/query",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["RAG"]
)
async def query_documents(
    request_data: QueryRequest,
    request: Request,
    client_id: str = Depends(verify_api_key)
):
    """Query client documents using RAG"""
    start = time.time()
    request_id = request.state.request_id
    
    log = logger.bind(request_id=request_id, client_id=client_id)
    log.info("Processing query", question_length=len(request_data.question))
    
    try:
        # Check rate limit
        if rate_limiter.is_exceeded(client_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in 60 seconds.")
        
        # Execute RAG pipeline
        result = await rag_pipeline(
            question=request_data.question,
            client_id=client_id,
            doc_type=request_data.doc_type,
            embed_model=request.app.state.embed_model,
            vector_db=request.app.state.vector_db,
            llm=request.app.state.llm_client,
            reranker=request.app.state.reranker
        )
        
        latency_ms = (time.time() - start) * 1000
        
        # Track costs
        LLM_COST.labels(client_id=client_id, model="gpt-4o").inc(result.cost_usd)
        LLM_TOKENS.labels(client_id=client_id, type="input").inc(result.input_tokens)
        LLM_TOKENS.labels(client_id=client_id, type="output").inc(result.output_tokens)
        
        log.info("Query completed", latency_ms=latency_ms, cost_usd=result.cost_usd)
        
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence,
            cost_usd=result.cost_usd,
            latency_ms=latency_ms,
            request_id=request_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        log.error("Query failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error processing query")

@app.get("/health", tags=["Operations"])
async def health_check(request: Request):
    """Health check endpoint — used by Docker/K8s"""
    checks = {}
    
    # Check vector DB connectivity
    try:
        request.app.state.vector_db.get_collections()
        checks["vector_db"] = "ok"
    except Exception:
        checks["vector_db"] = "error"
    
    # Check embedding model
    try:
        _ = request.app.state.embed_model.encode("test")
        checks["embed_model"] = "ok"
    except Exception:
        checks["embed_model"] = "error"
    
    all_ok = all(v == "ok" for v in checks.items())
    status_code = 200 if all_ok else 503
    
    return JSONResponse(content={"status": "healthy" if all_ok else "degraded", "checks": checks}, status_code=status_code)

@app.get("/ready", tags=["Operations"])
async def readiness_check(request: Request):
    """Readiness check — only ready when models are loaded"""
    if not hasattr(request.app.state, "embed_model"):
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    return {"status": "ready"}

@app.get("/metrics", tags=["Operations"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

---

## SECTION 3: DOCKER — PRODUCTION CONTAINERIZATION

### Dockerfile Best Practices for AI Apps

```dockerfile
# Multi-stage build for smaller image size
# Stage 1: Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ──────────────────────────────────────
# Stage 2: Runtime (smaller image)
FROM python:3.11-slim AS runtime

# Security: Run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy only installed packages from builder (not build tools)
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser app/ ./app/

# Environment variables (non-secret — secrets via env at runtime)
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check built into Docker image
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Startup command — use Gunicorn + Uvicorn for production (not just uvicorn)
CMD ["gunicorn", "app.main:app",
     "--worker-class", "uvicorn.workers.UvicornWorker",
     "--workers", "2",          # 2 workers per container (CPU bound: 1-2)
     "--bind", "0.0.0.0:8080",
     "--timeout", "120",        # 2 min timeout for LLM calls
     "--access-logfile", "-",   # Log to stdout
     "--error-logfile", "-"]
```

### Docker Compose — Full SME Stack

```yaml
# docker-compose.yml — Complete SME AI deployment stack
version: '3.8'

services:
  # ── AI API Service ──────────────────────────────
  ai-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - QDRANT_API_KEY=${QDRANT_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/sme_ai
      - LOG_LEVEL=INFO
      - ENVIRONMENT=production
    depends_on:
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      start_period: 60s  # Give time for model loading
      retries: 3
  
  # ── Vector Database ──────────────────────────────
  qdrant:
    image: qdrant/qdrant:v1.9.0
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 15s
      timeout: 5s
      retries: 5
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G  # Qdrant needs RAM for HNSW index
  
  # ── Cache (query results + embeddings) ───────────
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
    restart: unless-stopped
  
  # ── Relational DB (metadata, user data) ──────────
  postgres:
    image: postgres:15-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql  # Schema on first run
    environment:
      - POSTGRES_DB=sme_ai
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
    restart: unless-stopped
  
  # ── Monitoring ────────────────────────────────────
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped
  
  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    restart: unless-stopped
  
  # ── Nginx Reverse Proxy ───────────────────────────
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs  # SSL certificates
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - ai-api
    restart: unless-stopped

volumes:
  qdrant_data:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

---

## SECTION 4: CI/CD PIPELINE FOR AI SYSTEMS

### GitHub Actions — Complete Pipeline

```yaml
# .github/workflows/deploy.yml
name: AI API CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  DOCKER_IMAGE: ghcr.io/company/sme-ai-api
  PYTHON_VERSION: "3.11"

jobs:
  # ── Job 1: Tests ─────────────────────────────────
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=app --cov-report=xml
      
      - name: Run integration tests (with test Qdrant)
        run: |
          docker run -d -p 6333:6333 qdrant/qdrant
          sleep 5
          pytest tests/integration/ -v
      
      - name: Run RAG evaluation tests
        run: python tests/evaluate_rag.py --threshold 0.80
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY_TEST }}
      
      - name: Check test coverage
        run: |
          coverage report --fail-under=80
  
  # ── Job 2: Security Scan ─────────────────────────
  security:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Bandit security scan
        run: |
          pip install bandit
          bandit -r app/ -f json -o bandit-report.json
      
      - name: Check for secrets in code
        run: |
          pip install detect-secrets
          detect-secrets scan --baseline .secrets.baseline
  
  # ── Job 3: Build Docker Image ─────────────────────
  build:
    runs-on: ubuntu-latest
    needs: [test, security]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and Push Docker Image
        uses: docker/build-push-action@v5
        with:
          push: true
          tags: |
            ${{ env.DOCKER_IMAGE }}:${{ github.sha }}
            ${{ env.DOCKER_IMAGE }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
  
  # ── Job 4: Deploy to Staging ─────────────────────
  deploy-staging:
    runs-on: ubuntu-latest
    needs: build
    environment: staging
    steps:
      - name: Deploy to staging server
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.STAGING_HOST }}
          username: ${{ secrets.STAGING_USER }}
          key: ${{ secrets.STAGING_SSH_KEY }}
          script: |
            cd /opt/sme-ai
            docker pull ${{ env.DOCKER_IMAGE }}:${{ github.sha }}
            docker compose up -d ai-api
            sleep 30
            curl -f http://localhost:8080/health || exit 1
  
  # ── Job 5: RAG Regression Tests on Staging ────────
  regression-test:
    runs-on: ubuntu-latest
    needs: deploy-staging
    steps:
      - uses: actions/checkout@v4
      - name: Run RAG regression tests against staging
        run: |
          python tests/regression/test_rag_regression.py \
            --endpoint https://staging.sme-ai.example.com \
            --api-key ${{ secrets.STAGING_API_KEY }} \
            --min-faithfulness 0.85 \
            --min-context-recall 0.75
  
  # ── Job 6: Deploy to Production (Manual Approval) ─
  deploy-production:
    runs-on: ubuntu-latest
    needs: regression-test
    environment: production  # Requires manual approval in GitHub
    steps:
      - name: Deploy to production
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.PROD_HOST }}
          script: |
            cd /opt/sme-ai
            # Blue-green: pull new image, swap, verify, clean old
            docker pull ${{ env.DOCKER_IMAGE }}:${{ github.sha }}
            docker compose up -d --no-deps ai-api
            sleep 30
            curl -f https://api.sme-ai.example.com/health || \
              (docker compose rollback && exit 1)
```

---

## SECTION 5: MONITORING AI SYSTEMS IN PRODUCTION

### What to Monitor (AI-Specific + Standard)

**Standard infrastructure metrics:**
- API latency (p50, p95, p99) — alert if p99 > 10s for LLM calls
- Error rate (5xx/total) — alert if > 1%
- Pod/container health
- CPU, Memory, Disk usage

**AI-specific metrics:**
```python
# Track these in every production AI deployment

# 1. LLM Cost per client per day
# 2. Token usage (input vs output) — ratio shift signals prompting issues
# 3. Cache hit rate — low hit rate = cost inefficiency
# 4. Retrieval scores — distribution of top-1 similarity scores
# 5. "No relevant documents" rate — high rate = indexing or query issue
# 6. Answer length distribution — sudden change signals prompt drift
# 7. Latency breakdown: embedding + retrieval + reranking + LLM separately

import structlog
log = structlog.get_logger()

def log_rag_metrics(
    request_id: str,
    client_id: str,
    question: str,
    retrieval_scores: list[float],
    num_chunks_returned: int,
    faithfulness_proxy: float,  # Computed heuristic
    latency_breakdown: dict,
    cost_usd: float,
    cached: bool
):
    log.info("rag_query_complete",
        request_id=request_id,
        client_id=client_id,
        question_length=len(question),
        top_retrieval_score=max(retrieval_scores) if retrieval_scores else 0,
        avg_retrieval_score=sum(retrieval_scores)/len(retrieval_scores) if retrieval_scores else 0,
        chunks_returned=num_chunks_returned,
        no_relevant_docs=num_chunks_returned == 0,
        embed_latency_ms=latency_breakdown.get("embedding", 0),
        retrieval_latency_ms=latency_breakdown.get("retrieval", 0),
        rerank_latency_ms=latency_breakdown.get("reranking", 0),
        llm_latency_ms=latency_breakdown.get("llm", 0),
        total_latency_ms=sum(latency_breakdown.values()),
        cost_usd=cost_usd,
        cache_hit=cached
    )
```

### Grafana Dashboard (What to Build)

```
Dashboard 1: Client Usage Overview
├── Total queries per day per client
├── Average latency trend
├── Daily cost per client (bar chart)
├── Cache hit rate
└── Error rate

Dashboard 2: AI Quality Signals
├── "No documents found" rate (retrieval failure indicator)
├── Average top-1 retrieval score trend (score drop = embedding or doc quality issue)
├── Answer length distribution (sudden change = prompt issue)
├── Weekly Ragas evaluation scores
└── LLM model latency by model version

Dashboard 3: Cost Breakdown
├── Cost by client (sorted)
├── Cost by endpoint
├── Token usage: input vs output ratio
├── Cost trend vs query volume (cost/query should be stable)
└── Monthly projection vs budget
```

---

## SECTION 6: TESTING AI SYSTEMS — THE FULL TESTING PYRAMID

### Level 1: Unit Tests (Fast, No LLM Calls)

```python
# tests/unit/test_chunking.py
import pytest
from app.chunking import chunk_text, count_tokens

def test_chunk_size_within_limit():
    """Chunks must not exceed max_tokens"""
    long_text = "word " * 1000
    chunks = chunk_text(long_text, max_tokens=500)
    for chunk in chunks:
        assert count_tokens(chunk) <= 500, f"Chunk exceeds limit: {count_tokens(chunk)} tokens"

def test_chunk_overlap_preserves_context():
    """Chunks should have overlap — first words of chunk N+1 should appear in chunk N"""
    text = " ".join([f"sentence_{i}" for i in range(100)])
    chunks = chunk_text(text, max_tokens=100, overlap=20)
    if len(chunks) > 1:
        # Some content from chunk 0 should appear in chunk 1
        chunk0_words = set(chunks[0].split())
        chunk1_words = set(chunks[1].split())
        overlap_words = chunk0_words & chunk1_words
        assert len(overlap_words) > 0, "No overlap found between consecutive chunks"

def test_psi_calculation():
    """PSI should be near 0 for identical distributions"""
    expected = [0.1, 0.2, 0.3, 0.4]
    actual = [0.1, 0.2, 0.3, 0.4]
    psi = calculate_psi(expected, actual)
    assert psi < 0.01
```

### Level 2: Integration Tests (Real Components, Mocked LLM)

```python
# tests/integration/test_rag_pipeline.py
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def qdrant_test_client():
    """Start a Qdrant instance for testing"""
    # Uses qdrant running in Docker (from CI pipeline)
    client = QdrantClient(url="http://localhost:6333")
    yield client
    # Cleanup
    client.delete_collection("test_collection")

def test_end_to_end_rag_pipeline(qdrant_test_client):
    """Test full RAG pipeline without calling real LLM"""
    
    # Seed test data
    test_chunks = [
        {"text": "Payment terms are 30 days net.", "client_id": "test_client"},
        {"text": "The contract expires on December 31, 2025.", "client_id": "test_client"},
    ]
    index_chunks(test_chunks, "test_collection")
    
    # Mock LLM to avoid cost
    with patch("app.rag.llm_client") as mock_llm:
        mock_llm.chat.return_value = MagicMock(
            content="Payment terms are 30 days net.",
            usage=MagicMock(prompt_tokens=100, completion_tokens=20)
        )
        
        result = rag_query("What are the payment terms?", "test_client")
    
    assert result.answer is not None
    assert len(result.sources) > 0
    assert result.confidence > 0

def test_tenant_isolation(qdrant_test_client):
    """Critical: Client A must not see Client B's documents"""
    
    # Index documents for two different clients
    index_chunk("Secret client A document", client_id="client_a")
    index_chunk("Secret client B document", client_id="client_b")
    
    # Search as client_a
    results = search_documents("secret document", client_id="client_a")
    texts = [r.payload["text"] for r in results]
    
    assert all("client_a" in r.payload["client_id"] for r in results), \
        "CRITICAL: Client A retrieved Client B's document!"
    assert not any("client_b" in text.lower() for text in texts)
```

### Level 3: RAG Quality Tests (Golden Dataset)

```python
# tests/evaluate_rag.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

GOLDEN_TEST_CASES = [
    {
        "question": "What are the payment terms in the contract?",
        "ground_truth": "Payment terms are 30 days from invoice date.",
        # These will be filled by running the actual RAG
        "answer": None,
        "contexts": None
    },
    # ... 50 more test cases per client
]

def evaluate_rag_quality(min_faithfulness: float = 0.80) -> bool:
    """Run Ragas evaluation on golden test cases"""
    
    # Run each test case through live RAG system
    for case in GOLDEN_TEST_CASES:
        result = rag_query(case["question"], client_id="test_client")
        case["answer"] = result.answer
        case["contexts"] = result.source_texts
    
    # Evaluate with Ragas
    dataset = Dataset.from_list(GOLDEN_TEST_CASES)
    metrics = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
    
    print(f"Faithfulness: {metrics['faithfulness']:.3f} (min: {min_faithfulness})")
    print(f"Answer Relevancy: {metrics['answer_relevancy']:.3f}")
    print(f"Context Recall: {metrics['context_recall']:.3f}")
    
    if metrics["faithfulness"] < min_faithfulness:
        print(f"FAIL: Faithfulness {metrics['faithfulness']:.3f} below threshold {min_faithfulness}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args()
    
    success = evaluate_rag_quality(min_faithfulness=args.threshold)
    sys.exit(0 if success else 1)
```

---

## SECTION 7: PRODUCTION FAILURE SCENARIOS AND RESPONSES

### Failure 1: LLM API Is Down
**Detection:** 5xx responses from OpenAI/Anthropic
**Response:**
```python
async def call_llm_with_fallback(prompt: str) -> str:
    # Primary: GPT-4o
    try:
        return await openai_client.complete(prompt, model="gpt-4o")
    except (APIError, RateLimitError) as e:
        logger.warning("Primary LLM failed, trying fallback", error=str(e))
    
    # Fallback: Claude Sonnet
    try:
        return await anthropic_client.complete(prompt, model="claude-3-5-sonnet")
    except Exception as e:
        logger.error("Fallback LLM also failed", error=str(e))
    
    # Last resort: return cached/degraded response
    return "I'm temporarily unable to process your request. Please try again in a few minutes."
```

### Failure 2: Vector DB Unreachable
```python
# Circuit breaker pattern
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
def search_vectors(query_embedding, client_id):
    return qdrant_client.search(...)

# When circuit is OPEN: skip RAG, answer from LLM only (with warning)
try:
    chunks = search_vectors(query_embedding, client_id)
except CircuitBreakerError:
    logger.error("Vector DB circuit breaker open")
    # Return degraded response without RAG context
    answer = llm.answer(f"[Note: document search unavailable] {question}")
    return RAGResponse(answer=answer, sources=[], confidence=0.3)
```

### Failure 3: Prompt Injection Attack
```python
def sanitize_user_input(question: str) -> str:
    """Defend against prompt injection attempts"""
    
    # Remove instructions that could hijack the prompt
    injection_patterns = [
        r"ignore (previous|all|above) instructions",
        r"(you are|act as|pretend to be) (?!an AI|assistant)",
        r"print (your|the) (system|instructions|prompt)",
        r"jailbreak",
        r"DAN"
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            logger.warning("Potential prompt injection detected", question=question[:100])
            raise HTTPException(status_code=400, detail="Invalid query format")
    
    # Length limit
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")
    
    return question.strip()
```

---

## SECTION 8: DEPLOYMENT PATTERNS FOR SME CLIENTS

### Pattern 1: VPS Deployment (Most Common for Italian SMEs)

```bash
# Setup script for Ubuntu VPS (Hetzner, OVH, etc.)
#!/bin/bash
set -e

# Install Docker
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker

# Install docker-compose
apt-get install -y docker-compose-plugin

# Clone application
git clone https://github.com/company/sme-ai /opt/sme-ai
cd /opt/sme-ai

# Create .env file from secrets
cat > .env << EOF
OPENAI_API_KEY=${OPENAI_API_KEY}
QDRANT_API_KEY=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -hex 16)
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
EOF

# Setup SSL (Let's Encrypt)
apt-get install -y certbot
certbot certonly --standalone -d api.client-domain.it

# Start services
docker compose up -d

# Setup auto-renewal
echo "0 2 * * * certbot renew --quiet" | crontab -
```

### Pattern 2: On-Premise Windows Server (Italian Manufacturing)

```powershell
# Windows deployment — many Italian manufacturing SMEs use Windows Server
# Option 1: Docker Desktop for Windows
# Option 2: WSL2 + Docker
# Option 3: Native Windows app with PyInstaller

# For truly air-gapped on-prem:
# Package entire app as Windows service using pyinstaller + NSSM
```

### Pattern 3: Cloud — AWS/Azure with GDPR-EU Regions

```yaml
# Only use EU regions for Italian clients!
# AWS: eu-south-1 (Milan), eu-central-1 (Frankfurt)
# Azure: italynorth, westeurope
# GCP: europe-west8 (Milan)
```
