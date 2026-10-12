# Coding Exploration Prep — Brit Insurance Interview Prep

> [!NOTE]
> **Where this file sits.** The long-form **deep-dive** companion to `06-coding-exploration-playbook.md`. `06` gives you the room-shaped routine; this file gives you the full worked material behind it — the imperfect RAG sample and all **10 improvements** with before/after code, the complete hardened pipeline, the full **Git linear history** answer with the four strategies and trade-off table, Python patterns for AI engineers, CI/CD eval gates, MLOps/LLMOps patterns, production reliability patterns (deadline propagation, retry budgets, circuit breaker, idempotency), and 10 quick-fire coding Q&A. Harvested in full from the earlier `br_ins/` set.

> [!IMPORTANT]
> Based on actual interview questions: **RAG code review + improvement** and **Git linear history with teams**. This document covers both deeply, plus supporting Python/AI engineering patterns expected at Lead level.

---

## Table of Contents

1. [RAG Code Review Mastery](#1-rag-code-review-mastery)
2. [Git Linear History — Complete Answer](#2-git-linear-history---complete-answer)
3. [Python Patterns for AI Engineers](#3-python-patterns-for-ai-engineers)
4. [CI/CD for AI Systems](#4-cicd-for-ai-systems)
5. [MLOps / LLMOps Patterns](#5-mlops--llmops-patterns)
6. [Production Reliability Patterns](#6-production-reliability-patterns)
7. [Quick-Fire Coding Q&A](#7-quick-fire-coding-qa)

---

## 1. RAG Code Review Mastery

> [!NOTE]
> The interview asks you to **explain code you are shown** and **propose improvements**. This tests your ability to read unfamiliar code quickly, articulate its intent, and think critically about production-readiness — not just whether it "works."

---

### Sample RAG Code to Review (the kind they'll show)

This is deliberately imperfect production-style code. Study every line — you should be able to explain *each* choice and flag *each* problem.

```python
# ---- IMPERFECT RAG CODE FOR REVIEW ----
import openai
from azure.search.documents import SearchClient

def get_answer(question):
    embedding = openai.embeddings.create(
        model='text-embedding-ada-002',
        input=question
    ).data[0].embedding

    client = SearchClient(endpoint, index, credential)
    results = client.search(
        search_text=None,
        vector_queries=[{'vector': embedding, 'k': 3, 'fields': 'contentVector'}]
    )

    context = ''
    for r in results:
        context += r['content']

    response = openai.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role': 'system', 'content': 'Answer based on context: ' + context},
            {'role': 'user', 'content': question}
        ]
    )
    return response.choices[0].message.content
```

---

### How to Walk Through Code in 10 Minutes

Use this structured framework whenever you are handed unfamiliar code in an interview. It signals senior engineering instincts — you do not panic; you decompose.

#### Step 1 — High-Level Summary (2 minutes)

Narrate the **"what"** before the **"how"**. Say something like:

> "This is a Retrieval-Augmented Generation pipeline. It takes a user question, embeds it using OpenAI's `text-embedding-ada-002` model, retrieves the top-3 semantically similar document chunks from Azure AI Search using a vector query, concatenates those chunks into a context string, and then passes that context plus the original question to GPT-4 to generate an answer. This is the classic RAG pattern — embed → retrieve → generate."

This immediately tells the interviewer you understand the **architecture pattern** and are not just reading syntax.

#### Step 2 — Walk Through Each Component (5 minutes)

Break into the three RAG stages:

| Stage | Lines | What it does |
|---|---|---|
| **Embed** | `openai.embeddings.create(...)` | Converts the user's question into a float vector using Ada-002 |
| **Retrieve** | `SearchClient(...).search(...)` | Finds the k=3 nearest vectors in Azure AI Search index |
| **Generate** | `openai.chat.completions.create(...)` | Sends retrieved context + question to GPT-4 |

Point out **specific choices** made in the code:
- `search_text=None` means **pure vector search** (no keyword fallback)
- `k=3` is **hardcoded** — the number of retrieved chunks is not configurable
- `context = ''` with naive string concatenation — no metadata, no separator
- `SearchClient` is instantiated **inside** the function — new connection on every call
- **No try/except** anywhere — a single API 429 kills the entire request

#### Step 3 — Present Structured Improvements (3 minutes)

Group improvements by theme, not a random list:

```
Theme 1: Reliability     → error handling, retry, circuit breaker
Theme 2: Quality         → hybrid search, reranking, faithfulness check
Theme 3: Safety          → token counting, context length limits
Theme 4: Security        → Managed Identity, no bare credentials
Theme 5: Observability   → logging, tracing, evaluation
Theme 6: Performance     → connection pooling, async, batching
```

---

### The 10 Improvements to Suggest

> [!TIP]
> Always explain the **WHY** before showing code. The interviewer wants to know you understand the consequence, not just that you know the syntax.

---

#### Improvement 1 — Missing Error Handling

**Why it matters:** OpenAI and Azure Search both return HTTP errors (429 rate limit, 503 service unavailable, network timeouts). Without `try/except`, a single failed call raises an unhandled exception that crashes the caller, degrades the user experience, and may trigger cascading failures upstream.

```python
# ❌ BEFORE — naked API call
embedding = openai.embeddings.create(
    model='text-embedding-ada-002',
    input=question
).data[0].embedding

# ✅ AFTER — structured exception handling
try:
    response = openai.embeddings.create(
        model='text-embedding-ada-002',
        input=question
    )
    embedding = response.data[0].embedding
except openai.RateLimitError as e:
    logger.warning("Rate limit hit for embeddings: %s", e)
    raise  # let caller's retry decorator handle it
except openai.APIConnectionError as e:
    logger.error("Network error reaching OpenAI: %s", e)
    raise
except openai.OpenAIError as e:
    logger.error("Unexpected OpenAI error: %s", e)
    raise
```

---

#### Improvement 2 — Hardcoded `k=3`

**Why it matters:** The optimal number of retrieved chunks depends on: (a) how long each chunk is, (b) what the context window of your LLM is, and (c) the nature of the query. Three is almost certainly wrong for every use case. This should be a parameter with a sensible default, tuned via offline evaluation.

```python
# ❌ BEFORE
results = client.search(
    search_text=None,
    vector_queries=[{'vector': embedding, 'k': 3, 'fields': 'contentVector'}]
)

# ✅ AFTER — configurable, with a default determined by evaluation
def get_answer(
    question: str,
    top_k: int = 5,              # Tuned by offline RAGAS evaluation
    score_threshold: float = 0.75  # Discard low-confidence chunks
) -> str:
    ...
    results = client.search(
        search_text=None,
        vector_queries=[{
            'kind': 'vector',
            'vector': embedding,
            'k': top_k,
            'fields': 'contentVector',
            'exhaustive': True       # Use exact kNN, not HNSW approx
        }],
        select=['content', 'source', 'page_number', 'score']
    )
```

---

#### Improvement 3 — No Retry with Exponential Backoff for 429s

**Why it matters:** Azure OpenAI enforces token-per-minute and requests-per-minute quotas. In production, 429 errors are routine. Without retry logic, the system fails hard on transient quota exhaustion. Exponential backoff with **full jitter** prevents the thundering herd problem where all retriers hit the API simultaneously.

```python
import time
import random
from functools import wraps

def retry_with_backoff(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions=(openai.RateLimitError, openai.APIConnectionError)
):
    """Decorator: exponential backoff with full jitter."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error("Max retries exhausted: %s", e)
                        raise
                    # Full jitter: sleep random amount between 0 and cap
                    cap = min(max_delay, base_delay * (2 ** attempt))
                    sleep_time = random.uniform(0, cap)
                    logger.warning(
                        "Attempt %d/%d failed (%s). Retrying in %.2fs",
                        attempt + 1, max_attempts, e, sleep_time
                    )
                    time.sleep(sleep_time)
        return wrapper
    return decorator

@retry_with_backoff(max_attempts=5)
def embed_text(text: str) -> list[float]:
    return openai.embeddings.create(
        model='text-embedding-ada-002',
        input=text
    ).data[0].embedding
```

---

#### Improvement 4 — Semantic-Only Search (Should be Hybrid BM25 + Vector)

**Why it matters:** Pure vector search excels at semantic similarity but is poor at **exact keyword matching** (policy numbers, named entities, product codes). In insurance, a query like "Claim ID CLM-2024-00391" needs exact match. Hybrid search (BM25 lexical + vector semantic) with Reciprocal Rank Fusion (RRF) consistently outperforms either alone.

```python
# ❌ BEFORE — vector-only
results = client.search(
    search_text=None,
    vector_queries=[{'vector': embedding, 'k': 3, 'fields': 'contentVector'}]
)

# ✅ AFTER — hybrid BM25 + vector with RRF
from azure.search.documents.models import VectorizedQuery

vector_query = VectorizedQuery(
    vector=embedding,
    k_nearest_neighbors=top_k,
    fields='contentVector',
    exhaustive=False   # HNSW for speed in production
)

results = client.search(
    search_text=question,          # BM25 lexical search
    vector_queries=[vector_query], # Semantic vector search
    query_type='semantic',         # Azure Semantic Ranking reranks fusion results
    semantic_configuration_name='my-semantic-config',
    query_answer='extractive',     # Extract direct answer spans
    query_caption='extractive',    # Caption highlights
    select=['id', 'content', 'source', 'page_number'],
    top=top_k
)
```

**Why RRF works:** RRF score = Σ 1/(rank_i + 60). Each document's rank in both the lexical and vector result sets is combined, so documents appearing near the top of *both* lists score highest — regardless of absolute score scales.

---

#### Improvement 5 — No Reranking Step

**Why it matters:** Initial retrieval (whether vector or hybrid) is approximate. A cross-encoder reranker reads the query and each candidate document *together*, computing a true relevance score. This catches cases where a semantically similar chunk is actually off-topic, dramatically improving precision at the cost of ~50ms latency.

```python
# ✅ ADD a reranking step using a cross-encoder
# Option A: Use Azure AI Search's built-in semantic reranker (L2 reranker)
# Option B: Use a local cross-encoder for fine-grained control

from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(
    query: str,
    results: list[dict],
    top_n: int = 3
) -> list[dict]:
    """Use cross-encoder to rerank retrieved chunks."""
    pairs = [(query, r['content']) for r in results]
    scores = reranker.predict(pairs)
    
    ranked = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [doc for doc, score in ranked[:top_n] if score > 0.3]
```

---

#### Improvement 6 — Context Concatenation Loses Document Boundaries

**Why it matters:** When you do `context += r['content']`, you lose the source attribution. The LLM cannot distinguish where one document ends and another begins. This makes **citation generation** impossible and creates **hallucination risk** when the model confuses context from different sources.

```python
# ❌ BEFORE — context loses all structure
context = ''
for r in results:
    context += r['content']

# ✅ AFTER — structured context with metadata and separators
def build_context(results: list[dict]) -> tuple[str, list[dict]]:
    """Build context string with clear boundaries and return source metadata."""
    chunks = []
    sources = []
    
    for i, r in enumerate(results, 1):
        chunk = (
            f"[Document {i}]\n"
            f"Source: {r.get('source', 'unknown')}\n"
            f"Page: {r.get('page_number', 'N/A')}\n"
            f"---\n"
            f"{r['content']}\n"
            f"==="
        )
        chunks.append(chunk)
        sources.append({
            'index': i,
            'source': r.get('source'),
            'page': r.get('page_number'),
            'score': r.get('@search.score')
        })
    
    return '\n\n'.join(chunks), sources

context, sources = build_context(reranked_results)
# Pass sources back to caller for citation rendering
```

---

#### Improvement 7 — No Token Counting / Context Length Management

**Why it matters:** GPT-4 Turbo has a 128K context window, but sending 100K tokens costs ~$1/request and inflates latency. More importantly, `text-embedding-ada-002` has an **8,191 token limit** on input — if a document chunk exceeds this, the embedding call silently truncates or errors. You must count tokens before every API call.

```python
import tiktoken

# Token budget constants
MAX_CONTEXT_TOKENS = 4_096   # Reserve room for system prompt + answer
MAX_EMBEDDING_TOKENS = 8_000  # Ada-002 hard limit

def count_tokens(text: str, model: str = 'gpt-4') -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def fit_context_to_budget(
    chunks: list[str],
    token_budget: int = MAX_CONTEXT_TOKENS,
    model: str = 'gpt-4'
) -> str:
    """Greedily pack chunks into the token budget, highest-ranked first."""
    enc = tiktoken.encoding_for_model(model)
    selected = []
    used = 0
    
    for chunk in chunks:
        chunk_tokens = len(enc.encode(chunk))
        if used + chunk_tokens > token_budget:
            break
        selected.append(chunk)
        used += chunk_tokens
    
    return '\n\n'.join(selected)

# Before embedding, validate input size
def safe_embed(text: str) -> list[float]:
    token_count = count_tokens(text, model='text-embedding-ada-002')
    if token_count > MAX_EMBEDDING_TOKENS:
        # Truncate to fit — or split and embed separately
        enc = tiktoken.encoding_for_model('text-embedding-ada-002')
        tokens = enc.encode(text)[:MAX_EMBEDDING_TOKENS]
        text = enc.decode(tokens)
        logger.warning("Input truncated from to %d tokens for embedding", MAX_EMBEDDING_TOKENS)
    return embed_text(text)
```

---

#### Improvement 8 — Should Use Managed Identity (Not Bare Credentials)

**Why it matters:** The original code uses `credential` as a bare variable — likely an `AzureKeyCredential` with a hard-coded API key. Hard-coded keys in source code are a **critical security vulnerability**: they end up in version control, logs, and crash dumps. Azure Managed Identity eliminates secrets entirely — the compute identity (AKS pod, App Service) authenticates directly to Azure AD.

```python
# ❌ BEFORE — API key (never do this in production)
from azure.core.credentials import AzureKeyCredential
credential = AzureKeyCredential("sk-...hardcoded-key...")
client = SearchClient(endpoint, index, credential)

# ✅ AFTER — Managed Identity (zero secrets)
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

# DefaultAzureCredential tries: Managed Identity → CLI → VS Code → Environment
credential = DefaultAzureCredential()

# In AKS with Workload Identity, use ManagedIdentityCredential directly:
# credential = ManagedIdentityCredential(client_id=os.environ['AZURE_CLIENT_ID'])

# SearchClient is also created ONCE (at module level or via DI), not per-request
search_client = SearchClient(
    endpoint=os.environ['AZURE_SEARCH_ENDPOINT'],  # from Key Vault or App Config
    index_name=os.environ['AZURE_SEARCH_INDEX'],
    credential=credential
)
```

---

#### Improvement 9 — No Logging or Distributed Tracing

**Why it matters:** In production, when a user reports a wrong answer or a timeout, you need to reconstruct the entire request: what question was asked, which chunks were retrieved, what was sent to GPT-4, how long each step took. Without structured logging and distributed tracing, debugging is guesswork.

```python
import logging
import time
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

def get_answer(question: str, top_k: int = 5) -> dict:
    """RAG pipeline with full observability."""
    
    with tracer.start_as_current_span("rag.get_answer") as span:
        span.set_attribute("rag.question_length", len(question))
        span.set_attribute("rag.top_k", top_k)
        
        # --- EMBED ---
        t0 = time.monotonic()
        with tracer.start_as_current_span("rag.embed"):
            embedding = safe_embed(question)
        embed_ms = (time.monotonic() - t0) * 1000
        
        logger.info(
            "Embedding complete",
            extra={'embed_ms': embed_ms, 'question_preview': question[:80]}
        )
        span.set_attribute("rag.embed_ms", embed_ms)
        
        # --- RETRIEVE ---
        t1 = time.monotonic()
        with tracer.start_as_current_span("rag.retrieve"):
            raw_results = hybrid_search(embedding, question, top_k=top_k * 2)
        retrieve_ms = (time.monotonic() - t1) * 1000
        
        span.set_attribute("rag.retrieved_count", len(raw_results))
        span.set_attribute("rag.retrieve_ms", retrieve_ms)
        
        # --- RERANK ---
        reranked = rerank_results(question, raw_results, top_n=top_k)
        span.set_attribute("rag.reranked_count", len(reranked))
        
        # --- GENERATE ---
        context, sources = build_context(reranked)
        t2 = time.monotonic()
        
        with tracer.start_as_current_span("rag.generate"):
            answer = call_llm(question, context)
        generate_ms = (time.monotonic() - t2) * 1000
        
        span.set_attribute("rag.generate_ms", generate_ms)
        span.set_status(Status(StatusCode.OK))
        
        logger.info(
            "RAG pipeline complete",
            extra={
                'total_ms': embed_ms + retrieve_ms + generate_ms,
                'sources_used': [s['source'] for s in sources]
            }
        )
        
        return {'answer': answer, 'sources': sources}
```

---

#### Improvement 10 — No Evaluation / Faithfulness Checking

**Why it matters:** LLMs hallucinate. In insurance, a hallucinated answer about policy terms or claim eligibility is a **compliance and liability risk**. You need automated evaluation of: (a) **faithfulness** — is the answer grounded in the retrieved context? and (b) **answer relevance** — does the answer address the question?

```python
# ✅ Add RAGAS-style evaluation (or a lightweight faithfulness check)
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# In CI/CD evaluation pipeline (not in the hot path — too slow for real-time):
def evaluate_rag_quality(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str]
) -> dict:
    dataset = {
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths
    }
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )
    return result

# Lightweight production-time faithfulness heuristic:
def check_faithfulness_heuristic(answer: str, context: str) -> float:
    """
    Checks what fraction of answer sentences can be entailed from context.
    Cheap alternative to running a full NLI model in the hot path.
    Use an LLM-as-judge call on sampled traffic for deeper evaluation.
    """
    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    grounded = sum(1 for s in sentences if s.lower() in context.lower())
    return grounded / max(len(sentences), 1)
```

---

### Complete Improved RAG Code

This is the fully hardened version incorporating all 10 improvements. Show this at the end of your review to demonstrate you can produce production-quality code, not just critique.

```python
"""
rag_pipeline.py — Production-grade RAG pipeline for Azure OpenAI + Azure AI Search.

Key improvements over naive implementation:
  - Managed Identity authentication (no secrets in code)
  - Hybrid BM25 + vector search with Azure Semantic Ranking
  - Cross-encoder reranking
  - Token budgeting to avoid context overflow
  - Retry with exponential backoff + full jitter for 429s
  - Structured logging + OpenTelemetry distributed tracing
  - Document boundary metadata for citation rendering
  - Faithfulness evaluation hooks (sampled traffic)
"""

from __future__ import annotations

import logging
import os
import random
import time
from functools import wraps
from typing import Any

import tiktoken
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from sentence_transformers import CrossEncoder

import openai

# ─── Logging & Tracing Setup ──────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# ─── Constants (from environment — never hardcode) ────────────────────────────

AZURE_SEARCH_ENDPOINT = os.environ['AZURE_SEARCH_ENDPOINT']
AZURE_SEARCH_INDEX    = os.environ['AZURE_SEARCH_INDEX']
AZURE_OPENAI_ENDPOINT = os.environ['AZURE_OPENAI_ENDPOINT']
EMBED_MODEL           = os.environ.get('EMBED_MODEL', 'text-embedding-ada-002')
CHAT_MODEL            = os.environ.get('CHAT_MODEL', 'gpt-4')
SEMANTIC_CONFIG       = os.environ.get('SEMANTIC_CONFIG', 'default')

MAX_CONTEXT_TOKENS  = int(os.environ.get('MAX_CONTEXT_TOKENS', '4096'))
MAX_EMBED_TOKENS    = 8_000  # Ada-002 hard limit
DEFAULT_TOP_K       = int(os.environ.get('DEFAULT_TOP_K', '5'))

# ─── Authentication (Managed Identity — zero secrets) ─────────────────────────

_credential    = DefaultAzureCredential()
_search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=_credential
)
_openai_client = openai.AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_ad_token_provider=lambda: _credential.get_token(
        'https://cognitiveservices.azure.com/.default'
    ).token,
    api_version='2024-02-01'
)

_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ─── Retry Decorator ──────────────────────────────────────────────────────────

def retry_with_backoff(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError
    )
):
    """Exponential backoff with full jitter — prevents thundering herd."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error("All %d retries exhausted for %s: %s",
                                     max_attempts, func.__name__, e)
                        raise
                    cap = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0, cap)
                    logger.warning(
                        "%s attempt %d/%d failed (%s). Retrying in %.2fs",
                        func.__name__, attempt + 1, max_attempts, e, jitter
                    )
                    time.sleep(jitter)
        return wrapper
    return decorator

# ─── Token Utilities ──────────────────────────────────────────────────────────

def count_tokens(text: str, model: str = 'gpt-4') -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def truncate_to_token_limit(text: str, limit: int, model: str = 'gpt-4') -> str:
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) <= limit:
        return text
    logger.warning("Truncating text from %d to %d tokens", len(tokens), limit)
    return enc.decode(tokens[:limit])

def fit_chunks_to_budget(
    chunks: list[str],
    token_budget: int = MAX_CONTEXT_TOKENS,
    model: str = 'gpt-4'
) -> str:
    """Greedily select highest-ranked chunks that fit within token budget."""
    enc = tiktoken.encoding_for_model(model)
    selected, used = [], 0
    for chunk in chunks:
        chunk_tokens = len(enc.encode(chunk))
        if used + chunk_tokens > token_budget:
            break
        selected.append(chunk)
        used += chunk_tokens
    logger.info("Context budget: %d/%d tokens across %d chunks",
                used, token_budget, len(selected))
    return '\n\n'.join(selected)

# ─── Embed ────────────────────────────────────────────────────────────────────

@retry_with_backoff(max_attempts=5)
def embed_text(text: str) -> list[float]:
    """Embed with token-limit guard and retry."""
    text = truncate_to_token_limit(text, MAX_EMBED_TOKENS, model=EMBED_MODEL)
    response = _openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

# ─── Hybrid Retrieve ──────────────────────────────────────────────────────────

def hybrid_search(
    embedding: list[float],
    query_text: str,
    top_k: int = DEFAULT_TOP_K * 2  # Retrieve more, rerank down
) -> list[dict[str, Any]]:
    """Hybrid BM25 + vector search with Azure Semantic Ranking (RRF fusion)."""
    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=top_k,
        fields='contentVector'
    )
    results = _search_client.search(
        search_text=query_text,                       # BM25 lexical
        vector_queries=[vector_query],                # Semantic vector
        query_type='semantic',
        semantic_configuration_name=SEMANTIC_CONFIG,
        select=['id', 'content', 'source', 'page_number'],
        top=top_k
    )
    return [
        {
            'id':          r['id'],
            'content':     r['content'],
            'source':      r.get('source', 'unknown'),
            'page_number': r.get('page_number'),
            'search_score': r.get('@search.score', 0.0)
        }
        for r in results
    ]

# ─── Rerank ───────────────────────────────────────────────────────────────────

def rerank_results(
    query: str,
    results: list[dict],
    top_n: int = DEFAULT_TOP_K,
    score_threshold: float = 0.3
) -> list[dict]:
    """Cross-encoder reranking — query+doc pair scoring."""
    if not results:
        return []
    pairs = [(query, r['content']) for r in results]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [
        {**doc, 'rerank_score': float(score)}
        for doc, score in ranked[:top_n]
        if score >= score_threshold
    ]

# ─── Build Context ────────────────────────────────────────────────────────────

def build_context(
    results: list[dict],
    token_budget: int = MAX_CONTEXT_TOKENS
) -> tuple[str, list[dict]]:
    """
    Structured context with document boundaries and source metadata.
    Returns the context string and a list of sources for citation.
    """
    chunks, sources = [], []
    for i, r in enumerate(results, 1):
        chunk = (
            f"[Document {i}]\n"
            f"Source: {r['source']}  |  Page: {r.get('page_number', 'N/A')}\n"
            f"Relevance: {r.get('rerank_score', r.get('search_score', '?')):.3f}\n"
            f"---\n"
            f"{r['content']}\n"
            f"==="
        )
        chunks.append(chunk)
        sources.append({
            'index':       i,
            'source':      r['source'],
            'page':        r.get('page_number'),
            'score':       r.get('rerank_score', r.get('search_score'))
        })
    
    context = fit_chunks_to_budget(chunks, token_budget)
    return context, sources

# ─── Generate ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a knowledgeable insurance assistant. Answer the user's question ONLY using
the provided context. If the answer is not present in the context, say:
"I don't have enough information to answer that."

Always cite the document number (e.g., [Document 1]) when drawing on information.

Context:
{context}
"""

@retry_with_backoff(max_attempts=3)
def call_llm(question: str, context: str) -> str:
    """Generate answer from context with retry."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = _openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': question}
        ],
        temperature=0.0,   # Deterministic for factual Q&A
        max_tokens=512,
        seed=42            # Reproducibility (Azure OpenAI supports this)
    )
    return response.choices[0].message.content

# ─── Public Interface ─────────────────────────────────────────────────────────

def get_answer(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    context_token_budget: int = MAX_CONTEXT_TOKENS
) -> dict[str, Any]:
    """
    Production RAG pipeline:
      Embed → Hybrid Search → Rerank → Token-fit → Generate

    Returns:
        {
          'answer':  str,
          'sources': [{'index': 1, 'source': '...', 'page': 3, 'score': 0.92}],
          'latency': {'embed_ms': ..., 'retrieve_ms': ..., 'generate_ms': ...}
        }
    """
    with tracer.start_as_current_span('rag.get_answer') as span:
        span.set_attribute('rag.top_k', top_k)
        span.set_attribute('rag.question_len', len(question))

        timings: dict[str, float] = {}

        # 1 — Embed
        t = time.monotonic()
        with tracer.start_as_current_span('rag.embed'):
            embedding = embed_text(question)
        timings['embed_ms'] = (time.monotonic() - t) * 1000

        # 2 — Hybrid Retrieve (fetch 2× top_k, rerank down)
        t = time.monotonic()
        with tracer.start_as_current_span('rag.retrieve'):
            raw_results = hybrid_search(embedding, question, top_k=top_k * 2)
        timings['retrieve_ms'] = (time.monotonic() - t) * 1000
        span.set_attribute('rag.retrieved_count', len(raw_results))

        # 3 — Rerank
        reranked = rerank_results(question, raw_results, top_n=top_k)
        span.set_attribute('rag.reranked_count', len(reranked))

        if not reranked:
            logger.warning("No chunks survived reranking for question: %s", question[:80])
            return {
                'answer':  "I don't have enough information to answer that.",
                'sources': [],
                'latency': timings
            }

        # 4 — Build token-budgeted context
        context, sources = build_context(reranked, context_token_budget)

        # 5 — Generate
        t = time.monotonic()
        with tracer.start_as_current_span('rag.generate'):
            answer = call_llm(question, context)
        timings['generate_ms'] = (time.monotonic() - t) * 1000

        span.set_attribute('rag.total_ms',
                           sum(timings.values()))
        span.set_status(Status(StatusCode.OK))

        logger.info(
            'RAG pipeline complete | embed=%.0fms retrieve=%.0fms generate=%.0fms | sources=%s',
            timings['embed_ms'], timings['retrieve_ms'], timings['generate_ms'],
            [s['source'] for s in sources]
        )

        return {
            'answer':  answer,
            'sources': sources,
            'latency': timings
        }
```

---

## 2. Git Linear History — Complete Answer

> [!IMPORTANT]
> This is a common senior-level Git question. They want to know you understand **why** linear history matters operationally (bisect, rollbacks, release notes), not just the commands.

---

### The Question

> *"How can you have linear commit history in Git when working with multiple people in a team?"*

### Opening framing (say this first)

> "Linear history means every commit on `main` has exactly one parent — the commit graph looks like a straight line rather than a tangle of merge bubbles. This matters practically for `git bisect` (which relies on linearity), for readable release notes, and for clean rollbacks via `git revert`."

---

### Strategy 1 — Rebase + Squash Merge (Recommended for most teams)

**The developer workflow:**

```bash
# ── Day 1: Start feature branch ──────────────────────────────────────────────
git checkout -b feature/ai-submission-parser
git push -u origin feature/ai-submission-parser

# ── During development: messy WIP commits (these are fine — private branch) ──
git commit -m "wip: rough parser skeleton"
git commit -m "fix: handle null PDF pages"
git commit -m "wip: tests passing locally"

# ── Before raising PR: rebase onto latest main ───────────────────────────────
git fetch origin
git rebase origin/main
# Resolve any conflicts commit-by-commit, then:
# git rebase --continue

# Optional: squash WIP commits locally before PR (interactive rebase)
git rebase -i origin/main
# In the editor, mark all but the first commit as 's' (squash)
# Write a meaningful final commit message

# ── Push (force-with-lease — safe force push) ────────────────────────────────
git push --force-with-lease origin feature/ai-submission-parser

# ── PR is merged via "Squash and Merge" in GitHub/Azure DevOps ───────────────
# Result on main: ONE clean commit "feat: AI submission parser (#42)"
```

**Why `--force-with-lease` instead of `--force`?**

`--force` overwrites the remote branch regardless of what's there. If a teammate pushed to your branch (e.g., added a commit review fix), `--force` would silently delete their work. `--force-with-lease` checks that the remote tip matches your last known state — if someone else pushed, the command fails safely, prompting you to `git fetch` first.

---

### Strategy 2 — Trunk-Based Development

All developers commit (or short-lived branches < 2 days) directly to `main`. Work-in-progress features are hidden behind **feature flags**, not branches.

```python
# Feature flag pattern (Azure App Configuration / LaunchDarkly)
if feature_flags.is_enabled('ai-submission-parser', user=current_user):
    result = ai_parser.parse(document)
else:
    result = legacy_parser.parse(document)
```

```bash
# Developer flow (trunk-based)
git fetch origin
git rebase origin/main   # Always rebase, never merge
# make small, atomic commit
git commit -m "feat: add PDF extractor utility behind flag"
git push origin main     # Direct push (if allowed) OR very short-lived PR
```

**Why it works for linear history:** There are no long-lived branches to diverge and create merge bubbles. Every commit goes directly onto `main` in order.

---

### Strategy 3 — GitHub/Azure DevOps Branch Protection

The cleanest enforcement mechanism is **repository settings** — not relying on developer discipline.

```
Branch Protection Rules for 'main':
  ✅ Require pull request before merging
  ✅ Require 1+ approvals
  ✅ Require status checks (CI) to pass
  ✅ Require branches to be up to date before merging
  ✅ Allow only Squash Merge          ← THIS enforces linearity
  ❌ Disable Merge Commits
  ❌ Disable Rebase Merge (confusingly named — enables fast-forward which is fine)
```

---

### The Trade-offs Table

| Strategy | Linear? | Preserves granular WIP history? | Risk / Complexity |
|---|---|---|---|
| **Merge commit** (default) | ❌ No | ✅ Yes | Messy log, hard `git bisect`, noisy `git log` |
| **Rebase + Squash Merge** | ✅ Yes | ❌ No (squashed) | Requires team discipline; `--force-with-lease` needed |
| **Rebase Merge** (fast-forward) | ✅ Yes | ✅ Yes (each commit preserved) | Conflicts must be resolved per-commit during rebase |
| **Trunk-based dev** | ✅ Yes | ✅ Yes (small atomic commits) | Needs feature flags infrastructure; CI must be very fast |

---

### Golden Rules

> [!IMPORTANT]
> These three rules prevent the most common Git disasters in team environments.

1. **Never rebase commits already pushed to a shared branch.** Rebase rewrites SHA hashes. If others have based work on those commits, their histories diverge.

2. **Always use `git push --force-with-lease`**, never `git push --force`. Lease checks that you're not overwriting work you haven't seen.

3. **Enforce at the repo level**: use branch protection + squash-merge-only so linear history cannot be broken by accident, even by senior engineers on a late Friday.

---

### Bonus: `git bisect` — Why Linearity Matters

```bash
# With linear history, bisect reliably finds the commit that broke something:
git bisect start
git bisect bad HEAD                  # Current commit is broken
git bisect good v1.2.0              # This tag was good
# Git binary-searches the linear range: ~10 steps for 1000 commits
# Each step: git bisect good/bad
git bisect reset

# With merge-commit history, bisect must navigate a DAG — much harder,
# and octopus merge commits confuse the algorithm.
```

---

## 3. Python Patterns for AI Engineers

---

### Async/Await for Concurrent AI Calls

When embedding a batch of documents, serial calls are the bottleneck. `asyncio.gather` allows all embedding requests to run concurrently (I/O-bound — GIL is not a problem here).

```python
import asyncio
import openai

async def embed_async(client: openai.AsyncAzureOpenAI, text: str) -> list[float]:
    response = await client.embeddings.create(
        model='text-embedding-ada-002',
        input=text
    )
    return response.data[0].embedding

async def embed_batch_parallel(
    texts: list[str],
    max_concurrent: int = 20  # Stay within RPM limits
) -> list[list[float]]:
    """Embed N texts in parallel, respecting concurrency limit."""
    client = openai.AsyncAzureOpenAI(
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        azure_ad_token_provider=get_token,
        api_version='2024-02-01'
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_embed(text: str) -> list[float]:
        async with semaphore:
            return await embed_async(client, text)

    return await asyncio.gather(*[bounded_embed(t) for t in texts])

# Usage from synchronous code:
embeddings = asyncio.run(embed_batch_parallel(my_documents))

# Speedup: 20 concurrent calls vs serial = ~20× throughput (I/O-bound)
```

---

### Retry with Exponential Backoff + Full Jitter

(See [Improvement 3](#improvement-3----no-retry-with-exponential-backoff-for-429s) above for sync version.)

```python
import asyncio

def async_retry_with_backoff(max_attempts=5, base_delay=1.0, max_delay=60.0):
    """Async version of the retry decorator."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except (openai.RateLimitError, openai.APIConnectionError) as e:
                    if attempt == max_attempts - 1:
                        raise
                    cap = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0, cap)
                    await asyncio.sleep(jitter)  # non-blocking sleep
        return wrapper
    return decorator

@async_retry_with_backoff(max_attempts=5)
async def embed_with_retry(text: str) -> list[float]:
    ...
```

---

### Token Counting

```python
import tiktoken
from functools import lru_cache

@lru_cache(maxsize=8)
def get_encoder(model: str):
    """Cache encoders — they are expensive to construct."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding('cl100k_base')  # Default for GPT-4

def count_tokens(text: str, model: str = 'gpt-4') -> int:
    return len(get_encoder(model).encode(text))

def messages_token_count(messages: list[dict], model: str = 'gpt-4') -> int:
    """
    Count tokens for an OpenAI messages list including formatting overhead.
    GPT-4 adds 3 tokens per message + 3 tokens for reply priming.
    """
    enc = get_encoder(model)
    total = 3  # reply priming
    for msg in messages:
        total += 3  # per-message overhead
        total += len(enc.encode(msg.get('content', '')))
        total += len(enc.encode(msg.get('role', '')))
    return total

def fit_messages_to_window(
    system_prompt: str,
    user_question: str,
    context_chunks: list[str],
    max_tokens: int = 4096,
    model: str = 'gpt-4'
) -> list[dict]:
    """
    Build messages list that fits within the token window by
    binary-searching the number of context chunks.
    """
    enc = get_encoder(model)
    base_tokens = (
        count_tokens(system_prompt, model)
        + count_tokens(user_question, model)
        + 50  # safety margin
    )
    budget = max_tokens - base_tokens
    
    selected_chunks = []
    used = 0
    for chunk in context_chunks:
        chunk_tokens = len(enc.encode(chunk))
        if used + chunk_tokens > budget:
            break
        selected_chunks.append(chunk)
        used += chunk_tokens
    
    return [
        {'role': 'system', 'content': system_prompt + '\n\n' + '\n\n'.join(selected_chunks)},
        {'role': 'user',   'content': user_question}
    ]
```

---

### Structured Outputs with Pydantic

```python
from pydantic import BaseModel, Field
from typing import Literal
import openai
import json

# Insurance-specific structured output
class ClaimExtraction(BaseModel):
    claim_id:         str           = Field(description="Policy claim identifier, e.g. CLM-2024-00391")
    claimant_name:    str           = Field(description="Full name of the claimant")
    incident_date:    str           = Field(description="Date of incident in ISO format YYYY-MM-DD")
    incident_type:    Literal[
                          'property_damage',
                          'personal_injury',
                          'theft',
                          'liability',
                          'other'
                      ]              = Field(description="Category of claim")
    estimated_value:  float | None  = Field(description="Estimated claim value in GBP, null if unknown")
    urgency:          Literal['low', 'medium', 'high', 'critical']
    summary:          str           = Field(description="2-sentence summary of the claim")

def extract_claim(raw_text: str) -> ClaimExtraction:
    """Use function-calling / structured output to extract claim fields."""
    response = _openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {'role': 'system', 'content': 'Extract claim information from the text.'},
            {'role': 'user',   'content': raw_text}
        ],
        tools=[{
            'type': 'function',
            'function': {
                'name':        'extract_claim',
                'description': 'Extract structured claim data',
                'parameters':  ClaimExtraction.model_json_schema()
            }
        }],
        tool_choice={'type': 'function', 'function': {'name': 'extract_claim'}}
    )
    args = json.loads(
        response.choices[0].message.tool_calls[0].function.arguments
    )
    return ClaimExtraction(**args)
```

---

### Batch Processing for Embeddings

```python
from itertools import islice

def chunk_iterable(iterable, size: int):
    """Split any iterable into batches of `size`."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch

def embed_documents_batched(
    documents: list[str],
    batch_size: int = 100,      # Ada-002 max is 2048 inputs per request
    model: str = 'text-embedding-ada-002'
) -> list[list[float]]:
    """
    Batch-embed documents efficiently.
    Ada-002 allows up to 2048 inputs per API call — use this to cut
    the number of API round-trips by 100×.
    """
    all_embeddings = []
    for batch in chunk_iterable(documents, batch_size):
        response = _openai_client.embeddings.create(model=model, input=batch)
        # Results are returned in input order
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        logger.info("Embedded batch of %d documents", len(batch))
    return all_embeddings
```

---

## 4. CI/CD for AI Systems

> [!NOTE]
> Unlike classical software CI/CD, AI systems have **non-deterministic outputs**. You cannot assert `assertEqual(llm_answer, expected)`. Instead, you assert that **quality metrics** stay above thresholds and that **regressions** (drops in metric) are caught before production.

---

### Eval Gates in CI/CD

```yaml
# .github/workflows/rag-eval.yml
name: RAG Evaluation Gate

on:
  pull_request:
    paths:
      - 'src/rag/**'
      - 'prompts/**'
      - 'configs/**'

jobs:
  rag-eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run RAGAS evaluation
        run: |
          python scripts/eval_rag.py \
            --dataset data/eval/golden_qa_pairs.jsonl \
            --threshold-faithfulness 0.85 \
            --threshold-answer-relevancy 0.80 \
            --threshold-context-recall 0.75 \
            --output-file eval_results.json
        env:
          AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}

      - name: Check regression vs main branch baseline
        run: python scripts/check_regression.py --results eval_results.json --baseline baselines/main.json

      - name: Upload eval artifacts
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: eval_results.json
```

```python
# scripts/eval_rag.py (simplified)
import json
import sys
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

def run_eval(dataset_path: str, thresholds: dict) -> bool:
    dataset = load_golden_dataset(dataset_path)
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])

    failed = []
    for metric, threshold in thresholds.items():
        score = results[metric]
        print(f"{metric}: {score:.4f} (threshold: {threshold})")
        if score < threshold:
            failed.append(f"{metric}={score:.4f} below {threshold}")
    
    if failed:
        print("EVAL GATE FAILED:", failed)
        sys.exit(1)
    print("All eval gates passed.")
```

---

### Prompt Versioning

```
prompts/
├── system_prompt_v1.txt
├── system_prompt_v2.txt          ← Current production
├── system_prompt_v3.txt          ← Under A/B test
└── prompt_registry.json
```

```json
// prompt_registry.json
{
  "rag_system_prompt": {
    "production": "v2",
    "shadow":     "v3",
    "description": "Main RAG system prompt",
    "changelog": {
      "v2": "Added instruction to cite document numbers",
      "v3": "Chain-of-thought reasoning before answering"
    }
  }
}
```

---

### Canary Deployments for Model Updates

```
Traffic split for gpt-4-turbo → gpt-4o migration:

Week 1:  95% gpt-4-turbo | 5%  gpt-4o   ← Watch error rates and latency
Week 2:  80% gpt-4-turbo | 20% gpt-4o   ← Compare quality metrics
Week 3:  50% gpt-4-turbo | 50% gpt-4o   ← If metrics ≥ baseline, continue
Week 4:   0% gpt-4-turbo | 100% gpt-4o  ← Full cutover

Rollback trigger: If p99 latency spikes >20% or faithfulness drops >5%
```

```python
# Feature-flag based canary routing
import random

def select_model(user_id: str, canary_pct: float = 0.05) -> str:
    """Sticky canary: same user always gets same model."""
    if hash(user_id) % 100 < (canary_pct * 100):
        return 'gpt-4o'
    return 'gpt-4-turbo'
```

---

### Blue/Green Deployment for ML Models

```
Production namespace:    rag-service-blue  (current stable)
Staging namespace:       rag-service-green (new version)

Deployment steps:
1. Deploy green (new index + new prompts + new model)
2. Run smoke tests against green
3. Run full eval suite against green
4. Shift 100% traffic from blue to green (load balancer flip)
5. Keep blue running for 30 min (instant rollback)
6. Decommission blue

Key difference from canary: Blue/Green is all-or-nothing (no % split).
Use canary for model quality experiments; use blue/green for infrastructure upgrades.
```

---

## 5. MLOps / LLMOps Patterns

---

### Experiment Tracking with Azure ML

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job
from azure.identity import DefaultAzureCredential
import mlflow

mlflow.set_tracking_uri(os.environ['AZUREML_MLFLOW_URI'])

with mlflow.start_run(run_name='rag-eval-hybrid-search-v2'):
    mlflow.log_param('embed_model', 'text-embedding-ada-002')
    mlflow.log_param('top_k', 5)
    mlflow.log_param('reranker', 'ms-marco-MiniLM-L-6-v2')
    mlflow.log_param('search_type', 'hybrid')
    
    # Run evaluation
    results = run_eval(dataset_path='data/eval/golden_qa.jsonl')
    
    mlflow.log_metric('faithfulness',      results['faithfulness'])
    mlflow.log_metric('answer_relevancy',  results['answer_relevancy'])
    mlflow.log_metric('context_recall',    results['context_recall'])
    mlflow.log_metric('mean_latency_ms',   results['mean_latency_ms'])
    
    # Log artifacts
    mlflow.log_artifact('eval_results.json')
    mlflow.log_artifact('prompts/system_prompt_v2.txt')
```

---

### A/B Testing for LLM Responses

```python
class LLMExperiment:
    """
    Routes requests to control/treatment LLMs.
    Logs both responses and user feedback for offline analysis.
    """
    def __init__(self, control_model: str, treatment_model: str, treatment_pct: float = 0.1):
        self.models = {
            'control':   control_model,
            'treatment': treatment_model
        }
        self.treatment_pct = treatment_pct

    def get_answer(self, question: str, user_id: str) -> dict:
        variant = 'treatment' if self._in_treatment(user_id) else 'control'
        model   = self.models[variant]
        answer  = call_llm_with_model(question, model)
        
        # Log for offline evaluation
        experiment_logger.log({
            'user_id':   user_id,
            'variant':   variant,
            'model':     model,
            'question':  question,
            'answer':    answer,
            'timestamp': time.time()
        })
        return {'answer': answer, 'variant': variant}

    def _in_treatment(self, user_id: str) -> bool:
        return (hash(user_id) % 100) < (self.treatment_pct * 100)
```

---

### Prompt Drift Detection

```python
# Prompt drift: the same prompt produces different output distributions
# over time (due to model updates, data changes, or context shift).
# Detect by monitoring output embeddings in production.

import numpy as np
from scipy.spatial.distance import cosine

class PromptDriftDetector:
    def __init__(self, baseline_embeddings: list[list[float]], threshold: float = 0.05):
        self.baseline_centroid = np.mean(baseline_embeddings, axis=0)
        self.threshold = threshold

    def check_drift(self, recent_embeddings: list[list[float]]) -> dict:
        recent_centroid = np.mean(recent_embeddings, axis=0)
        drift = cosine(self.baseline_centroid, recent_centroid)
        return {
            'drift_score': drift,
            'drifted':     drift > self.threshold,
            'severity':    'high' if drift > self.threshold * 2 else 'medium' if drift > self.threshold else 'none'
        }
```

---

### Reproducibility: Version Everything

```yaml
# rag_config.yaml — checked into git, versioned with each deployment
version: "2024-11-15"
components:
  embed_model:   "text-embedding-ada-002"
  embed_version: "2"
  chat_model:    "gpt-4-turbo-2024-04-09"   # Pinned model version
  search_index:  "insurance-docs-v7"         # Indexed at specific date
  system_prompt: "prompts/system_v2.txt"
  system_prompt_sha: "d3f8a91..."            # Git SHA of prompt file
  reranker:      "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k:         5
  context_token_budget: 4096
```

> [!TIP]
> Pin model versions (e.g., `gpt-4-turbo-2024-04-09` not just `gpt-4-turbo`). Azure OpenAI rotates the `latest` alias — unpinned models can silently change behaviour and break your evals overnight.

---

## 6. Production Reliability Patterns

---

### Deadline Propagation

> [!IMPORTANT]
> The rule: **upstream timeout must always be shorter than downstream timeout**. If the API Gateway times out a user at 30s but your RAG pipeline has a 60s timeout internally, the user gets an error but the backend continues burning compute — **zombie work**.

```python
import asyncio
from contextvars import ContextVar

# Store the request deadline in a context variable (propagates through async tasks)
request_deadline: ContextVar[float] = ContextVar('request_deadline')

def set_deadline(timeout_seconds: float):
    request_deadline.set(time.monotonic() + timeout_seconds)

def remaining_budget_seconds() -> float:
    deadline = request_deadline.get(None)
    if deadline is None:
        return float('inf')
    return max(0.0, deadline - time.monotonic())

async def get_answer_with_deadline(question: str, total_timeout: float = 25.0) -> dict:
    set_deadline(total_timeout)
    try:
        return await asyncio.wait_for(
            _rag_pipeline(question),
            timeout=remaining_budget_seconds()
        )
    except asyncio.TimeoutError:
        logger.warning("RAG pipeline timed out after %.1fs", total_timeout)
        raise
```

---

### Retry Budgets (Token Bucket)

```python
import threading
import time

class RetryBudget:
    """
    Token bucket: allows at most `rate` retries per second, burst of `capacity`.
    Prevents retry amplification when downstream is unhealthy.
    """
    def __init__(self, rate: float = 10.0, capacity: float = 20.0):
        self.rate     = rate
        self.capacity = capacity
        self._tokens  = capacity
        self._last    = time.monotonic()
        self._lock    = threading.Lock()

    def consume(self) -> bool:
        """Returns True if a retry token is available."""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self.capacity,
                self._tokens + (now - self._last) * self.rate
            )
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

_retry_budget = RetryBudget(rate=5.0)  # max 5 retries/sec globally

def retry_with_budget(func, *args, **kwargs):
    if not _retry_budget.consume():
        raise RuntimeError("Retry budget exhausted — system under stress, failing fast")
    return func(*args, **kwargs)
```

---

### Circuit Breaker

```python
from enum import Enum

class CircuitState(Enum):
    CLOSED   = 'closed'    # Normal operation
    OPEN     = 'open'      # Failing fast — no requests sent
    HALF_OPEN = 'half_open' # Probing — one request allowed

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int   = 5,
        recovery_timeout:  float = 30.0,
        success_threshold: int   = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout  = recovery_timeout
        self.success_threshold = success_threshold
        self._state            = CircuitState.CLOSED
        self._failures         = 0
        self._successes        = 0
        self._opened_at        = 0.0

    def call(self, func, *args, **kwargs):
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at > self.recovery_timeout:
                self._state    = CircuitState.HALF_OPEN
                self._successes = 0
            else:
                raise RuntimeError("Circuit OPEN — downstream unavailable")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self):
        if self._state == CircuitState.HALF_OPEN:
            self._successes += 1
            if self._successes >= self.success_threshold:
                self._state   = CircuitState.CLOSED
                self._failures = 0
        else:
            self._failures = max(0, self._failures - 1)

    def _on_failure(self):
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state     = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.error("Circuit OPENED after %d failures", self._failures)

openai_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

def protected_llm_call(question: str, context: str) -> str:
    return openai_circuit.call(call_llm, question, context)
```

---

### Idempotency Keys for Writes

```python
import hashlib
import redis

_redis = redis.Redis.from_url(os.environ['REDIS_URL'])

def idempotent_write(key: str, write_fn, ttl_seconds: int = 86400):
    """
    Execute write_fn exactly once for a given idempotency key.
    Subsequent calls with the same key return the cached result.
    Used for agent tool calls (e.g., 'submit_claim', 'send_email').
    """
    cache_key = f"idem:{key}"
    cached = _redis.get(cache_key)
    if cached:
        logger.info("Idempotent: returning cached result for key %s", key)
        return json.loads(cached)
    
    result = write_fn()
    _redis.setex(cache_key, ttl_seconds, json.dumps(result))
    return result

# Usage: idempotency key = hash of (user_id, action, payload)
def submit_claim(user_id: str, claim_payload: dict) -> dict:
    idem_key = hashlib.sha256(
        f"{user_id}:{json.dumps(claim_payload, sort_keys=True)}".encode()
    ).hexdigest()
    
    return idempotent_write(
        key=idem_key,
        write_fn=lambda: _claims_api.submit(claim_payload)
    )
```

---

## 7. Quick-Fire Coding Q&A

> [!NOTE]
> For each question, deliver a **3-part answer**: definition → consequence → solution. This structure works for Lead-level interviews and shows systematic thinking.

---

### Q1: What is tail latency and how do you handle it?

**Answer:**

Tail latency refers to the latency at high percentiles (p95, p99, p999) of the distribution. The **mean** hides it — if 99% of requests complete in 200ms but 1% take 10s (due to a slow backend replica, GC pause, or retry storm), users still experience those 10s waits.

**Why it matters in AI systems:** LLM APIs exhibit high tail latency because GPU preemption, token streaming backpressure, and quota exhaustion cause infrequent but severe slowdowns.

**How to handle it:**

```python
# Strategy 1: Hedged requests (speculative execution)
# Send the same request to two backends simultaneously; use the first response.
import asyncio

async def hedged_embed(text: str, hedge_delay_ms: float = 100.0) -> list[float]:
    """
    Send request to primary. If no response in hedge_delay_ms,
    send identical request to secondary. Use whichever responds first.
    Cancel the slower one.
    """
    async def primary():
        return await embed_async(primary_client, text)
    
    async def hedged():
        await asyncio.sleep(hedge_delay_ms / 1000)
        return await embed_async(secondary_client, text)
    
    tasks = [asyncio.create_task(primary()), asyncio.create_task(hedged())]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    return done.pop().result()

# Strategy 2: Timeouts at every level (not just globally)
# Strategy 3: Avoid fan-out amplification — one slow shard kills the response
```

---

### Q2: How do you prevent retry amplification?

**Answer:**

Retry amplification: if a downstream service is at 100% load, retries add *more* load, causing more failures, causing more retries — a positive feedback loop that takes the service down completely.

**Solutions:**

1. **Retry budget** (token bucket) — cap the *rate* of retries system-wide, not per-request (see [Section 6](#6-production-reliability-patterns))
2. **Jitter** — randomize retry delays so retriers don't synchronize
3. **Circuit breaker** — stop sending requests when failure rate is high
4. **`Retry-After` header respect** — honour the server's backoff hint
5. **Idempotency** — so retries are safe, but this doesn't prevent amplification on its own

```python
# Respect Retry-After header from Azure OpenAI
except openai.RateLimitError as e:
    retry_after = float(e.response.headers.get('Retry-After', base_delay))
    time.sleep(retry_after + random.uniform(0, 1))  # Add jitter
```

---

### Q3: What is KV cache in LLMs?

**Answer:**

During autoregressive generation, each new token attends to all previous tokens via the attention mechanism. The **Key** and **Value** matrices for previous tokens are recomputed every step unless cached. The KV cache stores these K/V tensors between steps, reducing per-token computation from O(n²) to O(n) for the prefix.

**Practical implications:**

- **Prompt caching** (Azure OpenAI, Anthropic): If multiple requests share a common prefix (system prompt), the KV cache for that prefix can be reused across requests → **up to 90% latency reduction** for cached tokens, and significant cost savings.
- Cache is bounded by GPU VRAM. Long contexts (128K tokens) consume ~GBs of KV cache per session.
- **Strategies**: Use `system_prompt` for long invariant content (policy docs, instructions) that benefits from caching; keep variable content (user question) in the `user` role at the end.

```python
# Azure OpenAI: cached tokens are cheaper (check your pricing tier)
# Ensure system prompt is identical byte-for-byte across requests to hit cache
SYSTEM_PROMPT = open('prompts/system_v2.txt').read()  # load once, reuse

response = client.chat.completions.create(
    model='gpt-4o',
    messages=[
        {'role': 'system', 'content': SYSTEM_PROMPT},  # Likely cached after first call
        {'role': 'user',   'content': question}         # Always unique
    ]
)
```

---

### Q4: Explain Python GIL and when it matters for AI systems

**Answer:**

The **Global Interpreter Lock (GIL)** is a mutex in CPython that allows only one thread to execute Python bytecode at a time. This means multi-threading in Python does **not** provide true parallelism for CPU-bound tasks.

**When it matters (and doesn't) in AI systems:**

| Workload | GIL impact | Solution |
|---|---|---|
| Calling OpenAI API (I/O-bound) | **None** — GIL released during I/O | `asyncio` or `threading` both work fine |
| NumPy/PyTorch tensor ops | **None** — runs in C extensions that release GIL | Threads fine |
| Pure Python CPU: tokenization, JSON parsing | **Blocks** — only one thread runs | `multiprocessing` for true parallelism |
| Inference with local models (Python loop) | **Blocks** — use ProcessPoolExecutor | Subprocess per GPU |

```python
# I/O-bound: asyncio is best (low overhead, single thread)
# asyncio.gather() for parallel API calls — GIL doesn't matter

# CPU-bound Python preprocessing: use ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor

def preprocess(doc: str) -> str:
    # CPU-intensive: clean, chunk, extract entities
    return clean_and_chunk(doc)

with ProcessPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(preprocess, documents))
```

---

### Q5: How do you implement rate limiting for an LLM API?

**Answer:**

You need rate limiting on **two axes**: requests per minute (RPM) and tokens per minute (TPM).

```python
import time
import threading
from collections import deque

class SlidingWindowRateLimiter:
    """Token-counted sliding window rate limiter."""
    
    def __init__(self, max_tokens_per_minute: int = 90_000):
        self.max_tpm    = max_tokens_per_minute
        self._window    = deque()   # (timestamp, token_count)
        self._lock      = threading.Lock()

    def acquire(self, token_count: int, timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                now = time.monotonic()
                # Remove entries older than 60s
                while self._window and now - self._window[0][0] > 60.0:
                    self._window.popleft()
                
                used = sum(tc for _, tc in self._window)
                if used + token_count <= self.max_tpm:
                    self._window.append((now, token_count))
                    return True
            
            # Not enough budget — wait a bit
            time.sleep(0.1)
        
        return False  # Timed out waiting for budget

_limiter = SlidingWindowRateLimiter(max_tokens_per_minute=90_000)

def rate_limited_llm_call(messages: list[dict]) -> str:
    token_estimate = messages_token_count(messages) + 512  # + expected output
    if not _limiter.acquire(token_estimate, timeout=30.0):
        raise RuntimeError("Rate limit budget exhausted — try again later")
    return _openai_client.chat.completions.create(
        model=CHAT_MODEL, messages=messages
    ).choices[0].message.content
```

---

### Q6: What's the difference between async/await and multiprocessing in Python?

**Answer:**

| Aspect | `async/await` (asyncio) | `multiprocessing` |
|---|---|---|
| **Mechanism** | Single thread, cooperative multitasking | Multiple OS processes |
| **GIL** | Single thread, GIL irrelevant | Each process has its own GIL |
| **Best for** | I/O-bound: API calls, DB queries, network | CPU-bound: data preprocessing, tokenization |
| **Overhead** | Very low (event loop context switches) | High (process fork, IPC serialization) |
| **Memory** | Shared (one process) | Separate memory spaces |
| **Use in AI** | Parallel OpenAI calls, streaming | Embedding large corpora, PDF parsing |

```python
# asyncio: right choice for parallel API calls
async def embed_many(texts):
    return await asyncio.gather(*[embed_async(t) for t in texts])

# multiprocessing: right choice for CPU-heavy preprocessing
from concurrent.futures import ProcessPoolExecutor

def parse_pdf(path: str) -> str:
    # CPU-intensive, GIL-blocking
    return extract_text_from_pdf(path)

with ProcessPoolExecutor(max_workers=4) as pool:
    texts = list(pool.map(parse_pdf, pdf_paths))
```

---

### Q7: How do you handle streaming responses from an LLM?

**Answer:**

Streaming returns tokens as they are generated via Server-Sent Events (SSE). This dramatically improves **perceived latency** (Time to First Token) for chat interfaces.

```python
# FastAPI streaming endpoint
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

async def token_stream_generator(question: str, context: str):
    """Yield SSE-formatted tokens as they arrive from the LLM."""
    stream = await _openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {'role': 'system', 'content': f'Context: {context}'},
            {'role': 'user',   'content': question}
        ],
        stream=True
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            # SSE format: "data: <content>\n\n"
            yield f"data: {json.dumps({'token': delta.content})}\n\n"
    yield "data: [DONE]\n\n"

@app.get('/stream')
async def stream_answer(question: str):
    context = await get_context_async(question)
    return StreamingResponse(
        token_stream_generator(question, context),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'  # Disable nginx buffering
        }
    )
```

---

### Q8: What is SSE and when to use it over WebSockets?

**Answer:**

**SSE (Server-Sent Events):** Unidirectional HTTP/1.1 or HTTP/2 stream from server to client. The client opens a single long-lived `GET` request; the server pushes newline-delimited `data:` messages.

**WebSockets:** Full-duplex, bi-directional, persistent TCP connection after HTTP upgrade.

| Criterion | SSE | WebSocket |
|---|---|---|
| **Direction** | Server → Client only | Bi-directional |
| **Protocol** | HTTP/1.1 (text) | WS protocol (binary or text) |
| **Auto-reconnect** | Built-in (`EventSource`) | Must implement manually |
| **Firewall/proxy** | Works (it's HTTP) | Sometimes blocked |
| **Use case** | LLM token streaming, live feeds | Chat, collaborative editing, gaming |

**Use SSE for LLM streaming** — it is simpler, HTTP-native, and auto-reconnects. Use WebSockets only when the *client* also needs to send frequent messages mid-stream (e.g., a user typing interrupt).

```javascript
// Client-side SSE consumption
const source = new EventSource('/stream?question=What+is+my+excess%3F');
source.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.token) appendToChat(data.token);
};
source.onerror = () => source.close();
```

---

### Q9: How do you implement idempotency for agent tool calls?

**Answer:**

AI agents call tools (functions) that may have side effects: sending emails, submitting claims, writing to a database. If the agent retries a failed call, the tool might execute twice — duplicating the side effect. Idempotency guarantees that a tool called multiple times with the same inputs has the same effect as calling it once.

```python
import hashlib
import redis
import json
import functools
import time

_redis = redis.Redis.from_url(os.environ['REDIS_URL'])

def idempotent_tool(ttl: int = 3600):
    """
    Decorator: makes a tool call idempotent by caching results in Redis.
    Idempotency key = SHA256(function_name + sorted(kwargs)).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key_material = f"{func.__name__}:{json.dumps(kwargs, sort_keys=True)}"
            idem_key = "idem:" + hashlib.sha256(key_material.encode()).hexdigest()

            # Check cache
            cached = _redis.get(idem_key)
            if cached:
                return json.loads(cached)

            # Execute tool (exactly once)
            result = func(*args, **kwargs)
            _redis.setex(idem_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

@idempotent_tool(ttl=86400)
def submit_claim(policy_number: str, incident_date: str, amount: float) -> dict:
    """This tool is now safe to retry — duplicates are deduplicated in Redis."""
    return claims_api.create_claim(
        policy=policy_number,
        date=incident_date,
        value=amount
    )
```

> [!CAUTION]
> The idempotency key must cover all inputs that affect the outcome. If `policy_number` and `incident_date` are the same but `amount` differs, they are different operations and should not share an idempotency key. Use all relevant parameters in the key hash.

---

### Q10: How do you debug a slow RAG pipeline?

**Answer:**

A slow RAG pipeline can be caused by bottlenecks in any of the three stages. Approach this **systematically with instrumentation** before optimising.

**Step 1 — Instrument with spans:**

```python
# With OpenTelemetry (see Section 1 complete code), you get per-stage timings:
# embed_ms, retrieve_ms, rerank_ms, generate_ms
```

**Step 2 — Identify where the time goes:**

| Stage | Typical duration | Red flag |
|---|---|---|
| Embed | 50–200ms | >500ms → model overloaded, batch instead |
| Hybrid Retrieve | 20–100ms | >500ms → index too large, HNSW exhaustive mode, missing index |
| Rerank | 50–200ms | >500ms → cross-encoder too large, use smaller model |
| Generate | 500ms–5s | >5s → context too long, streaming → TTF token fix |

**Step 3 — Common fixes:**

```python
# Fix 1: Embed is slow → switch from sync to async batching
# Fix 2: Retrieve is slow → check HNSW exhaustive flag (should be False)
vector_query = VectorizedQuery(vector=embedding, exhaustive=False)  # ← HNSW, not exact

# Fix 3: Rerank is slow → use lighter model or skip for simple queries
FAST_RERANKER  = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2')  # 2-layer, fast
HEAVY_RERANKER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') # 6-layer, quality

# Fix 4: Generate is slow → context too long → tighten token budget
# Fix 5: Everything slow → parallelise embed + pre-warm the reranker at startup
# Fix 6: Check for connection-per-request anti-pattern (SearchClient created in loop)
```

**Step 4 — Use tail latency data:**

```python
# p50 slow → core logic is slow → fix algorithm
# p99 slow but p50 fine → occasional cold start, GC pause, or single slow shard
# All percentiles slow after deploy → new model is slower, new index exhaustive mode
```

---

*Harvested from `br_ins/07_coding_exploration_prep.md` on 2026-10-12 · kept verbatim · Target: Brit Insurance Lead AI Engineer*

> [!TIP]
> **Interview execution tip:** For the RAG code review, write the 10 improvement themes on paper first (30 seconds), then walk through them in order. This signals structured thinking far more than jumping to random issues. For Git, always ground your answer in a concrete example — draw the commit graph on the whiteboard.
