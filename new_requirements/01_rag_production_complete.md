# RAG PRODUCTION COMPLETE — Architecture, Chunking, Retrieval, Evaluation
## The most important topic for this role. Every detail covered.

---

## SECTION 1: RAG ARCHITECTURE — PRODUCTION GRADE

### What is RAG and Why It Matters for SME Clients

**Simple definition for clients:**
> "RAG lets your AI system answer questions using YOUR documents and data. Instead of the AI guessing, it first searches your knowledge base, finds relevant passages, then generates an answer grounded in what it found."

**The architecture (three stages):**
```
OFFLINE (Indexing Pipeline):
Document → Extract Text → Clean → Chunk → Embed → Store in Vector DB

ONLINE (Query Pipeline):
User Question → Embed → Search Vector DB → Retrieve Chunks → Construct Prompt → LLM → Answer
```

### The Two Phases in Production

**Phase 1 — Indexing (runs once + incrementally):**
```python
# Production indexing pipeline
def index_documents(documents: list[Document]) -> None:
    for doc in documents:
        # 1. Extract text (PDF, Word, Excel, HTML)
        text = extract_text(doc)
        
        # 2. Clean: normalize whitespace, remove headers/footers
        text = clean_text(text)
        
        # 3. Chunk with overlap
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        
        # 4. Embed each chunk
        embeddings = embed_model.encode(chunks)
        
        # 5. Store with metadata
        vector_db.upsert([
            {
                "id": f"{doc.id}_{i}",
                "embedding": emb,
                "text": chunk,
                "metadata": {
                    "source": doc.filename,
                    "page": chunk.page_num,
                    "doc_type": doc.type,
                    "created_at": doc.created_at,
                    "client_id": doc.client_id  # Multi-tenant isolation
                }
            }
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ])
```

**Phase 2 — Query (runs on every user question):**
```python
def rag_query(question: str, client_id: str) -> RAGResponse:
    # 1. Embed the question
    query_embedding = embed_model.encode(question)
    
    # 2. Retrieve relevant chunks (with metadata filter for tenant isolation)
    retrieved = vector_db.search(
        embedding=query_embedding,
        top_k=5,
        filter={"client_id": client_id}  # CRITICAL for multi-tenancy
    )
    
    # 3. Rerank results (optional but important for quality)
    reranked = reranker.rerank(question, [r.text for r in retrieved])
    
    # 4. Construct prompt with context
    context = "\n\n".join([r.text for r in reranked[:3]])
    prompt = build_rag_prompt(question, context)
    
    # 5. Generate answer
    response = llm.complete(prompt)
    
    # 6. Return with sources for transparency
    return RAGResponse(
        answer=response.text,
        sources=[r.metadata["source"] for r in reranked[:3]],
        confidence=calculate_confidence(reranked)
    )
```

---

## SECTION 2: CHUNKING STRATEGIES — CRITICAL PRODUCTION DETAIL

### Why Chunking Matters

**The core tradeoff:**
- **Too small chunks (< 100 tokens):** Miss context. A sentence alone may not have enough information.
- **Too large chunks (> 1000 tokens):** Dilute precision. Retrieving a 5-page chapter for a specific fact. Also hits context window limits.
- **Overlap too small:** Patterns split across chunk boundaries are missed.
- **Overlap too large:** Redundant retrieval, higher cost.

### Chunking Strategies by Document Type

**Strategy 1: Fixed-Size with Overlap (Default)**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # tokens (approx 375 words)
    chunk_overlap=100,     # tokens (20% overlap)
    separators=["\n\n", "\n", ". ", " ", ""],  # try these in order
    length_function=len    # or use tiktoken for exact token count
)
chunks = splitter.split_text(document_text)
```
- **When to use:** General documents, mixed content
- **When NOT to use:** Highly structured documents (invoices, tables)

**Strategy 2: Semantic Chunking (Higher Quality)**
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Split where semantic similarity drops significantly
splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=95           # 95th percentile of similarity drops
)
chunks = splitter.split_text(document_text)
```
- **When to use:** Legal contracts, technical manuals — where topic boundaries matter
- **When NOT to use:** Time-sensitive indexing (each split requires embedding calls = slow + expensive)

**Strategy 3: Document-Aware Chunking**
```python
# For PDFs with structure: split by section/heading
def chunk_by_section(pdf_text: str) -> list[str]:
    # Split on headings (regex patterns for typical Italian legal/business docs)
    sections = re.split(r'\n(?=(?:Art\.|Articolo|Capitolo|Sezione|\d+\.)\s)', pdf_text)
    chunks = []
    for section in sections:
        if len(section) > 1000:  # Large section: recursive split
            sub_chunks = fixed_size_splitter.split_text(section)
            chunks.extend(sub_chunks)
        else:
            chunks.append(section)
    return chunks
```
- **When to use:** Legal contracts, Italian fiscal documents (F24, fattura), contracts
- **When NOT to use:** Unstructured emails, notes

**Strategy 4: Parent-Child Chunking (for SME document Q&A)**
```python
# Parent: 2000-token sections (for context)
# Child: 200-token snippets (for precise retrieval)
# Retrieve child, but return parent as context to LLM

def parent_child_index(document: str):
    parent_chunks = split_into_sections(document, size=2000)
    for i, parent in enumerate(parent_chunks):
        child_chunks = split_into_snippets(parent, size=200)
        for child in child_chunks:
            vector_db.upsert({
                "embedding": embed(child),
                "text": child,
                "metadata": {
                    "parent_text": parent,  # Store parent for context retrieval
                    "parent_id": i
                }
            })

# At query time: retrieve child, return parent to LLM
retrieved_child = vector_db.search(query_embedding)
context_for_llm = retrieved_child.metadata["parent_text"]  # Richer context
```
- **Best for:** SME Q&A where you need precise retrieval + full context

### Token Counting (Production Must-Have)
```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def chunk_by_tokens(text: str, max_tokens: int = 500, overlap: int = 100) -> list[str]:
    enc = tiktoken.encoding_for_model("gpt-4o")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start = end - overlap  # Overlap by rolling back
    return chunks
```

---

## SECTION 3: EMBEDDING MODELS — WHEN TO USE WHICH

### Comparison Table

| Model | Dimensions | Cost | Quality | Speed | Use Case |
|-------|-----------|------|---------|-------|----------|
| OpenAI text-embedding-3-small | 1536 | $0.02/1M tokens | High | Fast | Most production cases |
| OpenAI text-embedding-3-large | 3072 | $0.13/1M tokens | Highest | Medium | Complex semantic search |
| Cohere embed-v3 | 1024 | $0.10/1M tokens | High | Fast | Multilingual (Italian!) |
| sentence-transformers/BAAI/bge-large | 1024 | Free (local) | High | Slow (CPU) | On-premise, GDPR sensitive |
| sentence-transformers/multilingual-e5-large | 1024 | Free (local) | Good | Slow | Italian + multilingual on-prem |
| Ollama (local) | Various | Free | Medium | Very slow (CPU) | Air-gapped SME environments |

**Italian language consideration (CRITICAL for this role):**
```
For Italian SME clients:
- OpenAI embeddings: work well for Italian (multilingual training)
- Cohere embed-multilingual-v3: explicitly multilingual, strong Italian
- multilingual-e5-large: best on-premise option for Italian
- BGE: mostly English, weaker for Italian
```

**Interview answer on embedding choice:**
> "For Italian SME clients, my default is OpenAI text-embedding-3-small — it handles Italian well, is cost-effective at $0.02/1M tokens, and the 1536 dimensions are sufficient for most SME document volumes. For GDPR-sensitive clients who can't send data to OpenAI, I deploy multilingual-e5-large locally via sentence-transformers — it runs on CPU, handles Italian, and keeps data on-premise. I'd test both on 50 representative client documents before final selection."

---

## SECTION 4: RETRIEVAL STRATEGIES — BEYOND SIMPLE VECTOR SEARCH

### Problem with Pure Vector Search
Vector similarity alone fails when:
- User asks "What is the invoice number for the order from March 15?" → keyword search finds exact numbers better
- Query and document use different vocabulary (the user says "payment" but document says "versamento")
- Need to filter by date range, document type, client

### Hybrid Search (Production Standard)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

def hybrid_search(
    query: str,
    collection: str,
    top_k: int = 10,
    alpha: float = 0.5  # 0=pure keyword, 1=pure semantic
) -> list[SearchResult]:
    
    # Dense embedding (semantic)
    dense_vector = embed_model.encode(query)
    
    # Sparse vector (BM25-style keyword)
    sparse_vector = bm25_encoder.encode_query(query)
    
    # Hybrid search with Reciprocal Rank Fusion
    results = qdrant_client.query_points(
        collection_name=collection,
        prefetch=[
            # Dense (semantic) results
            models.Prefetch(query=dense_vector, using="dense", limit=20),
            # Sparse (keyword) results
            models.Prefetch(query=SparseVector(**sparse_vector), using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),  # Reciprocal Rank Fusion
        limit=top_k
    )
    return results
```

**Why RRF (Reciprocal Rank Fusion)?**
> "RRF combines rankings from multiple sources: score = sum(1 / (k + rank_in_source)). It's robust to score scale differences — semantic search may return scores 0.7-0.9, while BM25 returns 0-100. RRF normalizes by rank position, not score value."

### Reranking (The Quality Multiplier)

```python
# Step 1: Retrieve top-20 candidates (cheap)
# Step 2: Rerank with cross-encoder (expensive but only on 20)
# Step 3: Return top-3 to LLM

from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # Fast + accurate

def retrieve_and_rerank(query: str, top_k_retrieve: int = 20, top_k_return: int = 5):
    # Cheap: vector search
    candidates = vector_db.search(query, top_k=top_k_retrieve)
    
    # Expensive: cross-encoder reranking
    scores = reranker.predict([(query, c.text) for c in candidates])
    
    # Return top-k after reranking
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k_return]]
```

**Why reranking?**
> "Vector similarity measures embedding proximity, not exact relevance to the query. A cross-encoder directly reads the query AND the chunk together — much more accurate. The tradeoff: cross-encoders are 100x slower. We use two-stage retrieval: fast vector search for 20 candidates, cross-encoder on those 20. Cost = cross-encoder inference on 20 items, not 1M."

### Metadata Filtering

```python
# CRITICAL for multi-tenant SME deployments
results = vector_db.search(
    query_vector=embed(question),
    query_filter={
        "must": [
            {"key": "client_id", "match": {"value": client_id}},  # Tenant isolation
            {"key": "doc_type", "match": {"value": "fattura"}},    # Document type filter
            {"key": "date_range", "range": {                        # Date range
                "gte": "2024-01-01",
                "lte": "2024-12-31"
            }}
        ]
    },
    limit=10
)
```

---

## SECTION 5: RETRIEVAL FAILURES AND HOW TO FIX THEM

### The 5 Failure Modes in Production

**Failure 1: Semantic gap (query and document use different words)**
- User: "when do I need to pay" → Document: "scadenza di pagamento" (Italian)
- Fix: Use multilingual embeddings, add Italian query translation step, BM25 as fallback

**Failure 2: Chunk boundary splitting critical information**
- The answer spans two chunks, neither chunk retrieved alone is complete
- Fix: Parent-child chunking, increase overlap, check for these during evaluation

**Failure 3: Noisy retrieval (top chunks are irrelevant)**
- Poor embedding model or low-quality documents
- Fix: Add reranking, improve cleaning pipeline, filter by similarity score threshold

```python
MIN_SIMILARITY_SCORE = 0.75  # Tunable threshold
retrieved = [r for r in raw_retrieved if r.score >= MIN_SIMILARITY_SCORE]
if not retrieved:
    return "I don't have enough information to answer this question."
```

**Failure 4: Answer hallucination (LLM generates info not in retrieved chunks)**
- Fix: Explicit "answer only from context" instruction + faithfulness check

```python
STRICT_RAG_PROMPT = """
Answer the question using ONLY the information provided in the context below.
If the answer is not in the context, say "I don't have this information in the provided documents."
Do NOT use any external knowledge.

Context:
{context}

Question: {question}
Answer:
"""
```

**Failure 5: Context stuffing overflow**
- Retrieving too many chunks → exceeds context window → truncated → wrong answers
- Fix: Estimate token count before sending, truncate or summarize if needed

```python
MAX_CONTEXT_TOKENS = 3000  # Leave room for question + system prompt
context_chunks = []
current_tokens = 0
for chunk in reranked_chunks:
    chunk_tokens = count_tokens(chunk.text)
    if current_tokens + chunk_tokens <= MAX_CONTEXT_TOKENS:
        context_chunks.append(chunk)
        current_tokens += chunk_tokens
    else:
        break
```

---

## SECTION 6: RAG EVALUATION — THE RAGAS FRAMEWORK

### The Four Ragas Metrics You Must Know

**1. Faithfulness (Most Important for Production)**
```
Faithfulness = (# statements in answer supported by retrieved context) / (total statements in answer)
```
- Measures: does the answer hallucinate?
- Target: > 0.90 in production
- Low faithfulness → model is making things up from training data, not retrieved docs

**2. Answer Relevancy**
```
Answer Relevancy = cosine_similarity(answer_embedding, question_embedding)
```
- Measures: does the answer actually address the question?
- Target: > 0.80
- Low relevancy → answer is on-topic but evasive

**3. Context Recall**
```
Context Recall = (# ground truth statements covered by retrieved context) / total ground truth statements
```
- Measures: did retrieval capture the necessary information?
- Target: > 0.75
- Low recall → retrieval is missing relevant chunks → fix chunking or search

**4. Context Precision**
```
Context Precision = (# relevant chunks in retrieved set) / (total retrieved chunks)
```
- Measures: is retrieval returning noise alongside relevant content?
- Target: > 0.70
- Low precision → reranker needed, or top_k too high

### Running Ragas in Production

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

def evaluate_rag_system(test_cases: list[dict]) -> dict:
    """
    test_cases format:
    [{"question": "...", "answer": "...", "contexts": ["...", "..."], "ground_truth": "..."}]
    """
    dataset = Dataset.from_list(test_cases)
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=evaluation_llm,      # Use stronger model for judging (GPT-4o)
        embeddings=embed_model,  # For relevancy calculation
    )
    
    return {
        "faithfulness": results["faithfulness"],
        "answer_relevancy": results["answer_relevancy"],
        "context_recall": results["context_recall"],
        "context_precision": results["context_precision"],
        "overall": (results["faithfulness"] + results["answer_relevancy"]) / 2
    }
```

### Building a Ground Truth Dataset for SME Clients

```
Step 1: Collect 50-100 representative questions from client (domain expert)
Step 2: Have domain expert provide ideal answers to each question
Step 3: Run RAG system on all questions
Step 4: Calculate Ragas metrics + manual spot check
Step 5: Set acceptance threshold (e.g., Faithfulness > 0.85, Context Recall > 0.75)
Step 6: Only deploy if thresholds met on entire test set
Step 7: Re-evaluate weekly in production on new test cases
```

### Regression Testing for RAG

```python
# Every code/prompt change must not degrade performance
def run_regression_tests(new_rag_system, baseline_metrics: dict) -> bool:
    current_metrics = evaluate_rag_system(REGRESSION_TEST_CASES)
    
    DEGRADATION_THRESHOLD = 0.05  # 5% degradation threshold
    
    for metric, baseline_value in baseline_metrics.items():
        current_value = current_metrics[metric]
        if current_value < baseline_value - DEGRADATION_THRESHOLD:
            print(f"REGRESSION DETECTED: {metric} dropped from {baseline_value:.3f} to {current_value:.3f}")
            return False
    
    return True  # Safe to deploy
```

---

## SECTION 7: ADVANCED RAG PATTERNS

### Self-RAG (Handle "I Don't Know")
```python
def self_rag_query(question: str) -> str:
    # Step 1: Check if retrieval is needed
    need_retrieval = llm.classify(f"Does answering '{question}' require looking up documents? Yes/No")
    
    if need_retrieval == "No":
        return llm.answer(question)  # Direct answer for general questions
    
    # Step 2: Retrieve
    chunks = retrieve(question)
    
    # Step 3: Check if retrieved chunks are relevant
    is_relevant = llm.classify(f"Are these chunks relevant to '{question}'? Yes/No\nChunks: {chunks}")
    
    if is_relevant == "No":
        return "I don't have this information in your documents."
    
    # Step 4: Generate with context
    answer = llm.generate(question, chunks)
    
    # Step 5: Check if answer is supported
    is_faithful = llm.classify(f"Is this answer supported by the chunks? Yes/No\nAnswer: {answer}\nChunks: {chunks}")
    
    if is_faithful == "No":
        return "I found relevant documents but cannot generate a reliable answer."
    
    return answer
```

### Multi-Hop RAG (For Complex Questions)
For questions requiring multiple document lookups:
```python
def multi_hop_rag(question: str, max_hops: int = 3) -> str:
    # Decompose complex question into sub-questions
    sub_questions = llm.decompose(question)
    # Example: "What is the total payment for all invoices from Cliente X in 2024?"
    # → ["Who is Cliente X?", "What invoices exist for Cliente X?", "What are the amounts?"]
    
    context_accumulation = []
    for sub_q in sub_questions:
        chunks = retrieve(sub_q, existing_context=context_accumulation)
        context_accumulation.extend(chunks)
    
    return llm.synthesize(question, context_accumulation)
```

### Corrective RAG (CRAG)
```python
def corrective_rag(question: str) -> str:
    chunks = retrieve(question)
    
    # Grade each retrieved chunk
    relevance_scores = [grade_relevance(question, chunk) for chunk in chunks]
    
    high_quality = [c for c, s in zip(chunks, relevance_scores) if s > 0.8]
    low_quality = [c for c, s in zip(chunks, relevance_scores) if s < 0.5]
    
    if len(high_quality) == 0 and len(low_quality) > 0:
        # All retrieved chunks are poor quality → fall back to web search
        web_results = web_search(question)
        return llm.answer_with_web(question, web_results)
    
    return llm.answer(question, high_quality)
```

---

## SECTION 8: PRODUCTION RAG ISSUES (CRITICAL FOR THIS ROLE)

### Issue 1: Indexing Pipeline Failures

**Problem:** Document processing fails silently — PDF extraction returns empty string, embedding call times out
**Detection:** Validate after each step

```python
def robust_index_document(doc_path: str) -> IndexResult:
    # Validate extraction
    text = extract_text(doc_path)
    if len(text.strip()) < 50:
        return IndexResult(status="failed", reason=f"Insufficient text extracted: {len(text)} chars")
    
    # Validate chunking
    chunks = chunk_text(text)
    if len(chunks) == 0:
        return IndexResult(status="failed", reason="No chunks produced")
    
    # Validate embedding (retry with backoff)
    try:
        embeddings = embed_with_retry(chunks, max_retries=3, backoff=2.0)
    except EmbeddingError as e:
        return IndexResult(status="failed", reason=f"Embedding failed: {e}")
    
    # Commit to vector DB
    vector_db.upsert(chunks, embeddings, metadata)
    
    return IndexResult(status="success", chunks_indexed=len(chunks))
```

### Issue 2: Stale Index (Document Updated but Not Re-indexed)

```python
# Track document versions
def upsert_document(doc_id: str, doc_path: str):
    # Hash document content to detect changes
    new_hash = md5(open(doc_path, 'rb').read()).hexdigest()
    existing = index_registry.get(doc_id)
    
    if existing and existing.content_hash == new_hash:
        return  # No change, skip re-indexing
    
    # Delete old chunks for this document
    vector_db.delete(filter={"doc_id": doc_id})
    
    # Re-index with new content
    index_document(doc_path, doc_id=doc_id)
    
    # Update registry
    index_registry.update(doc_id, content_hash=new_hash, indexed_at=datetime.now())
```

### Issue 3: Cost Explosion

**Problem:** Each query calls embedding + LLM → costs add up fast for SME clients
**Solution:** Caching + batching

```python
import hashlib
from functools import lru_cache
import redis

cache = redis.Redis(host='localhost', port=6379)
CACHE_TTL = 3600  # 1 hour

def cached_embed(text: str) -> list[float]:
    cache_key = f"embed:{hashlib.md5(text.encode()).hexdigest()}"
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)
    
    embedding = embed_model.encode(text)
    cache.setex(cache_key, CACHE_TTL, json.dumps(embedding))
    return embedding

def cached_rag_response(question: str, client_id: str) -> str:
    # Cache exact question answers
    cache_key = f"rag:{client_id}:{hashlib.md5(question.encode()).hexdigest()}"
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)
    
    response = rag_query(question, client_id)
    # Cache for 1 hour — SME docs don't change that often
    cache.setex(cache_key, CACHE_TTL, json.dumps(response))
    return response
```

**Cost monitoring:**
```python
import tiktoken

def estimate_query_cost(prompt: str, expected_output_tokens: int = 500) -> float:
    enc = tiktoken.encoding_for_model("gpt-4o")
    input_tokens = len(enc.encode(prompt))
    
    # GPT-4o pricing (2025): $5/1M input, $15/1M output
    input_cost = (input_tokens / 1_000_000) * 5
    output_cost = (expected_output_tokens / 1_000_000) * 15
    
    return input_cost + output_cost

# Track costs per client
def log_query_cost(client_id: str, cost: float):
    cost_tracker.increment(f"cost:{client_id}:{date.today()}", cost)
    monthly_cost = cost_tracker.get(f"cost:{client_id}:{month}")
    
    if monthly_cost > CLIENT_COST_LIMIT:
        alert_sales_team(client_id, monthly_cost)
```

### Issue 4: Multi-Tenant Data Isolation

**Critical for SME clients:** Client A must NEVER see Client B's documents.

```python
# Option 1: Separate collections per client (strongest isolation, higher cost)
def get_client_collection(client_id: str) -> str:
    return f"client_{client_id}_documents"

# Option 2: Shared collection with metadata filtering (lower cost, relies on filter correctness)
def search_with_tenant_filter(query: str, client_id: str):
    return vector_db.search(
        query_vector=embed(query),
        query_filter={"must": [{"key": "client_id", "match": {"value": client_id}}]},
        limit=10
    )

# NEVER do this (no tenant filter = data leak):
def DANGEROUS_search(query: str):
    return vector_db.search(query_vector=embed(query), limit=10)  # Returns all clients' data!
```

**Interview answer on multi-tenancy:**
> "For SME clients, I use metadata filtering with a mandatory client_id filter on every search operation. To prevent the 'missing filter bug' that would leak data across clients, I wrap the search function to always inject the client_id from the authenticated session — it's impossible to call search without a tenant context. For extremely sensitive clients (legal, healthcare), I use separate Qdrant collections per client to provide true data isolation at the infrastructure level."

---

## SECTION 9: RAG FOR SPECIFIC SME USE CASES

### Use Case 1: Invoice/Contract Q&A
- Document type: PDFs, sometimes scanned
- Key challenge: Tables in invoices (pricelist, line items)
- Solution: Table-aware extraction (Camelot for digital PDFs, GPT-4V for scanned tables)
- Chunking: Split by section (header, line items, payment terms separately)

### Use Case 2: Italian Legal Document Search
- Document type: Legal contracts, circulari, normativa
- Key challenge: Italian legal language, article references ("ex art. 1456 c.c.")
- Solution: Italian-language embedding model, structured chunking by article number
- Special: Legal citation graph (article X references article Y)

### Use Case 3: Internal Knowledge Base (HR, Procedures)
- Document type: Word docs, internal wikis, PDFs
- Key challenge: Outdated documents still in index
- Solution: Expiry dates in metadata, crawl-based freshness checks

### Use Case 4: Product Manual / Technical Support
- Document type: User manuals, technical specs
- Key challenge: Similar questions phrased differently by different operators
- Solution: Query expansion (generate multiple paraphrases before search)

```python
def query_expansion(question: str) -> list[str]:
    expansion_prompt = f"""Generate 3 alternative phrasings of this question in Italian:
    Original: {question}
    Return as JSON array: ["version1", "version2", "version3"]"""
    
    expanded = llm.generate(expansion_prompt)
    variants = json.loads(expanded)
    return [question] + variants  # Original + 3 expansions

def search_with_expansion(question: str):
    variants = query_expansion(question)
    all_results = []
    for variant in variants:
        results = vector_db.search(embed(variant), top_k=5)
        all_results.extend(results)
    
    # Deduplicate and rerank
    deduplicated = deduplicate_by_chunk_id(all_results)
    return reranker.rerank(question, deduplicated)[:5]
```
