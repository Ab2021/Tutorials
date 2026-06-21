# 🔍 RAG & Vector Infrastructure — Deep Dive Interview Prep
### Abhishek Bhardwaj | Solutions Architect ML/AI @ Huge

---

> [!IMPORTANT]
> The JD explicitly requires: *"RAG, semantic routing and caching"*, *"Vector Infrastructure"*, *"Embeddings, Vectors"*. Your Chubb Fraud Detection RAG system and FAISS/Chroma experience are your primary anchors. Connect everything to real production experience.

---

## SECTION 1: RAG Fundamentals & Architecture

---

### Q1.1: "Walk me through your RAG pipeline at Chubb end to end."

**Why they're asking**: They want to see that you built a production RAG system, not just read about one. They'll drill into every component.

**Model Answer**:

The RAG pipeline for insurance fraud detection at Chubb was an end-to-end system processing unstructured claims documents — police reports, medical records, adjuster notes, policy documents — to surface fraud signals. Let me walk through every layer.

**Ingestion & Preprocessing**: Claims documents arrive as PDFs, Word documents, and scanned images via our document management system. The pipeline extracted text using a combination of PDFMiner (for digital PDFs), Tesseract OCR (for scanned documents), and table extraction logic (camelot for structured tables within PDFs). We normalized encoding, stripped boilerplate headers/footers, and structured the text into a canonical document format with metadata: `{claim_id, policy_number, document_type, date, insured_name, extracted_text}`.

**Chunking Strategy**: We used **hierarchical chunking** — a parent-child approach. Parent chunks were semantic sections of the document (a claim's "incident description" as one chunk, "medical assessment" as another). Child chunks were smaller 256-token windows within each section. At retrieval time, we retrieved child chunks (better semantic precision) but returned the parent context (more coherent text to the LLM). This small-to-big retrieval pattern significantly improved answer quality compared to naive fixed-size chunking.

For the fraud detection use case specifically, we added **claim-aware chunking**: we identified structured fields (claim date, amount, coverage type) and always kept them as standalone metadata fields rather than embedding them in text chunks. This allowed hard metadata filtering during retrieval (e.g., "only retrieve from Auto claims in the last 90 days").

**Embedding & Indexing**: We used `text-embedding-ada-002` for most documents, but evaluated `BGE-large-en-v1.5` (BAAI's model) and found it outperformed ada-002 by ~4% on our domain-specific retrieval benchmark. We ultimately kept ada-002 for production due to infrastructure simplicity (single API, no self-hosting), with plans to migrate to BGE with a self-hosted sentence-transformers deployment.

We indexed into **FAISS with an IVF-HNSW hybrid index**: IVF (Inverted File Index) for partitioning the vector space into ~500 clusters (nlist=500), enabling fast approximate search, with HNSW's graph-based structure for high-accuracy ANN within each cluster. This gave us sub-50ms retrieval for a 2M-vector index.

```
Architecture:
Claims Docs → Text Extraction → Chunking (parent-child) → Embedding → FAISS Index
                ↓                                              ↑
           Metadata DB (PostgreSQL)                   Metadata filtering
                ↓
          At query time:
Query → Embed query → FAISS similarity search (top-20) → Re-rank (top-5) → LLM
```

**Retrieval**: We used **hybrid retrieval** — combining dense retrieval (FAISS cosine similarity) with sparse BM25 retrieval (via Elasticsearch on the same document corpus). Results were merged using **Reciprocal Rank Fusion (RRF)**: `RRF_score(doc) = Σ 1/(k + rank_dense) + 1/(k + rank_sparse)` where k=60. Hybrid retrieval outperformed either alone by ~8% on our evaluation set, especially for queries with specific medical codes or policy terms (where BM25's exact matching excels).

**Re-ranking**: Top-20 candidates from RRF were re-ranked using a cross-encoder (we used `cross-encoder/ms-marco-MiniLM-L-6-v2` initially, then moved to Cohere Rerank API for better quality). The cross-encoder scores query-document relevance jointly rather than independently, which is more accurate but too slow to run on all documents. Running it on the top-20 pre-filtered candidates was efficient.

**Generation**: The top-5 re-ranked chunks were passed to GPT-4 with a fraud-analysis prompt that instructed the model to: (a) identify specific fraud indicators present in the retrieved context, (b) cite the specific claim element supporting each indicator, and (c) output a structured fraud risk assessment with confidence levels. The citation requirement was key for explainability — insurance adjusters need to see exactly which part of the document triggered the fraud flag.

**Cross-questions**:

**Q: "Why did you choose FAISS over a managed vector database like Pinecone?"**
Answer: At the time of the Chubb implementation, we had strict data residency requirements — all insurance claims data had to remain within our AWS VPC. A managed cloud vector DB would have required data leaving our perimeter or complex VPC peering. FAISS running within our SageMaker infrastructure gave us full control. For a Huge client scenario without such constraints, I'd absolutely evaluate Vertex AI Vector Search (native GCP integration) or Pinecone for faster time-to-value.

**Q: "In your fraud detection RAG system, how did you chunk the claims documents?"**
Answer: *(covered above in the model answer — point to the hierarchical/parent-child strategy and claim-aware metadata chunking)*

---

### Q1.2: "Explain the difference between Naive RAG, Advanced RAG, and Modular RAG."

**Model Answer**:

**Naive RAG** is the classic retrieve-then-generate pipeline: embed the query, retrieve top-k documents, concatenate them into the prompt, generate. Simple to implement, but suffers from several failure modes: (a) poor retrieval quality propagates directly to poor generation, (b) no feedback between generation quality and retrieval, (c) fixed retrieval regardless of query type.

**Advanced RAG** addresses Naive RAG's failures with improvements at each stage:
- *Pre-retrieval*: Query transformation (HyDE, multi-query expansion, step-back prompting) to improve retrieval quality
- *Retrieval*: Hybrid retrieval (dense + sparse + RRF), re-ranking, parent-child retrieval
- *Post-retrieval*: Contextual compression (extract only relevant sentences from retrieved chunks), re-ordering (put most relevant chunks in the middle — models attend to beginning and end best)

**Modular RAG** is the most flexible — it treats RAG as a composable system of independent modules (retrieval module, re-ranking module, memory module, fusion module) that can be swapped, combined, or replaced. This maps to how we'd architect a production system at Huge: the retrieval strategy for a legal compliance use case is completely different from a customer support use case, but the generation and memory modules can be shared. Modular RAG enables this separation of concerns.

| Feature | Naive RAG | Advanced RAG | Modular RAG |
|---------|-----------|-------------|------------|
| Query transformation | ❌ | ✅ (HyDE, multi-query) | ✅ Pluggable |
| Hybrid retrieval | ❌ | ✅ | ✅ |
| Re-ranking | ❌ | ✅ | ✅ |
| Feedback loop | ❌ | Partial (CRAG) | ✅ |
| Flexibility | Low | Medium | High |
| Complexity | Low | Medium | High |

---

## SECTION 2: Advanced RAG Techniques

---

### Q2.1: "What is HyDE? Explain with an example."

**Model Answer**:

HyDE (Hypothetical Document Embeddings) addresses a fundamental mismatch in RAG: **queries and documents have different linguistic distributions**. A query like "Why was this claim denied?" is short and question-form. The relevant document chunk is a long, formal claim denial letter in statement form. These have different embeddings, so direct query-to-document similarity is suboptimal.

HyDE's insight: instead of embedding the query directly, **ask the LLM to generate a hypothetical document that would answer the query**, then embed *that hypothetical document*. The hypothetical document is in the same linguistic space as the corpus, so similarity search is much more effective.

**Example** (insurance fraud context):
- Query: "What are common indicators of staged auto accidents?"
- HyDE generates: "Common indicators of staged auto accidents include: multiple claimants reporting identical injuries, claims filed within 24 hours of policy inception, witnesses who are known associates of the claimant, damage inconsistent with the reported collision speed, and prior claims by the same insured at short intervals..."
- This hypothetical document embeds much closer to actual fraud investigation reports in the corpus than the short query would.

In my Chubb system, HyDE improved retrieval recall by ~12% on fraud-indicator queries compared to direct query embedding, with minimal latency overhead (one extra LLM call, but using a fast model like GPT-3.5 for the hypothesis generation).

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def hyde_retrieve(query: str, vectorstore, k: int = 5) -> list:
    # Step 1: Generate hypothetical document
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    hypothesis_prompt = f"""Generate a detailed passage that would directly answer this question.
    Write it as if it were from an insurance industry document.
    Question: {query}
    Passage:"""
    hypothesis = llm.invoke(hypothesis_prompt).content
    
    # Step 2: Embed the hypothesis (not the query)
    embeddings = OpenAIEmbeddings()
    hypothesis_embedding = embeddings.embed_query(hypothesis)
    
    # Step 3: Retrieve using hypothesis embedding
    results = vectorstore.similarity_search_by_vector(hypothesis_embedding, k=k)
    return results
```

---

### Q2.2: "What is Corrective RAG (CRAG)? When would you use it?"

**Model Answer**:

CRAG introduces a **retrieval quality assessment step** between retrieval and generation. Instead of blindly passing retrieved documents to the LLM regardless of quality, CRAG evaluates whether the retrieved documents are actually relevant to the query. If not, it triggers a fallback — typically a web search or alternative knowledge source.

The CRAG workflow:
1. Retrieve documents from vector store
2. **Evaluate retrieval quality** using a lightweight evaluator (a small LLM or a cross-encoder scoring query-document relevance)
3. If confidence is high (>threshold): proceed to generation
4. If confidence is medium: combine retrieved docs with web-searched context
5. If confidence is low: discard retrieved docs entirely and fall back to web search

This is particularly valuable for **open-domain QA** where the vector store may not have coverage of every possible query. For Huge's use case: a brand assistant that primarily answers questions from internal brand guidelines, but can fall back to web search for questions about competitor brands or industry trends.

In my fraud detection system, I implemented a simpler version: after retrieval, a fast cross-encoder scored each retrieved chunk against the query. If all scores were below 0.3, I triggered a "low retrieval confidence" response that told the LLM to explicitly state it couldn't find direct evidence and base its assessment on general fraud patterns rather than specific retrieved context.

⚠️ **Trap**: Don't confuse CRAG with Self-RAG.  
**Self-RAG**: The model itself decides *whether to retrieve* on each generation step (using special tokens like `[Retrieve]`, `[No Retrieve]`) — it's about retrieval timing. **CRAG**: Retrieval always happens, but quality is assessed *after* retrieval — it's about retrieval quality gating.

---

### Q2.3: "What is Agentic Chunking? How does it differ from semantic chunking?"

**Model Answer**:

Traditional chunking is rule-based (fixed size, sentence boundary, recursive character splitting). Semantic chunking uses an embedding model to identify natural topic boundaries (split where embedding similarity between adjacent sentences drops sharply). Both are still fundamentally **local** — they make chunking decisions based on the text structure alone.

**Agentic chunking** (popularized by Greg Kamradt) uses an LLM to make chunking decisions based on **document semantics and purpose**. The LLM is asked: "Does this sentence start a new proposition or is it a continuation of the previous one?" This captures the document's logical structure, not just syntactic boundaries.

For insurance claim documents with mixed content (structured policy info, narrative incident description, medical records tables, adjuster comments), agentic chunking is far superior — it correctly identifies section boundaries that semantic chunking misses because the embedding similarity within a "Policy Details" section can be very high even when the content crosses logical boundaries.

The tradeoff: agentic chunking is expensive (one LLM call per chunk boundary decision) and slow. I'd use it for a one-time indexing pass on a curated document library, not for real-time ingestion of high-volume claims.

---

## SECTION 3: Vector Databases — Deep Architecture

---

### Q3.1: "Compare FAISS to Pinecone, Qdrant, Weaviate, pgvector, and Vertex AI Vector Search."

**Model Answer**:

| Feature | FAISS | Pinecone | Qdrant | Weaviate | pgvector | Vertex AI VS |
|---------|-------|----------|--------|----------|----------|-------------|
| **Type** | Library | Managed SaaS | Self/Cloud | Self/Cloud | PostgreSQL ext | Managed GCP |
| **Scaling** | Manual | Automatic | Manual | Manual | Manual | Automatic |
| **Filtering** | Post-filter | Pre/Post | In-filter | In-filter | Pre-filter | Pre-filter |
| **Multi-tenant** | Manual | Namespaces | Collections | Multi-tenancy native | Row-level security | Index-per-project |
| **Hybrid search** | BM25 manual | ❌ (sparse beta) | ✅ native | ✅ native | PGSearch needed | ❌ |
| **Data residency** | Full control | Vendor cloud | Full control | Full control | Full control | GCP region |
| **Cost** | Infra only | Per-vector + query | Infra only | Infra only | Infra only | GCP pricing |
| **Best for** | Research/tight infra control | Fast MVP, no ops | Production self-hosted, high filter rate | Hybrid search, knowledge graphs | Existing Postgres apps | GCP-native, Huge's stack |

**My recommendation for Huge's GCP-first architecture**: **Vertex AI Vector Search** for production workloads requiring GCP integration (direct connections to BigQuery, Vertex AI models, no data egress), **Qdrant** for self-hosted scenarios requiring complex filtering (Qdrant's payload filtering is the most efficient — it filters BEFORE ANN search, not after), **pgvector** for applications already using Cloud SQL/AlloyDB where operational simplicity trumps performance.

**FAISS index types deep dive**:
- `IndexFlatL2`: Exact search, no approximation. Use for <1M vectors or when precision is critical. O(n) per query.
- `IndexIVFFlat`: Partitions vectors into k clusters (nlist). At query time, only searches nprobe clusters. ~10-100x faster than Flat with some recall loss. Best for 1M-100M vectors.
- `IndexHNSWFlat`: Graph-based ANN. Fastest retrieval with high recall, but large memory footprint. Best when you can afford the RAM.
- `IndexIVFPQ`: IVF + Product Quantization for memory compression. Compresses vectors by ~8-32x at the cost of some accuracy. Best for 100M+ vectors with memory constraints.

```python
import faiss
import numpy as np

d = 1536  # OpenAI ada-002 embedding dimension
n_vectors = 2_000_000

# Production FAISS index for 2M vectors
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, 500)  # 500 clusters
index.nprobe = 20  # Search 20 clusters per query (recall vs speed tradeoff)

# Train on representative sample (required for IVF)
sample = np.random.randn(100_000, d).astype('float32')
index.train(sample)
index.add(vectors)  # Add all 2M vectors

# GPU acceleration for large-scale
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
```

---

### Q3.2: "What is Vertex AI Vector Search and why is it relevant for Huge?"

**Model Answer**:

Vertex AI Vector Search (formerly Matching Engine) is Google's managed ANN service, built on the same ScaNN (Scalable Nearest Neighbor) technology that powers Google Search and YouTube recommendations internally. It's important for Huge because:

**Native GCP Integration**: No data movement costs or security boundaries between BigQuery (where Huge's data likely lives), Vertex AI models (for embeddings), and Vector Search (for retrieval). A single service account can handle the entire pipeline.

**Scale**: Handles billions of vectors with <10ms latency at 99th percentile — far beyond what FAISS can do without significant engineering effort on sharding and orchestration.

**Streaming updates**: Unlike FAISS (which requires index rebuilding for updates), Vertex AI Vector Search supports streaming updates — new embeddings are available for retrieval within minutes.

**Deployment architecture for Huge**:
```
Document Ingestion → Cloud Functions → Vertex AI Embeddings API 
    → Vector Search Index (update stream)
    → BigQuery (metadata + full documents)

Query Time:
User Query → Vertex AI Embeddings → Vector Search (ANN) → BigQuery (metadata join)
    → Re-ranking (Cloud Run service) → Vertex AI Gemini (generation)
```

---

## SECTION 4: Semantic Caching & Routing

---

### Q4.1: "What is semantic caching? How does it work? Design it for production."

**Why they're asking**: Explicitly in the JD. Shows you think about latency and cost optimization, not just functionality.

**Model Answer**:

Semantic caching recognizes that in real-world LLM applications, many queries are semantically similar even if lexically different. "Show me fraud claims from January" and "What fraudulent claims occurred in January?" should return the same cached answer. Keyword-based caching would miss this; semantic caching catches it.

**Architecture**:
1. Incoming query → embed using fast embedding model (text-embedding-3-small, ~$0.00002/query)
2. Compare embedding against a cache store (Redis with vector similarity or a small FAISS index)
3. If cosine similarity > threshold (typically 0.92-0.95): return cached response immediately
4. If cache miss: execute full RAG+LLM pipeline, store `{embedding, query, response, timestamp}` in cache
5. Cache eviction: TTL-based (expire fraud analysis results after 24h since underlying data may change) + LRU

**Implementation with GPTCache**:
```python
from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# Initialize semantic cache
onnx = Onnx()  # Fast local embedding model
data_manager = get_data_manager(
    CacheBase("sqlite"),  # Metadata store
    VectorBase("faiss", dimension=onnx.dimension)  # Vector similarity
)
cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
)
cache.set_openai_key()

# Now all openai calls check cache automatically
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What fraudulent claims occurred in January?"}],
    cache_obj=cache
)
```

**Production considerations**:
- **Threshold tuning**: Too high (0.98) → almost no cache hits, little value. Too low (0.85) → false hits (semantically similar but different enough to need different answers). I recommend starting at 0.92 and adjusting based on user feedback signals.
- **Invalidation**: Cache must be invalidated when underlying data changes (new claims ingested). Use event-driven invalidation via a Pub/Sub message when new data lands.
- **Privacy**: Never cache PII-containing responses. Add a PII detector pre-cache-write and skip caching if PII is detected in the response.

---

### Q4.2: "Design a semantic routing system for Huge serving multiple client domains."

**Model Answer**:

Huge serves clients across retail (McDonald's, IKEA), tech (Google), telecom (Verizon), and consumer goods (Nike). A single query router must understand intent and route to the appropriate specialized handler.

**Router Architecture**:

```
Incoming Query
    ↓
[Intent Classifier] → Embed query → Cosine sim against route prototypes
    ↓
Route Decision:
├── [Brand/Creative RAG] → Nike/McDonald's brand guidelines, campaign history
├── [Analytics Agent] → Marketing performance data, attribution, MMM
├── [Technical Documentation RAG] → API docs, integration specs
├── [Customer Intelligence Agent] → CLV, segmentation, propensity scores
└── [General Conversation] → Fallback for unclear intent

Each route has:
├── Dedicated vector store (client-specific data isolation)
├── Specialized LLM prompt (domain-specific system prompt)
└── Custom re-ranking model (trained on domain feedback)
```

**Client isolation** is critical at Huge: Nike's brand data must never be retrievable when serving a McDonald's query. Implement via:
1. **Separate vector namespaces**: Each client gets a separate FAISS index or Pinecone namespace
2. **Metadata hard-filtering**: Every document has `client_id` metadata; all queries are filtered by the authenticated client's ID
3. **Access control at the router**: The router validates the user's client credential before dispatching to any handler

💡 **Key insight**: "The hardest part isn't the routing logic — it's building and maintaining high-quality route prototypes. The prototypes must be updated regularly as new query patterns emerge. I'd set up a feedback loop: when users reroute or reject an answer, log the query and use it to augment the prototype embeddings for that route."

---

## SECTION 5: Production RAG Evaluation — RAGAS

---

### Q5.1: "How do you evaluate a RAG system? What is RAGAS?"

**Model Answer**:

RAGAS (RAG Assessment) is an evaluation framework providing reference-free metrics for RAG systems — meaning you don't need human-labeled ground truth for every query.

**Four core RAGAS metrics**:

1. **Faithfulness** (0-1): Does the generated answer contain only statements that can be directly inferred from the retrieved context? Measures hallucination. Method: LLM decomposes the answer into individual claims, then verifies each claim against the context.

2. **Answer Relevancy** (0-1): Is the answer relevant to the user's question? Measures if the system answered the actual question asked. Method: LLM generates multiple questions that the answer could be an answer to; cosine similarity between these generated questions and the original measures relevancy.

3. **Context Recall** (0-1): Did the retrieval system retrieve all the information needed to answer the question? Requires a reference answer. Method: Decomposes the reference answer into claims; checks which claims are supportable by the retrieved context.

4. **Context Precision** (0-1): What proportion of the retrieved context is actually relevant? Measures retrieval efficiency. Higher precision = less noise fed to the LLM.

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": ["What are fraud indicators for auto claims?"],
    "answer": ["Auto claim fraud indicators include staged accidents..."],
    "contexts": [["Retrieved chunk 1...", "Retrieved chunk 2..."]],
    "ground_truth": ["Ground truth answer from subject matter expert"]  # needed for recall
}

dataset = Dataset.from_dict(eval_data)
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision]
)
print(results)
# Output: {'faithfulness': 0.87, 'answer_relevancy': 0.91, 'context_recall': 0.78, 'context_precision': 0.82}
```

**Production RAGAS pipeline**: Run RAGAS on 2% of production traffic daily, tracking metric trends over time. Set alert thresholds (faithfulness < 0.80 triggers investigation). Use low-faithfulness examples to identify retrieval gaps and update the vector index.

**RAG vs Fine-tuning decision framework**:

| Scenario | Use RAG | Use Fine-tuning | Use Both |
|---------|---------|----------------|---------|
| Frequent knowledge updates | ✅ | ❌ | ✅ |
| Specific format/style required | ❌ | ✅ | ✅ |
| Domain vocabulary adaptation | ❌ | ✅ | ✅ |
| Precise factual recall needed | ✅ | ❌ | ✅ |
| Limited training data | ✅ | ❌ | ❌ |
| Reasoning style change | ❌ | ✅ | — |

My rule of thumb: **RAG for facts, fine-tuning for behavior**. Use RAG when the knowledge changes frequently or is too large for the context window. Use fine-tuning when you need the model to respond in a specific format, tone, or reasoning pattern. For my BERT insurance NLP model, fine-tuning was correct because I needed domain vocabulary recognition (insurance jargon) and specific output formats (NER spans), not just factual retrieval.

---

## SECTION 6: GraphRAG

---

### Q6.1: "What is GraphRAG? How does it differ from standard RAG? How does your Neo4j experience connect?"

**Model Answer**:

Standard RAG retrieves document chunks via vector similarity — great for "what does document X say about Y?" but poor for **relationship queries** like "what entities are connected to fraudster X and what patterns do they share?" GraphRAG addresses this by representing the document corpus as a **knowledge graph** with entities and relationships, enabling graph traversal + semantic retrieval combined.

Microsoft's GraphRAG (2024) specifically focuses on **community-based summarization**: it extracts entities and relationships from documents using an LLM, builds a knowledge graph, applies community detection (Leiden algorithm) to find clusters of closely related entities, and pre-generates hierarchical summaries of each community. This enables two query modes:
- **Global search**: "What are the main themes in this document corpus?" → traverses community summaries top-down
- **Local search**: "What do we know about entity X?" → graph traversal + vector search

**Connection to my Neo4j experience**: In my Pharma Rep Communication System at Axtria, I built exactly this pattern — a Neo4j knowledge graph where nodes were doctors, drugs, therapeutic areas, and publications; edges were relationships like "prescribes," "specializes_in," "authored." The GPT-4 generation layer retrieved structured facts from Neo4j (Cypher queries) + unstructured text from a vector store, combining both for the final communication.

For Huge's fraud detection or marketing use cases, GraphRAG would enable: "What are all the claims connected to this medical provider, and what is the fraud risk pattern for providers with this network structure?" — a graph traversal + community detection question that standard RAG cannot answer.

```cypher
// Neo4j Cypher: Find connected fraud network
MATCH (claimant:Person)-[:FILED]->(claim:Claim)-[:TREATED_BY]->(provider:Provider)
WHERE provider.fraud_score > 0.7
WITH claimant, claim, provider
MATCH (claimant)-[:CONNECTED_TO*1..3]-(related:Person)-[:FILED]->(related_claim:Claim)
RETURN claimant.id, collect(related.id) as network, count(related_claim) as network_claim_count
ORDER BY network_claim_count DESC
```

---

## SECTION 7: Huge-Specific RAG Applications

---

### Q7.1: "Design a multi-tenant RAG system for Huge serving Google, McDonald's, and Nike simultaneously."

**Model Answer**:

The core challenge is **data isolation with shared infrastructure**. Here's my architecture:

**Data Layer**:
- Each client gets a **dedicated collection** in Qdrant (or dedicated Vertex AI Vector Search index)
- All document embeddings carry `client_id`, `project_id`, `classification_level` metadata
- All queries are JWT-authenticated, and the `client_id` from the JWT is **injected as a mandatory filter** — not controllable by the user

**Application Layer**:
- Shared embedding service (cost efficient — same model)
- Client-specific system prompts loaded at query time from a configuration store
- Client-specific re-ranking models (trained on client-specific feedback)
- Client-specific semantic routes (Nike: brand, product, campaign; McDonald's: menu, location, promotion)

**Audit & Compliance Layer**:
- Every query and retrieved document is logged with `{client_id, query_hash, retrieved_doc_ids, response_hash, timestamp}`
- Monthly audit reports showing which documents were accessed for each client
- Ability to purge all data for a specific client (GDPR right to erasure)

```python
def multi_tenant_retrieve(query: str, client_id: str, jwt_token: str) -> list:
    # Validate JWT and extract verified client_id (never trust user-provided)
    verified_client_id = validate_and_extract_jwt(jwt_token)
    assert verified_client_id == client_id, "Client ID mismatch"
    
    # Embed query
    query_embedding = embed_model.encode(query)
    
    # Query with mandatory client filter (cannot be bypassed)
    results = qdrant_client.search(
        collection_name="huge_documents",
        query_vector=query_embedding,
        query_filter=Filter(must=[
            FieldCondition(key="client_id", match=MatchValue(value=verified_client_id))
        ]),
        limit=20
    )
    
    # Re-rank with client-specific model
    reranker = load_client_reranker(verified_client_id)
    return reranker.rerank(query, results, top_k=5)
```

💡 **Key insight**: "The vector database filter is the security perimeter. It must be applied at the infrastructure level, not the application level — you don't want a bug in the routing logic to accidentally cross client data boundaries. I'd use VPC Service Controls on GCP to add an additional network-level guarantee."

---

*End of RAG & Vector Infrastructure Document*

---

> [!TIP]
> When asked about RAG in your interview, always anchor back to the **Chubb fraud detection system**. Mention: hierarchical chunking, hybrid retrieval (FAISS + Elasticsearch BM25 + RRF), cross-encoder re-ranking, and RAGAS evaluation. These are specific enough to prove hands-on experience.
