# RAG Architecture – Complete Deep Dive
## AIP-C01 – Domain 1 (31%) Core Competency

---

## 🏛️ What is RAG?

**Retrieval-Augmented Generation (RAG)** is an architectural pattern that enhances FM responses by:
1. **Retrieving** relevant documents from an external knowledge base at query time
2. **Augmenting** the prompt with the retrieved context
3. **Generating** a response grounded in that context

```
User Query
    ↓
[Embedding Model] → Query Vector
    ↓
[Vector Store] → ANN Search → Top-K Chunks
    ↓
[Prompt Augmentation] = System Prompt + Context Chunks + User Query
    ↓
[Foundation Model] → Response (Grounded in Context)
    ↓
[Optional: Guardrails Check] → Safe, Grounded Response
```

**Why RAG instead of Fine-Tuning?**

| Criterion | RAG | Fine-Tuning |
|-----------|-----|-------------|
| Data freshness | ✅ Real-time updates | ❌ Requires retraining |
| Source attribution | ✅ Citations available | ❌ Model "memorizes" |
| Cost | ✅ Lower | ❌ High compute cost |
| Confidential data | ✅ Stays in knowledge base | ❌ Embedded in model |
| Style/format adaptation | ❌ Limited | ✅ Excellent |

---

## 📦 Chunking Strategies

Chunking splits documents into manageable pieces for embedding. **The choice of chunking strategy directly impacts retrieval quality.**

### Strategy Comparison (Critical for Exam)

| Strategy | Description | Chunk Size | Best For |
|----------|-------------|-----------|---------|
| **Default/Fixed-Size** | Uniform token count splits | ~300 tokens | General purpose; simple docs |
| **Fixed-Size with Overlap** | Uniform splits + sliding window | 300 tokens + 10-20% overlap | Prevents boundary information loss |
| **Semantic** | Split by topic/meaning coherence | Variable | Narrative, complex documents |
| **Hierarchical** | Parent-child: small child for search, large parent for context | Child: 100-300 tokens, Parent: 500-1000 tokens | Long structured documents |
| **No Chunking** | Entire document as one chunk | Entire doc | Very short, atomic documents |
| **Custom (Lambda)** | Lambda function defines boundaries | Any | Domain-specific structure |

### Hierarchical Chunking – Exam Favorite ⭐

```
Document
  ├── Parent Chunk 1 (500 tokens) → Provides context to model
  │     ├── Child Chunk 1a (100 tokens) → Used for retrieval
  │     └── Child Chunk 1b (100 tokens) → Used for retrieval
  └── Parent Chunk 2 (500 tokens)
        ├── Child Chunk 2a (100 tokens)
        └── Child Chunk 2b (100 tokens)

Query → Match child chunks → Return parent chunk to LLM
```

**Why Hierarchical?**
- Small child chunks → **Precise retrieval** (surgical accuracy)
- Large parent chunks → **Full context** passed to model
- Best of both worlds: precision + comprehensiveness

> **Exam Question Pattern:** "A document retrieval system returns correct documents but the model lacks sufficient surrounding context" → **Hierarchical Chunking**

---

### What to Avoid: Larger Chunks ≠ Cost Optimization

> ❌ **Exam Trap:** "Increase chunk size to reduce embedding costs"  
> ✅ **Correct:** Larger chunks cause **semantic dilution** → worse retrieval quality

**Reasoning:**
- Larger chunks = fewer total embeddings → fewer API calls → lower cost
- BUT: More surrounding text dilutes specific semantic meaning
- Better cost optimization: **Reduce embedding dimensionality** (e.g., 256 or 384 for Titan)

---

## 🔢 Embedding Models & Dimensionality

### Amazon Titan Embeddings V2

Supports multiple output dimensions:
- `256` – Lowest cost, fastest; acceptable for simple use cases
- `384` – Balanced performance
- `512` – Good precision
- `1024` – Maximum precision (default)

**Cost Optimization Strategy:**
```
Choose lowest dimensionality that still meets domain accuracy requirements
↓
Test at 384 or 512 first
↓
Only use 1024 if precision critically required
```

### Embedding at Scale: Dimensionality Trade-offs

| Dimension | Storage/Vector | Query Speed | Precision |
|-----------|---------------|-------------|-----------|
| 256 | 1 KB | Fastest | Lower |
| 512 | 2 KB | Fast | Medium |
| 1024 | 4 KB | Moderate | Highest |

> **Exam Pattern:** "10 million embeddings causing high storage and query latency" → **Reduce dimensionality, NOT remove data or switch models**

---

## 🗃️ Vector Stores for Bedrock

### Supported Vector Stores

| Store | Type | Best For | Exam Notes |
|-------|------|---------|-----------|
| **Amazon OpenSearch Serverless** | Managed serverless | 10M+ embeddings; real-time; metadata filtering | ✅ Default choice; low operational overhead |
| **Aurora PostgreSQL + pgvector** | Managed relational DB | When relational data + vectors needed together | SQL for hybrid queries |
| **Amazon S3 Vectors** | Object-based | Cost-effective; limited filtering | New service; note filterable vs non-filterable |
| **Amazon Neptune Analytics** | Graph + vector | When graph relationships matter | Graph-based RAG |
| **Pinecone** | Third-party | High-performance vector search | Available via Bedrock integration |
| **Redis Enterprise** | In-memory | Ultra-low latency | Available via Bedrock integration |
| **MongoDB Atlas** | Document + vector | Document-heavy workloads | Available via Bedrock integration |

### Amazon OpenSearch Serverless – Deep Dive ⭐

**HNSW Index (Hierarchical Navigable Small World)**
- Default indexing algorithm for vector search
- Approximate Nearest Neighbor (ANN) search
- Trades slight precision loss for massive speed gains

**Shard Optimization:**
```
Problem: Millions of small shards → High coordination overhead → Slow queries
Solution: Consolidate to fewer, larger shards (target: 30-50 GB per shard)

Large shard benefits:
- Better cache locality for HNSW graph traversal
- Reduced cross-shard coordination overhead
- Lower query latency at scale
```

> **Exam Pattern:** "OpenSearch index suffers high latency with millions of small shards" → **Consolidate to fewer, larger shards (30-50 GB)**

---

## 🔍 Retrieval Strategies

### 1. Semantic (Vector) Search

```
Query → Embedding → Cosine Similarity → Top-K Results
```

- Great for conceptual, natural-language queries
- Misses exact term/acronym matches

### 2. Keyword (BM25/Lexical) Search

```
Query → Tokenize → TF-IDF/BM25 Scoring → Top-K Results
```

- Great for exact medical terms, acronyms, codes
- Misses synonyms and paraphrases

### 3. Hybrid Search ⭐ (Exam Favorite)

```
Vector Score + Keyword Score → Combined Ranking → Top-K Results
```

**When to use hybrid search:**
- Medical terminology (exact acronym match + semantic understanding)
- Legal documents (exact clause references + contextual search)
- Product catalogs (exact SKU + description-based search)

> **Exam Pattern:** "RAG misses exact medical acronyms but returns semantically similar docs" → **Configure Hybrid Search in OpenSearch (vector + BM25)**

### 4. Metadata Filtering

```python
filter = {
    "andAll": [
        {"equals": {"key": "department", "value": "finance"}},
        {"greaterThan": {"key": "version", "value": "2.0"}},
        {"notEquals": {"key": "status", "value": "deprecated"}}
    ]
}
```

**Use cases:**
- Exclude outdated documents (filter by date)
- Enforce access control (filter by department)
- Version control (filter by document version)

> **Exam Pattern:** "RAG returns semantically correct but outdated documents" → **Verify metadata filtering logic**

---

## 📄 Document Pre-Processing

### Amazon Bedrock Data Automation (BDA)

**Purpose:** Convert complex unstructured documents (PDFs, images, audio, video) into structured JSON/HTML before ingestion.

```
Invoice PDF → BDA → {"vendor": "Acme", "total": 1250.00, "line_items": [...]}
Medical Report PDF → BDA → {"patient_id": "...", "diagnosis": "...", "treatments": [...]}
```

**Why BDA over standard OCR?**
- Understands document **layout and structure** (tables, forms, headers)
- Preserves relationships between elements
- Outputs structured JSON ready for downstream processing

> **Exam Pattern:** "Prepare complex PDFs and images for foundation model ingestion with semantic understanding" → **Amazon Bedrock Data Automation** (NOT Amazon Textract for semantic layout understanding)

### Amazon Textract vs BDA

| Aspect | Amazon Textract | Bedrock Data Automation |
|--------|----------------|------------------------|
| Focus | OCR + Forms/Tables | Semantic document understanding |
| Output | Raw text, key-value pairs | Structured JSON with semantic context |
| Use in Bedrock | Data preprocessing | Bedrock Knowledge Base integration |
| Exam use | Invoice structured extraction (forms/tables) | Complex document → Knowledge Base |

---

## 📊 RAG Evaluation Metrics

### Retrieval Metrics

| Metric | Measures | Formula |
|--------|---------|---------|
| **Context Precision** | % of retrieved chunks that are relevant | Relevant chunks / Total retrieved |
| **Context Recall** | % of relevant chunks actually retrieved | Retrieved relevant / Total relevant |
| **Citation Precision** | % of citations that are relevant to the answer | Relevant citations / Total citations |

### Generation Metrics

| Metric | Measures | What It Tests |
|--------|---------|--------------|
| **Faithfulness** | Is the answer grounded in context? | No hallucinations |
| **Answer Relevance** | Does the answer address the question? | On-topic responses |
| **Correctness** | Is the answer factually accurate? | Ground truth comparison |
| **Completeness** | Does the answer cover all key points? | Coverage check |

### Bedrock Model Evaluation for RAG

```python
response = bedrock.create_evaluation_job(
    jobName="rag-evaluation-001",
    roleArn="arn:aws:iam::...",
    evaluatorModelConfig={
        "bedrockEvaluatorModels": [{"modelIdentifier": "amazon.nova-pro-v1:0"}]
    },
    inferenceConfig={
        "ragConfigs": [{
            "knowledgeBaseConfig": {
                "retrieveAndGenerateConfig": {
                    "knowledgeBaseId": "KB123",
                    "modelArn": "arn:aws:bedrock:..."
                }
            }
        }]
    },
    outputDataConfig={"s3Uri": "s3://eval-results/"},
    evaluationConfig={
        "automated": {
            "datasetMetricConfigs": [{
                "taskType": "QuestionAndAnswer",
                "dataset": {"name": "test-dataset", "datasetLocation": {"s3Uri": "s3://test-data/"}},
                "metricNames": ["Faithfulness", "Completeness", "Helpfulness"]
            }]
        }
    }
)
```

---

## 🔁 RAG Freshness & Consistency

### Problem: Outdated Documents in Retrieval

**Symptoms:** Model returns factually outdated answers that were correct historically

**Root Cause Diagnosis:**
1. Documents NOT tagged with metadata (e.g., no `effective_date` field)
2. Metadata filters NOT configured in retrieval query
3. Filters configured but not applied correctly (logic error)

**Fix:** Verify metadata filtering is applied at query time

```python
# Correct: Filter applied at retrieval time
filter = {"greaterThanOrEquals": {"key": "effective_date", "value": "2025-01-01"}}
```

### Problem: Inconsistent Answers After Content Updates

**Root Cause:** LLMs are non-deterministic; new documents change retrieval results

**Fix:** Regression testing with fixed ground-truth dataset

```
After every ingestion job:
1. Run fixed test question set
2. Compare against expected answers using Bedrock Model Evaluations
3. Fail deployment if quality metrics regress
```

> **Exam Trap:** "Use response hashing to detect inconsistencies" → ❌ LLMs are non-deterministic; hashes always differ even for correct answers

---

## 🔒 Row-Level Security in RAG

### Problem: Users should only see documents they're authorized to access

**Wrong Approach:** S3 bucket policies (they control access to files, NOT to retrieved content)

**Correct Approach:**

```
Step 1: Tag documents with access metadata
         {"clearance_level": "confidential", "department": "finance"}

Step 2: Store user's permissions in session/JWT
         {"user_id": "u123", "clearance": "internal", "department": "finance"}

Step 3: Apply metadata filters at query time based on user's permissions
         filter = {"equals": {"key": "clearance_level", "value": user.clearance}}
```

**Key Insight:** Security is enforced at **retrieval time**, not storage time. This allows a **single Knowledge Base** to serve multiple security tiers without re-indexing.

---

## 🏗️ Enterprise RAG Architecture Pattern

```
                         Users
                           │
                    [API Gateway]
                           │
                    [Lambda Function]
                    ┌──────┤
                    │      │ 1. Validate user permissions
                    │      │ 2. Build metadata filter from user context
                    │      │ 3. Call Bedrock RetrieveAndGenerate with filter
                    │      │ 4. Return grounded response
                    └──────┘
                           │
             ┌─────────────▼──────────────┐
             │   Bedrock Knowledge Base   │
             │                            │
             │  [Metadata Filtering]      │
             │  department=finance AND    │
             │  version>=2.0 AND          │
             │  effective_date>=2025-01   │
             │                            │
             │  [Vector Store]            │
             │  OpenSearch Serverless     │
             └────────────────────────────┘
```

---

## 📝 Practice Questions (RAG)

**Q1:** A medical RAG application correctly retrieves semantically similar documents but frequently misses exact medical acronyms like "COPD" or "MI". What should the developer do?

- A. Switch to a larger embedding model with more dimensions  
- B. Configure hybrid search combining vector similarity with keyword BM25 matching  
- C. Implement post-retrieval Lambda to filter by acronym presence  
- D. Create a separate knowledge base for acronyms only  

**Answer: B** – Hybrid search combines semantic (handles synonyms) with keyword (handles exact terms/acronyms). Least operational overhead.

---

**Q2:** A company's RAG system is returning correct answers but citing irrelevant documents. Which evaluation approach should they use?

- A. Check ROUGE scores against ground truth  
- B. Analyze citation precision and context relevance from Bedrock RAG evaluation jobs  
- C. Compare embedding cosine similarity of retrieved chunks  
- D. Monitor CloudWatch latency metrics  

**Answer: B** – Citation precision and context relevance specifically measure the link between retrieved documents and the final answer. ROUGE measures text overlap, not citation accuracy.

---

**Q3:** A company ingests documents into a Bedrock Knowledge Base from S3. New documents should be indexed automatically within minutes of upload. What is the MOST efficient approach?

- A. Use Amazon EventBridge to schedule ingestion jobs every 5 minutes  
- B. Use AWS Glue to scan S3 on a schedule and trigger ingestion  
- C. Configure S3 Event Notifications to trigger a Lambda function that calls StartIngestionJob  
- D. Manually trigger ingestion from the Bedrock console  

**Answer: C** – Event-driven: S3 notification → Lambda → StartIngestionJob is native, near real-time, and requires minimal custom code.

---

*Next: [04_bedrock_agents.md](./04_bedrock_agents.md)*
