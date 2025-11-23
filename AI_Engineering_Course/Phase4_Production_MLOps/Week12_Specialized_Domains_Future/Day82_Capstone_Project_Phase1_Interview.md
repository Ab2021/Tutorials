# Day 82: Capstone Project Phase 1 - Planning & Architecture
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How do you estimate the cost of a RAG system?

**Answer:**
- **Storage:** Vector DB cost (RAM/Disk). 1M vectors (1536 dim) ≈ 4GB RAM.
- **Ingestion:** Embedding cost. 1M pages ≈ 500M tokens ≈ $10 (OpenAI).
- **Inference:** LLM cost. 1 query = 1k input tokens + 500 output tokens ≈ $0.02.
- **Traffic:** 1000 queries/day = $20/day.

#### Q2: Why separate Ingestion from the API?

**Answer:**
- **Blocking:** Parsing a 100-page PDF takes 30 seconds. You don't want to block the API thread.
- **Scaling:** You might need 10 workers for ingestion but only 2 for chat. Decoupling allows independent scaling.
- **Reliability:** If ingestion crashes (bad PDF), it shouldn't take down the chat API.

#### Q3: What is "Metadata Filtering" and why is it crucial for Enterprise RAG?

**Answer:**
- **Security:** "Show me Q3 revenue" -> Must filter by `user_access_list`.
- **Precision:** "Show me Q3 revenue for Apple" -> Filter by `doc_type="10-K"` and `ticker="AAPL"`.
- **Performance:** Filtering reduces the search space before vector similarity.

#### Q4: How do you handle PDF tables?

**Answer:**
- **Naive:** Text extraction destroys table structure.
- **Better:** Use a VLM (GPT-4o) or specialized model (Table Transformer) to convert table image to Markdown/HTML.
- **Indexing:** Index the Markdown representation.

#### Q5: Explain the "Hybrid Search" strategy.

**Answer:**
- **Vector Search:** Good for semantic meaning ("financial performance").
- **Keyword Search (BM25):** Good for exact matches ("Project X-15").
- **Hybrid:** Combine scores (Reciprocal Rank Fusion). Essential for technical domains with specific acronyms.

---

### Production Challenges

#### Challenge 1: The "100MB PDF"

**Scenario:** User uploads a massive scanned PDF. Parser OOMs (Out of Memory).
**Root Cause:** Loading entire file into RAM.
**Solution:**
- **Streaming:** Process page by page.
- **Chunking:** Split PDF into 10-page chunks before processing.

#### Challenge 2: Duplicate Documents

**Scenario:** User uploads "Report_v1.pdf" and "Report_v1_final.pdf". Search returns duplicate chunks.
**Root Cause:** No deduplication.
**Solution:**
- **Content Hash:** Calculate MD5 of file content. If exists, reject upload or point to existing ID.

#### Challenge 3: Latency Spikes

**Scenario:** P99 latency hits 10 seconds.
**Root Cause:** Re-ranker model is too slow on CPU.
**Solution:**
- **GPU:** Move Re-ranker to GPU.
- **Quantization:** Use ONNX runtime for Re-ranker.
- **Parallel:** Run retrieval and re-ranking in parallel threads.

#### Challenge 4: Versioning

**Scenario:** You change the embedding model. Old vectors are now incompatible.
**Root Cause:** Model drift.
**Solution:**
- **Re-indexing:** You must re-embed all documents.
- **Collection Versioning:** Create `collection_v2`, backfill, then switch alias `prod` to `v2`.

#### Challenge 5: Rate Limiting

**Scenario:** One user uploads 1000 files, clogging the queue.
**Root Cause:** No fairness.
**Solution:**
- **Per-User Queue:** Round-robin processing of user jobs.
- **Quotas:** Max 50MB per day per user.

### System Design Scenario: Designing the Ingestion Pipeline

**Requirement:** Process 10k PDFs/day.
**Design:**
1.  **Upload:** S3 Presigned URL (Direct upload from browser).
2.  **Trigger:** S3 Event Notification -> SQS Queue.
3.  **Compute:** Lambda (for small files) or Fargate (for large OCR jobs).
4.  **Step Function:** Orchestrate Parse -> Chunk -> Embed -> Store.
5.  **Error Handling:** Dead Letter Queue (DLQ) for failed PDFs.

### Summary Checklist for Production
- [ ] **Async:** Use **Background Workers** for ingestion.
- [ ] **Security:** Implement **RBAC** at the API level.
- [ ] **Storage:** Use **S3** for raw files, **Vector DB** for chunks.
- [ ] **Hybrid:** Enable **Keyword + Vector** search.
- [ ] **Observability:** Trace every request ID.
