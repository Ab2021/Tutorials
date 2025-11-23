# Day 83: Capstone Project Phase 2 - Implementation (MVP)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How do you handle "Garbage In, Garbage Out" in RAG?

**Answer:**
- **Parsing:** The quality of the RAG system is 80% determined by the parser.
- **Cleaning:** Remove headers, footers, and boilerplate text (legal disclaimers) that dilute the vector search.
- **Chunking:** Ensure chunks respect sentence/paragraph boundaries (Semantic Chunking) rather than arbitrary character counts.

#### Q2: Why use `uuid` for vector IDs instead of integers?

**Answer:**
- **Distributed Systems:** UUIDs can be generated anywhere without coordination (no central auto-increment counter).
- **Security:** Harder to guess.
- **Uniqueness:** Guaranteed unique across shards.

#### Q3: What is the difference between `upsert` and `update` in Vector DBs?

**Answer:**
- **Upsert:** Insert if not exists, Update if exists. Idempotent. Best for ingestion pipelines (safe to re-run).
- **Update:** Modifies existing record. Fails if not found.

#### Q4: How do you debug a RAG system that gives wrong answers?

**Answer:**
- **Inspect Retrieval:** Look at the top 3 chunks retrieved. Do they contain the answer?
- **If No:** It's a Retrieval problem (Bad embedding, bad chunking).
- **If Yes:** It's a Generation problem (Model hallucination, bad prompt).

#### Q5: Why Dockerize the application?

**Answer:**
- **Reproducibility:** "Works on my machine" -> "Works everywhere".
- **Deployment:** K8s/ECS expects containers.
- **Dependencies:** Encapsulates Python libs, system dependencies (poppler-utils for PDF), and env vars.

---

### Production Challenges

#### Challenge 1: PDF Parsing is Slow

**Scenario:** Parsing a 50-page PDF takes 2 minutes. API times out.
**Root Cause:** OCR/Parsing is CPU intensive.
**Solution:**
- **Async:** Return "202 Accepted" immediately. Process in background.
- **Webhooks:** Notify client when processing is done.

#### Challenge 2: Context Window Exceeded

**Scenario:** Retrieved 10 chunks. Total tokens = 10,000. Model limit = 8,000.
**Root Cause:** Too many chunks or chunks too large.
**Solution:**
- **Token Counting:** Count tokens before sending. Truncate if necessary.
- **Re-ranking:** Pick top 5 instead of top 10.

#### Challenge 3: "I don't know"

**Scenario:** User asks a question not in the docs. Model hallucinates an answer.
**Root Cause:** Model wants to be helpful.
**Solution:**
- **System Prompt:** "If the answer is not in the context, say 'I don't know'. Do not make things up."
- **Confidence Score:** If retrieval scores are low (< 0.7), don't even call the LLM.

#### Challenge 4: Dirty Text (Ligatures)

**Scenario:** "fi" becomes a single character. "Office" -> "Of ice". Search fails.
**Root Cause:** PDF encoding.
**Solution:**
- **Normalization:** Use `ftfy` (Fix Text For You) library to clean unicode issues.

#### Challenge 5: Environment Variables

**Scenario:** Hardcoding API keys in `main.py`. Committed to Git. Revoked.
**Root Cause:** Bad security practice.
**Solution:**
- **.env:** Load from `.env` file locally.
- **Secrets Manager:** Inject as env vars in Docker/Production.

### System Design Scenario: Scaling the MVP

**Requirement:** Support 100 concurrent users.
**Design:**
1.  **API:** Run 5 replicas of FastAPI behind Nginx.
2.  **DB:** Qdrant Cloud (Managed) or Cluster mode.
3.  **Ingestion:** Separate worker fleet (Celery/Redis) to handle uploads without blocking chat.
4.  **Cache:** Redis to cache answers for identical queries.

### Summary Checklist for Production
- [ ] **Parsing:** Use **Unstructured** or **LlamaParse**.
- [ ] **Chunking:** Use **RecursiveCharacterTextSplitter**.
- [ ] **Prompt:** Enforce **"I don't know"**.
- [ ] **Infrastructure:** Use **Docker Compose**.
- [ ] **Async:** Move ingestion to **Background Tasks**.
