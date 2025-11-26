# Day 54: Interview Questions & Answers

## Conceptual Questions

### Q1: How do you handle "Re-indexing"?
**Answer:**
*   **Scenario**: User edits a document. The old vectors in Qdrant are now stale.
*   **Strategy**:
    1.  **Delete Old**: `qdrant.delete(filter={doc_id: 1})`.
    2.  **Insert New**: Embed new content and insert.
*   **Optimization**: Only re-embed changed *chunks* (requires tracking chunk hashes).

### Q2: Why use a separate "AI Service" instead of putting this logic in "Doc Service"?
**Answer:**
*   **Resource Isolation**: Embedding/LLM calls are CPU/IO intensive. We don't want to slow down the CRUD API.
*   **Scaling**: We might want 10 instances of `AI Service` to handle heavy indexing, but only 2 instances of `Doc Service`.

### Q3: What is the latency of this RAG pipeline?
**Answer:**
*   **Ingestion**: Async (Kafka). User doesn't wait. Latency ~1-5s.
*   **Query**: Sync (HTTP).
    *   Embedding: 200ms.
    *   Vector Search: 50ms.
    *   LLM Generation: 2-5s (Streaming helps UX).

---

## Scenario-Based Questions

### Q4: The AI is answering questions using *deleted* text. Why?
**Answer:**
*   **Cause**: You forgot to delete the old vectors when the doc was updated.
*   **Fix**: Ensure the Indexer performs a `delete_by_filter(doc_id)` before upserting new chunks.

### Q5: How do you secure the AI Chat?
**Answer:**
*   **Authorization**: Ensure the user actually has access to `doc_id`.
*   **Check**: `AI Service` should call `Auth Service` (or check JWT scopes) to verify `read:doc:1` permission before searching Qdrant.

---

## Behavioral / Role-Specific Questions

### Q6: A user uploads a 1000-page PDF. How do you handle it?
**Answer:**
*   **Async Processing**: Don't block the HTTP request. Return "Processing...".
*   **Batching**: Split into batches of 10 pages. Send to Kafka.
*   **Progress Bar**: Use WebSockets to push progress updates to the UI ("Indexed 50%...").
