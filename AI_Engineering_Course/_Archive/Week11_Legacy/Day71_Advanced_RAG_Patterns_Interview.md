# Day 71: Advanced RAG Patterns
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the "Lost in the Middle" phenomenon?

**Answer:**
- **Observation:** LLMs are good at using information at the *beginning* and *end* of the context window, but often ignore information in the *middle*.
- **Implication for RAG:** If you retrieve 20 documents and the answer is in document #10, the model might miss it.
- **Solution:** Re-ranking. Place the most relevant documents at the start and end of the context (U-shaped distribution).

#### Q2: How does GraphRAG improve upon Vector RAG?

**Answer:**
- **Vector RAG:** Finds semantically similar chunks. Good for "What is X?". Bad for "How are X and Y related?" if they don't appear in the same chunk.
- **GraphRAG:** Explicitly models relationships. Can traverse edges to find connections across documents. Good for multi-hop reasoning and global summarization.

#### Q3: Explain the concept of "Re-ranking" in RAG.

**Answer:**
- **Two-Stage Retrieval:**
  1.  **Bi-Encoder (Vector Search):** Fast, retrieves top 100 candidates. Approximate.
  2.  **Cross-Encoder (Re-ranker):** Slow, scores (Query, Doc) pairs accurately. Re-orders top 100 to find top 5.
- **Benefit:** Drastically improves precision. Allows using a cheaper vector search while maintaining high quality.

#### Q4: What is HyDE and when does it fail?

**Answer:**
- **HyDE:** Generates a hypothetical answer to search for.
- **Success:** Good when the query is short/ambiguous but the answer has a distinct structure.
- **Failure:** If the LLM hallucinates a *wrong* hypothetical answer (e.g., wrong facts), the retrieval will find documents matching the hallucination, reinforcing the error (Confirmation Bias).

#### Q5: Why use a Parent Document Retriever?

**Answer:**
- **Problem:** Small chunks are good for matching (precise vectors) but bad for generation (lack context). Large chunks are bad for matching (diluted vectors) but good for generation.
- **Solution:** Decouple indexing from retrieval. Index small chunks, but return their parent (large chunk) to the LLM. Best of both worlds.

---

### Production Challenges

#### Challenge 1: Retrieval Latency

**Scenario:** RAG Fusion + Re-ranking takes 3 seconds. User leaves.
**Root Cause:** Multiple LLM calls and heavy Cross-Encoder.
**Solution:**
- **Parallelism:** Run RAG Fusion queries in parallel.
- **Faster Re-ranker:** Use a smaller Cross-Encoder (MiniLM) or ColBERT (Late Interaction).
- **Hybrid Search:** Use sparse (BM25) + dense without re-ranking for simple queries.

#### Challenge 2: Hallucination despite RAG

**Scenario:** Model ignores the retrieved context and answers from its pre-training memory (hallucinating).
**Root Cause:** "Strong Prior". The model thinks it knows better.
**Solution:**
- **Prompt Engineering:** "Answer ONLY using the provided context."
- **Self-RAG:** Train the model to output `<IsSup>` (Supported) token.
- **Logit Bias:** Suppress tokens not present in the context (extreme).

#### Challenge 3: Conflicting Information

**Scenario:** Doc A says "Revenue: $1M". Doc B says "Revenue: $2M". Model outputs mixed/wrong answer.
**Root Cause:** RAG retrieves contradictory chunks.
**Solution:**
- **Metadata Filtering:** Prioritize recent documents (Date filter).
- **Source Citing:** Ask model to cite sources. "Doc A says X, but Doc B says Y".
- **Multi-View:** Present both viewpoints.

#### Challenge 4: Question Decomposition

**Scenario:** User asks "Compare the climate policy of US and China". RAG retrieves generic policy docs but misses the comparison points.
**Root Cause:** Complex query needs breaking down.
**Solution:**
- **Decomposition:** Break into "What is US climate policy?" and "What is China climate policy?". Retrieve for both. Combine.

#### Challenge 5: The "Empty Search" Problem

**Scenario:** User asks about a topic not in the DB. Vector search returns nearest (irrelevant) garbage. Model tries to answer using garbage.
**Root Cause:** Vector search always returns *something*.
**Solution:**
- **Thresholding:** If similarity score < 0.7, return "I don't know".
- **CRAG:** If scores are low, trigger Web Search.

### System Design Scenario: Legal Case Research Assistant

**Requirement:** Find precedents for a specific contract clause.
**Design:**
1.  **Chunking:** Semantic chunking (by clause/paragraph).
2.  **Enrichment:** Add metadata (Case Year, Judge, Jurisdiction).
3.  **Retrieval:** Hybrid (Keywords for "Section 409A" + Vector for semantic meaning).
4.  **Pattern:** Parent Document Retriever (Show full clause context).
5.  **Verification:** Self-RAG to ensure citation accuracy.

### Summary Checklist for Production
- [ ] **Re-ranking:** Always use a **Cross-Encoder**.
- [ ] **Hybrid:** Combine **Vector + Keyword** search.
- [ ] **Query:** Use **Query Expansion/Fusion**.
- [ ] **Context:** Use **Parent Document Retriever**.
- [ ] **Fallback:** Implement **Thresholds** for "I don't know".
