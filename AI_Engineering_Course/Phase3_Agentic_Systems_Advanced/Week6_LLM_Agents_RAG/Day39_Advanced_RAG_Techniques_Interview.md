# Day 39: Advanced RAG Techniques
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is HyDE and when should you use it?

**Answer:**
- **HyDE:** Hypothetical Document Embeddings. Generate a hypothetical answer to the query, embed it, use it for retrieval.
- **Intuition:** Answer embeddings are closer to relevant documents than question embeddings in vector space.
- **When to Use:** When queries are short or vague (e.g., "AI safety" vs "What are the challenges in AI safety?").
- **Trade-off:** Requires an extra LLM call (cost + latency).

#### Q2: Explain Self-RAG and how it differs from basic RAG.

**Answer:**
- **Basic RAG:** Single retrieval step, then generate.
- **Self-RAG:** Iterative. Retrieve → Generate → Reflect (do I need more info?) → Retrieve again → Generate.
- **Benefit:** Handles multi-hop questions (e.g., "Who is the spouse of the author of X?").
- **Challenge:** More expensive (multiple retrieval + generation cycles).

#### Q3: What is Reciprocal Rank Fusion (RRF)?

**Answer:**
- **RRF:** Method to combine rankings from multiple retrieval systems.
- **Formula:** $\text{RRF}(d) = \sum_{r} \frac{1}{k + \text{rank}_r(d)}$ where $k$ is typically 60.
- **Use Case:** Combine BM25 (keyword) + Dense (semantic) retrieval.
- **Benefit:** More robust than single method. Typically +10-15% precision.

#### Q4: What is Parent-Child chunking and why use it?

**Answer:**
- **Concept:** Store small chunks for retrieval precision, but return larger parent chunks for context.
- **Example:** Retrieve sentences (128 tokens), return paragraphs (512 tokens).
- **Benefit:** Best of both worlds—precise retrieval + rich context.
- **Implementation:** Maintain mapping from child chunk ID to parent chunk ID.

#### Q5: How do you add citations to RAG outputs?

**Answer:**
- **Method 1:** Number retrieved documents [1], [2], etc. Prompt LLM to cite sources.
- **Method 2:** Track which chunk each sentence comes from during generation.
- **Validation:** Check if citations actually support the claims (faithfulness).
- **Benefit:** Transparency, verifiability, trust.

---

### Production Challenges

#### Challenge 1: HyDE Generates Poor Hypothetical Answers

**Scenario:** HyDE generates a wrong hypothetical answer, leading to poor retrieval.
**Example:** Query: "Capital of France?" → HyDE: "The capital of France is Berlin" → Retrieves documents about Berlin.
**Solution:**
- **Temperature:** Use temperature=0 for more factual hypothetical answers.
- **Prompt Engineering:** "Generate a factual, accurate answer..."
- **Fallback:** If HyDE retrieval fails, fall back to direct query retrieval.
- **Ensemble:** Combine HyDE retrieval + direct query retrieval using RRF.

#### Challenge 2: Self-RAG Infinite Loops

**Scenario:** Self-RAG keeps saying "NEED_MORE_INFO" and never converges.
**Root Cause:** LLM is overly cautious or the information doesn't exist in the knowledge base.
**Solution:**
- **Max Iterations:** Hard limit (e.g., 3 iterations).
- **Confidence Threshold:** If LLM says "I'm not sure" after 3 iterations, return "I don't know."
- **Reflection Prompt:** "If the information is not available after 3 attempts, admit you don't know."

#### Challenge 3: Parent-Child Chunking Storage Overhead

**Scenario:** You store both parent and child chunks. Storage doubles.
**Analysis:**
- 1M documents → 10M child chunks + 2M parent chunks.
- Embeddings: 10M * 1536 * 4 bytes = 60 GB.
**Solution:**
- **Store Only Child Embeddings:** Embed only child chunks. Store parent chunks as text (no embeddings).
- **On-Demand Parent Retrieval:** Retrieve child chunk IDs, fetch parent chunks from database.
- **Compression:** Use quantized embeddings (int8 instead of float32) to save 75% space.

#### Challenge 4: Multi-Query RAG Cost Explosion

**Scenario:** Generate 5 query variations. Each retrieves top 10 docs. Total: 50 docs. Embedding cost is high.
**Solution:**
- **Fewer Variations:** Generate 2-3 variations instead of 5.
- **Smaller Top-K:** Retrieve top 5 per query instead of top 10.
- **Deduplication:** Deduplicate before embedding (if using re-ranking).
- **Caching:** Cache query variations for common questions.

#### Challenge 5: Citation Hallucinations

**Scenario:** LLM adds citations [1], [2] but the claims are not actually in those sources.
**Root Cause:** LLM hallucinates and adds fake citations to appear credible.
**Solution:**
- **Faithfulness Check:** After generation, verify each cited claim against the source.
  - Extract claim: "The company offers 15 days PTO [1]"
  - Check source [1]: Does it mention "15 days PTO"?
  - If not, flag or remove citation.
- **Prompt Engineering:** "ONLY cite sources if the information is explicitly stated in them."
- **Post-Processing:** Use an LLM to verify citations (LLM-as-a-Judge for faithfulness).

### Summary Checklist for Production
- [ ] **HyDE:** Use **temperature=0** for factual hypothetical answers.
- [ ] **Self-RAG:** Set **max iterations=3** to prevent infinite loops.
- [ ] **Parent-Child:** Store **only child embeddings**, fetch parents on-demand.
- [ ] **Multi-Query:** Generate **2-3 variations**, not 5+.
- [ ] **Citations:** **Verify faithfulness** of citations post-generation.
- [ ] **Fusion:** Use **RRF** to combine BM25 + dense retrieval.
