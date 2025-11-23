# Day 38: RAG (Retrieval-Augmented Generation) Fundamentals
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is RAG and why is it useful?

**Answer:**
- **RAG:** Retrieval-Augmented Generation. Combines information retrieval with LLM generation.
- **Process:** Retrieve relevant documents from a knowledge base, add them to the prompt, generate answer.
- **Benefits:**
  - **Up-to-date Information:** Can access current data (not limited by training cutoff).
  - **Reduced Hallucinations:** Grounds answers in retrieved facts.
  - **Source Attribution:** Can cite specific documents.
  - **Domain-Specific Knowledge:** Works with proprietary or specialized data.

#### Q2: What is the difference between sparse and dense retrieval?

**Answer:**
- **Sparse (BM25):** Keyword-based matching with TF-IDF weighting. Fast, interpretable, good for exact matches.
- **Dense (Embeddings):** Semantic similarity using vector embeddings. Captures meaning, handles synonyms.
- **Example:** Query "car" won't match document with "automobile" in sparse, but will in dense.
- **Hybrid:** Combine both for best results. $\text{Score} = \alpha \cdot \text{BM25} + (1-\alpha) \cdot \text{Cosine}$

#### Q3: How do you choose chunk size for RAG?

**Answer:**
- **Trade-offs:**
  - **Small chunks (128-256 tokens):** Precise retrieval, but may lack context.
  - **Large chunks (512-1024 tokens):** More context, but less precise, may exceed LLM context window.
- **Best Practice:** 256-512 tokens with 50-100 token overlap.
- **Domain-Specific:** For code, chunk by function. For legal docs, chunk by section.

#### Q4: What is reranking and why use it?

**Answer:**
- **Problem:** Initial retrieval (embedding similarity) may not perfectly rank results.
- **Reranking:** Use a cross-encoder to re-score the top N candidates based on query-document relevance.
- **Process:** Retrieve top 20 with bi-encoder (fast), rerank with cross-encoder (slow but accurate), return top 5.
- **Benefit:** 10-20% improvement in retrieval quality.

#### Q5: How do you evaluate RAG system performance?

**Answer:**
- **Retrieval Metrics:**
  - **Recall@K:** % of relevant docs in top K.
  - **MRR:** Mean Reciprocal Rank of first relevant doc.
- **Generation Metrics:**
  - **Faithfulness:** Does answer match retrieved context?
  - **Answer Relevance:** Does answer address the question?
- **End-to-End:**
  - **Accuracy:** % of correct answers.
  - **Hallucination Rate:** % of unsupported claims.

---

### Production Challenges

#### Challenge 1: Retrieval Quality Issues

**Scenario:** RAG retrieves irrelevant documents. Answers are wrong or hallucinated.
**Root Causes:**
- **Poor Chunking:** Chunks are too small/large or split mid-sentence.
- **Embedding Model Mismatch:** Using general embeddings for domain-specific data.
- **Query Formulation:** User query is vague or ambiguous.
**Solution:**
- **Improve Chunking:** Use semantic chunking (split at paragraphs, not mid-sentence).
- **Fine-Tune Embeddings:** Fine-tune embedding model on your domain data.
- **Query Expansion:** Rephrase query or generate multiple variations.
- **Hybrid Search:** Combine BM25 + dense retrieval.

#### Challenge 2: Context Window Overflow

**Scenario:** You retrieve 10 documents (5000 tokens). LLM context window is 4096 tokens. Overflow.
**Solution:**
- **Retrieve Fewer Docs:** Top 3 instead of top 10.
- **Summarize Retrieved Docs:** Use LLM to summarize each document before adding to prompt.
- **Hierarchical Retrieval:** First retrieve sections, then drill down to paragraphs.
- **Larger Context Model:** Use GPT-4-32k or Claude-3 (200k context).

#### Challenge 3: Latency

**Scenario:** RAG takes 5 seconds (2s retrieval + 3s generation). Too slow for production.
**Solution:**
- **Optimize Retrieval:**
  - Use FAISS or Pinecone (fast vector search).
  - Cache frequent queries.
- **Optimize Generation:**
  - Use streaming (show partial results as they're generated).
  - Use smaller LLM for simple queries (GPT-3.5 instead of GPT-4).
- **Parallel Processing:** Retrieve and generate in parallel if possible.

#### Challenge 4: Hallucinations Despite RAG

**Scenario:** RAG still hallucinates. LLM adds information not in the retrieved context.
**Root Cause:** LLM is trained to be helpful and will "fill in gaps" even when instructed not to.
**Solution:**
- **Stronger Prompt:** "ONLY use information from the context. If the answer is not in the context, say 'I don't know.'"
- **Post-Processing:** Check if answer contains claims not in the context. Flag or filter.
- **Faithfulness Classifier:** Train a model to detect hallucinations.
- **Citation Requirement:** Force the LLM to cite sources for every claim.

#### Challenge 5: Embedding Cost

**Scenario:** You have 1M documents. Embedding them costs $20 (OpenAI). Re-embedding on updates is expensive.
**Solution:**
- **Incremental Updates:** Only embed new/changed documents.
- **Cheaper Embeddings:** Use open-source models (Sentence-BERT) instead of OpenAI.
- **Batch Processing:** Embed in large batches to reduce API overhead.
- **Caching:** Store embeddings in a database. Don't recompute.

### Summary Checklist for Production
- [ ] **Chunking:** Use **256-512 tokens** with **50-100 overlap**.
- [ ] **Retrieval:** Use **hybrid search** (BM25 + dense).
- [ ] **Reranking:** Apply **cross-encoder reranking** for top 10 candidates.
- [ ] **Prompt:** Instruct LLM to **cite sources** and **admit uncertainty**.
- [ ] **Evaluation:** Track **Recall@K** and **Hallucination Rate**.
- [ ] **Optimization:** Use **FAISS** for fast search, **cache** frequent queries.
