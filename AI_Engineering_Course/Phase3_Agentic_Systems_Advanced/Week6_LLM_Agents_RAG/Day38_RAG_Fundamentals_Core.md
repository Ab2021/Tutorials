# Day 38: RAG (Retrieval-Augmented Generation) Fundamentals
## Core Concepts & Theory

### The Knowledge Problem

**LLM Limitations:**
- **Static Knowledge:** Training data cutoff (e.g., GPT-4 trained on data up to April 2023).
- **Hallucinations:** Models generate plausible but incorrect information.
- **No Source Attribution:** Cannot cite where information came from.
- **Domain-Specific Knowledge:** Lacks specialized or proprietary information.

**RAG Solution:**
Augment the LLM with external knowledge retrieved at query time.

### 1. RAG Architecture

**Three-Stage Pipeline:**
```
1. Retrieval: Find relevant documents from a knowledge base
2. Augmentation: Add retrieved documents to the prompt
3. Generation: LLM generates response using the augmented context
```

**Example:**
```
User Query: "What is the company's vacation policy?"

1. Retrieval:
   - Search knowledge base for "vacation policy"
   - Find: "Employees get 15 days PTO per year..."

2. Augmentation:
   - Prompt: "Context: Employees get 15 days PTO per year...\nQuestion: What is the company's vacation policy?\nAnswer:"

3. Generation:
   - LLM: "The company provides 15 days of paid time off per year."
```

### 2. Retrieval Methods

**Sparse Retrieval (BM25):**
- **Method:** Keyword matching with TF-IDF weighting.
- **Pros:** Fast, interpretable, works well for exact matches.
- **Cons:** Misses semantic similarity (e.g., "car" vs "automobile").

**Dense Retrieval (Embeddings):**
- **Method:** Encode query and documents as vectors. Find nearest neighbors.
- **Model:** Sentence-BERT, OpenAI Embeddings, Cohere Embed.
- **Pros:** Captures semantic similarity.
- **Cons:** Slower, requires vector database.

**Hybrid Retrieval:**
- Combine sparse + dense.
- **Formula:** $\text{Score} = \alpha \cdot \text{BM25} + (1-\alpha) \cdot \text{Cosine Similarity}$

### 3. Embedding Models

**Sentence-BERT (SBERT):**
- Fine-tuned BERT for sentence embeddings.
- **Dimension:** 384 or 768.
- **Use Case:** General-purpose semantic search.

**OpenAI Embeddings (text-embedding-3-small/large):**
- **Dimension:** 1536 (small), 3072 (large).
- **Cost:** $0.02 per 1M tokens (small).
- **Performance:** State-of-the-art on MTEB benchmark.

**Cohere Embed:**
- **Dimension:** 1024 or 4096.
- **Multilingual:** Supports 100+ languages.

**Domain-Specific:**
- Fine-tune on your domain data for better performance.

### 4. Vector Similarity Metrics

**Cosine Similarity:**
$$ \text{sim}(A, B) = \frac{A \cdot B}{||A|| \cdot ||B||} $$
- Range: [-1, 1]. Higher = more similar.
- **Use Case:** Most common for text embeddings.

**Euclidean Distance:**
$$ d(A, B) = ||A - B|| = \sqrt{\sum (a_i - b_i)^2} $$
- Range: [0, ∞]. Lower = more similar.

**Dot Product:**
$$ \text{sim}(A, B) = A \cdot B = \sum a_i b_i $$
- Faster than cosine (no normalization).
- **Use Case:** When embeddings are already normalized.

### 5. Chunking Strategies

**Fixed-Size Chunking:**
- Split documents into fixed-length chunks (e.g., 512 tokens).
- **Pros:** Simple, consistent.
- **Cons:** May split mid-sentence or mid-paragraph.

**Semantic Chunking:**
- Split at natural boundaries (paragraphs, sections).
- **Pros:** Preserves context.
- **Cons:** Variable chunk sizes.

**Overlapping Chunks:**
- Add overlap between chunks (e.g., 50 tokens).
- **Pros:** Reduces information loss at boundaries.
- **Cons:** Redundancy, more storage.

**Hierarchical Chunking:**
- Create chunks at multiple levels (sentence, paragraph, section).
- **Pros:** Flexible retrieval granularity.
- **Cons:** Complex indexing.

### 6. RAG Prompt Template

**Basic Template:**
```
Context:
{retrieved_documents}

Question: {user_query}

Answer based on the context above. If the answer is not in the context, say "I don't know."

Answer:
```

**Advanced Template:**
```
You are a helpful assistant. Answer the question based on the provided context.

Context:
{retrieved_documents}

Question: {user_query}

Instructions:
- Use only information from the context
- Cite sources using [1], [2], etc.
- If the answer is not in the context, say "I don't have enough information to answer this question."

Answer:
```

### 7. Evaluation Metrics

**Retrieval Quality:**
- **Recall@K:** % of relevant documents in top K results.
- **Precision@K:** % of top K results that are relevant.
- **MRR (Mean Reciprocal Rank):** Average of 1/rank of first relevant result.

**Generation Quality:**
- **Faithfulness:** Does the answer match the retrieved context?
- **Answer Relevance:** Does the answer address the question?
- **Context Relevance:** Are the retrieved documents relevant?

**End-to-End:**
- **Accuracy:** % of correct answers.
- **Hallucination Rate:** % of answers with unsupported claims.

### 8. RAG vs. Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|:-------|:----|:------------|
| **Knowledge Update** | Real-time | Requires retraining |
| **Source Attribution** | Yes (cites documents) | No |
| **Cost** | Low (inference only) | High (training) |
| **Latency** | Higher (retrieval + generation) | Lower (generation only) |
| **Use Case** | Dynamic knowledge, Q&A | Task-specific behavior |

### Real-World Examples

**Perplexity.ai:**
- Search engine powered by RAG.
- Retrieves web pages, generates answers with citations.

**ChatGPT with Browsing:**
- Retrieves current web content.
- Answers questions about recent events.

**Notion AI:**
- RAG over user's Notion workspace.
- Answers questions about personal notes and documents.

**GitHub Copilot Chat:**
- RAG over codebase and documentation.
- Answers coding questions with context.

### Summary

**RAG Benefits:**
- Up-to-date information.
- Reduced hallucinations.
- Source attribution.
- Domain-specific knowledge.

**RAG Challenges:**
- Retrieval quality (garbage in, garbage out).
- Latency (retrieval + generation).
- Context window limits.
- Cost (embeddings + vector DB).

### Next Steps
In the Deep Dive, we will implement a complete RAG system with chunking, embedding, retrieval, and generation.
