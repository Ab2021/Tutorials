# Day 39: Advanced RAG Techniques
## Core Concepts & Theory

### Beyond Basic RAG

Basic RAG (retrieve → augment → generate) has limitations:
- **Context Window Constraints:** Can't fit all relevant documents.
- **Retrieval Noise:** Irrelevant documents dilute the context.
- **Multi-Hop Reasoning:** Single retrieval step insufficient for complex queries.
- **Temporal Dynamics:** Knowledge base changes over time.

### 1. Query Transformation

**Query Rewriting:**
- Transform user query into better search queries.
- **Example:** "What's the capital?" → "What is the capital city of France?"

**Query Expansion:**
- Generate multiple variations of the query.
- **Example:** "AI safety" → ["AI safety", "artificial intelligence alignment", "safe AGI"]

**HyDE (Hypothetical Document Embeddings):**
- Generate a hypothetical answer, embed it, use it for retrieval.
- **Intuition:** The answer embedding is closer to relevant documents than the question embedding.

**Implementation:**
```python
def hyde_retrieval(query: str, rag_system):
    # Generate hypothetical answer
    hypothetical_answer = llm.generate(f"Answer this question: {query}")
    
    # Embed and retrieve using the answer
    results = rag_system.retrieve(hypothetical_answer)
    
    return results
```

### 2. Multi-Query RAG

**Concept:** Generate multiple queries, retrieve for each, merge results.

**Process:**
1. **Generate Queries:** "What is X?" → ["What is X?", "Define X", "Explain X"]
2. **Retrieve:** Get top K for each query.
3. **Merge:** Deduplicate and rerank.

**Benefits:**
- Captures different aspects of the question.
- More robust to query formulation.

### 3. Iterative Retrieval (Multi-Hop)

**Problem:** Complex questions require multiple retrieval steps.
**Example:** "Who is the spouse of the author of 'The Great Gatsby'?"
- Step 1: Retrieve "F. Scott Fitzgerald wrote The Great Gatsby"
- Step 2: Retrieve "F. Scott Fitzgerald married Zelda Fitzgerald"

**Self-RAG Algorithm:**
1. **Retrieve:** Get initial documents.
2. **Generate:** Produce partial answer.
3. **Critique:** Is more information needed?
4. **Retrieve Again:** If yes, retrieve based on partial answer.
5. **Repeat:** Until complete.

### 4. Agentic RAG

**Concept:** Agent decides when and what to retrieve.

**ReAct + RAG:**
```
Thought: I need to find who wrote The Great Gatsby
Action: retrieve["author of The Great Gatsby"]
Observation: F. Scott Fitzgerald

Thought: Now I need to find his spouse
Action: retrieve["F. Scott Fitzgerald spouse"]
Observation: Zelda Fitzgerald

Answer: Zelda Fitzgerald
```

### 5. Contextual Compression

**Problem:** Retrieved documents contain irrelevant information.
**Solution:** Extract only relevant sentences/paragraphs.

**Methods:**
- **Extractive:** Use NER or keyword matching to extract relevant sentences.
- **Abstractive:** Use LLM to summarize each document.

**Implementation:**
```python
def compress_context(query: str, documents: List[str]) -> List[str]:
    compressed = []
    for doc in documents:
        prompt = f"Extract only the sentences relevant to: {query}\n\nDocument: {doc}\n\nRelevant sentences:"
        relevant = llm.generate(prompt)
        compressed.append(relevant)
    return compressed
```

### 6. Fusion Retrieval

**Reciprocal Rank Fusion (RRF):**
- Combine rankings from multiple retrieval methods.
- **Formula:** $\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$
- **k:** Constant (typically 60).

**Example:**
- BM25 ranks: [doc1, doc3, doc2]
- Dense ranks: [doc2, doc1, doc3]
- RRF: [doc1, doc2, doc3]

### 7. Parent-Child Chunking

**Concept:** Store small chunks for retrieval, but return larger parent chunks for context.

**Structure:**
- **Child Chunks:** 128 tokens (for precise retrieval).
- **Parent Chunks:** 512 tokens (for rich context).

**Process:**
1. Retrieve child chunks (precise).
2. Return corresponding parent chunks (context-rich).

### 8. Metadata Filtering

**Concept:** Filter documents by metadata before retrieval.

**Metadata Examples:**
- **Date:** Only retrieve documents from last 30 days.
- **Source:** Only retrieve from specific departments.
- **Author:** Only retrieve documents by certain authors.

**Query:**
```python
results = vector_db.search(
    query_embedding,
    filter={"date": {"$gte": "2024-01-01"}, "department": "Engineering"}
)
```

### 9. Reranking Strategies

**Cross-Encoder Reranking:**
- Use a model trained specifically for relevance scoring.
- **Models:** ms-marco-MiniLM, bge-reranker.

**LLM-as-Reranker:**
- Use GPT-4 to score relevance.
- **Prompt:** "Rate the relevance of this document to the query (0-10)."

**Diversity Reranking:**
- Maximize diversity in top K results (avoid redundancy).
- **MMR (Maximal Marginal Relevance):** Balance relevance and diversity.

### 10. RAG with Citations

**Inline Citations:**
```
The company offers 15 days PTO [1]. Remote work is allowed up to 3 days per week [2].

Sources:
[1] Employee Handbook, Section 4.2
[2] Remote Work Policy, Page 3
```

**Implementation:**
- Track which chunk each sentence comes from.
- Add citation markers during generation.

### Real-World Examples

**Perplexity Pro Search:**
- Multi-query RAG with web search.
- Inline citations with source links.

**ChatGPT with Bing:**
- Iterative retrieval (multi-hop).
- Metadata filtering (recent results).

**Notion AI:**
- Parent-child chunking (retrieve sentences, return paragraphs).
- Metadata filtering (by workspace, page, date).

### Summary Table

| Technique | Problem Solved | Complexity | Benefit |
|:----------|:---------------|:-----------|:--------|
| **Query Transformation** | Poor query formulation | Low | +10-15% recall |
| **Multi-Query** | Single query limitation | Medium | +15-20% recall |
| **Iterative Retrieval** | Multi-hop reasoning | High | +20-30% accuracy |
| **Contextual Compression** | Context window overflow | Medium | 2-3x more docs in context |
| **Fusion Retrieval** | Single retrieval method bias | Low | +10% precision |
| **Parent-Child** | Precision vs context trade-off | Medium | +15% both |
| **Reranking** | Initial ranking errors | Medium | +10-20% precision |

### Next Steps
In the Deep Dive, we will implement Self-RAG, Fusion Retrieval, and Parent-Child chunking with complete code examples.
