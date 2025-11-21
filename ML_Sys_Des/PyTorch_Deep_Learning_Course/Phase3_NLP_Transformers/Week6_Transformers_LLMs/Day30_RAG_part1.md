# Day 30: RAG - Deep Dive

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: HNSW, Re-ranking, and Agents

## 1. Vector Search Algorithms

How to find nearest neighbors in 1M vectors?
Brute Force (Flat): $O(N)$. Too slow.
**HNSW (Hierarchical Navigable Small World)**:
*   Graph-based index.
*   Layers of graphs. Top layer has long links (Highways). Bottom layer has short links (Local).
*   $O(\log N)$ search.
*   Standard in FAISS/Pinecone.

## 2. Re-ranking (Cross-Encoders)

Bi-Encoders (Retrievers) are fast but less accurate (dot product).
Cross-Encoders (Re-rankers) are slow but accurate (full attention).
**Pipeline**:
1.  Retrieve Top-100 with Bi-Encoder (Fast).
2.  Re-rank Top-100 with Cross-Encoder (Slow).
3.  Feed Top-5 to LLM.

## 3. Recursive Chunking

Splitting text by fixed characters breaks sentences.
**RecursiveCharacterTextSplitter**:
1.  Try splitting by `\n\n`.
2.  If too big, split by `\n`.
3.  If too big, split by ` `.
*   Preserves semantic structure.

## 4. Agents (ReAct)

**Reason + Act**.
Prompt:
```
Question: What is the weather in NY?
Thought: I need to check weather tool.
Action: WeatherTool(NY)
Observation: 25C.
Thought: I have the answer.
Answer: It is 25C.
```
*   The LLM runs in a loop until it finishes the task.

## 5. Vector Databases

*   **Pinecone**: Managed, scalable.
*   **Chroma**: Local, open-source.
*   **FAISS**: Meta's library, raw index.
*   **Weaviate**: Hybrid search built-in.
