# Day 53: Agentic RAG (Self-Querying & Filter Extraction)
## Core Concepts & Theory

### The Metadata Problem

Standard RAG (Dense Retrieval) relies on semantic similarity.
*   *Query:* "What did we spend on marketing in Q3?"
*   *Vector Search:* Finds chunks discussing "marketing" and "spending".
*   *Failure:* It might retrieve Q1 or Q2 data if they look semantically similar. It ignores the structured constraint "Q3".

**Agentic RAG** solves this by bridging the gap between **Unstructured Search** (Vectors) and **Structured Filtering** (SQL/Metadata).

### 1. Self-Querying Retrievers

A **Self-Querying Retriever** is a component that uses an LLM to parse a natural language query into two parts:
1.  **The Vector Query:** The semantic core ("marketing spend").
2.  **The Metadata Filter:** The structured constraints (`quarter == "Q3"`).

**Process:**
*   *User:* "Show me sci-fi movies from the 90s rated over 8.0."
*   *LLM Parse:*
    *   `query`: "sci-fi movies"
    *   `filter`: `genre == "sci-fi" AND year >= 1990 AND year < 2000 AND rating > 8.0`
*   *Execution:* The Vector DB applies the filter *first* (pre-filtering), then searches vectors within that subset.

### 2. Filter Extraction Techniques

How do we teach the LLM the schema?
*   **Schema Injection:** You must provide the list of valid metadata fields and their types to the LLM.
*   **Cardinality Check:** For categorical fields (e.g., "Genre"), provide the list of valid values. If the list is huge, use a retrieval step to find valid values.

### 3. Auto-Retrieval (LlamaIndex)

LlamaIndex calls this **Auto-Retrieval**. It's not just about filtering; it's about inferring the right parameters for the search engine.
*   *Top-K:* "Give me the top 3..." -> `k=3`.
*   *Alpha:* Balancing Keyword vs Vector search (Hybrid Search).

### 4. Recursive Retrieval

Sometimes the chunk you retrieve isn't the chunk you want to feed the LLM.
*   **Parent Document Retriever:** Index small chunks (sentence level) for precise search. When a hit is found, retrieve the *parent* chunk (paragraph/page level) to give the LLM more context.
*   **Sentence Window Retrieval:** Retrieve the matching sentence, but return the 5 sentences before and after it.

### 5. Document Hierarchies (The Tree)

Agentic RAG often treats data as a tree, not a flat list.
*   **Root:** Summaries of all documents.
*   **Nodes:** Sections/Chapters.
*   **Leaves:** Actual text chunks.
*   **Traversal:** The agent starts at the root ("Which document talks about marketing?"), selects a node, and drills down. This is efficient for massive datasets.

### 6. Corrective RAG (CRAG)

A pattern where the agent evaluates the quality of retrieval.
*   **Ambiguous:** If the retrieval score is low, the agent might decide to use Web Search instead of the Vector DB.
*   **Irrelevant:** If the retrieved docs don't contain the answer, the agent discards them and tries a different query.

### Summary

Agentic RAG moves beyond "embed and pray". It treats the Vector DB as a structured database that can be queried with precision. By extracting filters and traversing hierarchies, we can answer complex questions that require both semantic understanding and structured logic.
