# Day 53: Agentic RAG (Self-Querying & Filter Extraction)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between "Pre-filtering" and "Post-filtering" in Vector Search?

**Answer:**
*   **Pre-filtering:** Apply the metadata filter *before* the vector search. The search only happens on the subset of documents.
    *   *Pros:* Accurate. Guaranteed to respect the filter.
    *   *Cons:* If the filter is too restrictive (e.g., `id == 123`), the vector search might be slow or irrelevant if the index isn't optimized for that specific filter pattern (HNSW vs Flat).
*   **Post-filtering:** Perform the vector search on the whole dataset, get top K results, then remove the ones that don't match the filter.
    *   *Pros:* Simple to implement.
    *   *Cons:* **The "K" Problem.** If you retrieve top 10 docs and none of them match the filter, you return 0 results, even if relevant docs exist further down the list.

#### Q2: How does "Self-Querying" handle queries that contain *no* metadata?

**Answer:**
The LLM should be robust enough to output an empty filter.
*   *Query:* "Tell me about space."
*   *Structured Output:* `{"query": "space", "filter": None}`
*   *Execution:* Standard vector search.
*   **Risk:** If the LLM hallucinates a filter (e.g., `genre == "space"` when "space" isn't a valid genre), you get zero results. This requires schema validation.

#### Q3: Explain "Sentence Window Retrieval".

**Answer:**
*   **Indexing:** You split the document into sentences. You embed each sentence individually.
*   **Storage:** You store the sentence *plus* a "window" of 3 sentences before and after it in the metadata (or a separate key-value store).
*   **Retrieval:** You search for the single sentence (high semantic precision).
*   **Generation:** You feed the *window* to the LLM (high context).
*   **Benefit:** Solves the "Context Fragmentation" problem where the answer is split across two chunks.

#### Q4: What is the "Cardinality Problem" in Self-Querying?

**Answer:**
If you have a metadata field `author` with 1 million unique values, you cannot inject the list of all authors into the LLM prompt.
**Solution:**
1.  **Vectorize Metadata:** Create a separate vector index for the authors.
2.  **Two-Step:** The LLM extracts the name "John Smith". You search the "Author Index" to find the canonical ID for "John Smith". You then use that ID in the main search filter.

### Production Challenges

#### Challenge 1: The "Date" Hallucination

**Scenario:** User says "last week". LLM filters `date > 2020-01-01`.
**Root Cause:** The LLM doesn't know what "today" is, or struggles with date math.
**Solution:**
*   **System Prompt Injection:** Always include `Current Date: YYYY-MM-DD` in the prompt.
*   **Tool Use:** Instead of asking the LLM to calculate the date, give it a `calculate_date_range(relative_str)` tool.

#### Challenge 2: Filter-Query Mismatch

**Scenario:**
*   Query: "Comedy movies."
*   LLM Output: `query: "Comedy", filter: genre == "Comedy"`
*   Result: Redundant. The vector search for "Comedy" already finds comedy movies. The filter restricts it further.
*   **Problem:** If the user meant "Funny movies" (which might be tagged "RomCom"), the hard filter `genre == "Comedy"` excludes them.
**Solution:**
*   **Soft Filters:** Prefer vector search for subjective terms ("Funny", "Scary"). Use hard filters only for objective facts ("Year", "Director").

#### Challenge 3: Latency of Two-Step RAG

**Scenario:** Self-Querying adds an extra LLM call (0.5s - 1.0s) before retrieval. Total latency doubles.
**Root Cause:** Sequential dependency.
**Solution:**
*   **Parallelism:** Run a standard vector search *in parallel* with the Self-Query generation. If the Self-Query returns a strict filter, use that result. If it fails/takes too long, fall back to the standard search results.
*   **Fine-tuning:** Fine-tune a small model (Phi-2, Llama-3-8B) specifically for query structuring. It will be faster than GPT-4.

#### Challenge 4: Schema Drift

**Scenario:** You add a new field `language` to your metadata. The LLM doesn't know about it and never filters by language.
**Root Cause:** The prompt is hardcoded.
**Solution:**
*   **Dynamic Prompts:** Fetch the schema from the Vector DB configuration at runtime and inject it into the prompt.

### System Design Scenario: E-Commerce Search Agent

**Requirement:** "Find me a red dress under $50."
**Design:**
1.  **Schema:** `color` (categorical), `category` (categorical), `price` (float).
2.  **Self-Query:**
    *   `query`: "dress"
    *   `filter`: `color == "red" AND category == "dress" AND price < 50`
3.  **Hybrid Search:**
    *   Use **Keyword Search** for "dress" (high precision).
    *   Use **Vector Search** for "red" (visual similarity).
    *   Apply **Hard Filter** for price.
4.  **Ranking:** Boost results that match the user's style profile (Personalization).

### Summary Checklist for Production
*   [ ] **Schema:** Define explicit metadata fields with types.
*   [ ] **Dates:** Inject current date into prompt.
*   [ ] **Cardinality:** Handle high-cardinality fields with lookups.
*   [ ] **Latency:** Monitor the cost of the query-structuring step.
*   [ ] **Fallback:** Always have a fallback if the structured query returns 0 results.
