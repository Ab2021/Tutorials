# Day 52: RAG with Tools (Query Decomposition & Routing)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the "Lost in the Middle" phenomenon and how does Query Decomposition help?

**Answer:**
*   **Phenomenon:** LLMs tend to pay more attention to the beginning and end of the context window, ignoring information in the middle. If you retrieve 20 documents and the answer is in document #10, the model might miss it.
*   **Decomposition:** By breaking a complex query into 3 simple sub-queries, you retrieve fewer, more targeted documents for each sub-query. You then synthesize the answer incrementally. This keeps the context focused and reduces the "middle" noise.

#### Q2: When would you use a "Semantic Router" (Embedding-based) vs an "LLM Router" (Prompt-based)?

**Answer:**
*   **Semantic Router:** Use for high-volume, low-latency, well-defined categories. (e.g., classifying "Refund" vs "Tech Support"). It's fast (just dot product) and cheap.
*   **LLM Router:** Use for complex, ambiguous routing where reasoning is required. (e.g., "Does this legal question require the Contract DB or the Case Law DB?"). It's slower and costs per token, but handles nuance better.

#### Q3: Explain "HyDE" (Hypothetical Document Embeddings). What is its main weakness?

**Answer:**
*   **Concept:** Generate a fake answer, embed it, search.
*   **Weakness:** **Hallucination bias.** If the LLM has a strong misconception about the topic (e.g., it thinks the Earth is flat), the hypothetical answer will be about flat earth. The retrieval will then find "Flat Earth" documents, reinforcing the error. HyDE works best when the model knows the *vocabulary* but not the specific *facts*.

#### Q4: How do you handle "Multi-Hop" reasoning in RAG?

**Answer:**
*   **Problem:** Question: "Who is the CEO of the company that acquired GitHub?"
    *   Hop 1: "Who acquired GitHub?" -> Microsoft.
    *   Hop 2: "Who is CEO of Microsoft?" -> Satya Nadella.
*   **Solution:** **Iterative Retrieval (or ReAct).** The agent retrieves, reads, realizes it needs more info, generates a new query, retrieves again. Standard "One-Shot RAG" fails here.

### Production Challenges

#### Challenge 1: Router Latency

**Scenario:** You added an LLM Router step. Now every request takes +1 second.
**Root Cause:** Sequential LLM call.
**Solution:**
*   **Distillation:** Train a small BERT classifier (or use a small LLM like Haiku/GPT-3.5) to do the routing.
*   **Parallelism:** If the cost of retrieval is low, query *all* sources in parallel and let the synthesizer decide which data to use (trading compute for latency).

#### Challenge 2: Decomposition Explosion

**Scenario:** The user asks a vague question. The decomposer generates 10 sub-questions. The system makes 10 vector DB calls and 10 LLM calls. Latency spikes to 30s.
**Root Cause:** Unconstrained decomposition.
**Solution:**
*   **Limit Sub-questions:** Prompt the decomposer to "Generate max 3 sub-questions".
*   **Async Execution:** Run the sub-question retrieval/answering in parallel (Python `asyncio.gather`).

#### Challenge 3: Conflicting Information

**Scenario:**
*   Doc A (2021): "Revenue is $1M".
*   Doc B (2023): "Revenue is $2M".
*   Agent retrieves both and says: "Revenue is $1M and $2M."
**Root Cause:** Lack of temporal reasoning or conflict resolution.
**Solution:**
*   **Metadata Filtering:** Filter by `date > 2022`.
*   **Ranking:** Sort retrieved chunks by date.
*   **Synthesis Prompt:** "If documents conflict, prioritize the most recent one."

#### Challenge 4: The "Empty Search" Loop

**Scenario:**
*   Agent searches "Project X". No results.
*   Agent rewrites to "Project X details". No results.
*   Agent loops forever.
**Root Cause:** The data just isn't there.
**Solution:**
*   **Max Retries:** Stop after 2 rewrites.
*   **Fallback:** If vector search fails, try keyword search (BM25). If that fails, ask the user for clarification.

### System Design Scenario: Legal Research Assistant

**Requirement:** Answer questions based on Case Law (Millions of docs) and Statutes (Thousands of docs).
**Design:**
1.  **Ingestion:** Separate indices for `cases` and `statutes`.
2.  **Router:** "Does this query relate to a specific case or a general law?"
3.  **Decomposition:** "What are the elements of fraud?" -> "What is the statute for fraud?" + "What are key cases interpreting fraud?"
4.  **Citation:** The generator must cite `[Source: Case X, Page 12]`.
5.  **Verification:** A post-retrieval step checks if the cited text actually supports the claim (Hallucination check).

### Summary Checklist for Production
*   [ ] **Routing:** Use a fast router (Semantic or Small LLM) to avoid latency.
*   [ ] **Decomposition:** Parallelize sub-queries.
*   [ ] **Fallbacks:** Implement Keyword Search if Vector Search fails.
*   [ ] **Citations:** Enforce citation format in the prompt.
*   [ ] **HyDE:** Use cautiously; validate against standard retrieval.
