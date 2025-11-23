# Day 52: RAG with Tools (Query Decomposition & Routing)
## Core Concepts & Theory

### RAG is just a Tool

Traditionally, RAG is seen as a pipeline: `Input -> Retrieve -> Generate`.
In an Agentic world, **Retrieval is just another tool**.
This shift allows for much more dynamic behaviors. The agent can decide *if* it needs to retrieve, *what* to retrieve, and *how many times* to retrieve.

### 1. Query Decomposition (Sub-Question Query Engine)

Complex questions often cannot be answered by a single vector search.
*   *Question:* "Compare the revenue growth of Apple and Microsoft in 2023."
*   *Naive RAG:* Searches for the whole string. Gets mixed results.
*   *Decomposition:* The agent breaks this into:
    1.  "What was Apple's revenue growth in 2023?"
    2.  "What was Microsoft's revenue growth in 2023?"
    3.  *Synthesize:* Compare the two results.

**Mechanism:**
The LLM generates a list of sub-questions. Each sub-question is executed against the vector store (or different vector stores).

### 2. Query Routing

Not all questions need the same data source.
*   *Router:* A classifier (LLM or simple keyword) that directs the query.
*   *Sources:*
    *   **Vector DB:** For semantic queries ("How do I reset my password?").
    *   **SQL DB:** For structured queries ("How many users signed up yesterday?").
    *   **Web Search:** For recent events ("What is the stock price today?").

**Types of Routers:**
*   **LLM Router:** "Given the user query, select the best tool from [Vector, SQL, Web]."
*   **Semantic Router:** Embed the query and compare it to "canonical" queries for each tool. Faster/Cheaper than LLM.

### 3. Step-Back Prompting (Abstraction)

Sometimes the user's query is too specific.
*   *User:* "Why did my Python code fail with error X in library Y?"
*   *Step-Back:* The agent generates a more abstract question: "How does library Y handle error X generally?"
*   *Retrieval:* Search for both the specific and the abstract question.
*   *Benefit:* Provides high-level context + specific details.

### 4. HyDE (Hypothetical Document Embeddings)

A technique to improve retrieval for short/ambiguous queries.
1.  **Hallucinate:** The LLM generates a *hypothetical answer* to the user's question.
2.  **Embed:** Embed this hypothetical answer.
3.  **Retrieve:** Search the vector DB using this embedding.
4.  **Why?** The hypothetical answer matches the *distribution* of the target documents better than the raw question does.

### 5. Multi-Document Agents

When you have 100 PDFs, putting them all in one index can be noisy.
**Pattern:**
*   Create 1 Vector Index per PDF (or per topic).
*   Wrap each Index as a Tool (`Apple_10K_Tool`, `Microsoft_10K_Tool`).
*   The Agent selects which document to read based on the question.

### 6. Self-Correction in RAG

*   **Retrieval:** Agent searches.
*   **Grade:** Agent checks "Does this retrieved text actually answer the question?"
*   **Loop:** If No, rewrite the query and search again.
*   **Benefit:** Prevents hallucinations based on irrelevant context.

### Summary

Moving from "Linear RAG" to "Agentic RAG" involves adding reasoning steps before and after retrieval. Decomposition breaks down complexity. Routing selects the right source. Self-correction ensures quality. This turns a dumb search engine into a research assistant.
