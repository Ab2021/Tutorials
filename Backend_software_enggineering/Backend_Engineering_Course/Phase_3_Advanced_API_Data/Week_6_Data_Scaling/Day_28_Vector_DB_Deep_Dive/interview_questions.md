# Day 28: Interview Questions & Answers

## Conceptual Questions

### Q1: Why is "Pre-filtering" better than "Post-filtering" in Vector Search?
**Answer:**
*   **Scenario**: Find "Shoes" (Vector) that are "Red" (Metadata).
*   **Post-filtering**: Find top 100 "Shoes". Then filter for "Red".
    *   *Risk*: What if none of the top 100 are red? You return 0 results, even if red shoes exist.
*   **Pre-filtering**: Filter for "Red" first. Then search for "Shoes" *within* that subset. (Supported by Qdrant/Weaviate).

### Q2: What is the "Context Window" limit in RAG and how does Vector DB help?
**Answer:**
*   **Limit**: LLMs (like GPT-4) can only read X amount of text (e.g., 128k tokens). You can't feed it your entire company wiki.
*   **Role**: Vector DB acts as "Long Term Memory". It retrieves only the *relevant* chunks (e.g., 3 paragraphs) to fit into the Context Window.

### Q3: Explain "Dimensionality" in embeddings.
**Answer:**
*   The number of floating point numbers in the vector (e.g., 1536 for OpenAI, 384 for MiniLM).
*   **Trade-off**: Higher dimensionality = More semantic nuance, but Slower search and More RAM usage.

---

## Scenario-Based Questions

### Q4: You built a RAG bot, but it keeps hallucinating answers not in the docs. How do you fix it?
**Answer:**
1.  **Prompt Engineering**: Tell the LLM "Answer ONLY using the provided context. If unsure, say 'I don't know'."
2.  **Threshold**: If the Vector Search score is low (e.g., < 0.7), don't send anything to the LLM. Tell the user "No relevant docs found."
3.  **Citations**: Ask LLM to cite the source ID.

### Q5: How do you scale a Vector DB to 1 Billion vectors?
**Answer:**
*   **RAM**: Vectors are RAM hungry. 1B vectors * 1536 dims * 4 bytes = ~6 TB of RAM.
*   **Quantization**: Compress vectors (Float32 -> Int8). Reduces RAM by 4x with minimal accuracy loss.
*   **Disk-based Index**: Use engines (like Qdrant's disk storage) that keep vectors on SSD and only the HNSW graph in RAM.
*   **Sharding**: Distribute vectors across multiple nodes.

---

## Behavioral / Role-Specific Questions

### Q6: A stakeholder wants to use a Vector DB for *everything*, including keyword search. Do you agree?
**Answer:**
*   **No**.
*   **Weakness**: Vector search is bad at exact matches (Part numbers, User IDs, Acronyms). "IT" might match "Computer", but if I search for the "IT" department, I want exactness.
*   **Solution**: **Hybrid Search**. Use Elasticsearch/Postgres for keywords + Vector DB for semantic. Or use a DB that does both (Qdrant/Weaviate/Mongodb Atlas).
