# Day 43: Interview Questions & Answers

## Conceptual Questions

### Q1: Why do we need "Chunk Overlap"?
**Answer:**
*   **Scenario**: A sentence is split in half. "The CEO is... [CUT] ...John Smith."
*   **Result**: Chunk 1 has "The CEO is". Chunk 2 has "John Smith". Neither chunk has the full fact.
*   **Overlap**: If we overlap by 50 chars, Chunk 1 has "The CEO is John Smith". Chunk 2 has "is John Smith". The context is preserved.

### Q2: What is the "Lost in the Middle" phenomenon?
**Answer:**
*   **Observation**: LLMs are good at using information at the *start* and *end* of the prompt, but often ignore information in the *middle*.
*   **Impact on RAG**: If you retrieve 10 chunks, the answer might be in Chunk 5, and the LLM might miss it.
*   **Fix**: Re-rank chunks. Put the most relevant chunks at the start/end.

### Q3: Fine-Tuning vs RAG. When to use which?
**Answer:**
*   **RAG**: For **Knowledge**. (Facts, Documents, Real-time data).
*   **Fine-Tuning**: For **Behavior**. (Tone, Style, Format, Medical Terminology).
*   **Analogy**: RAG is giving the student a textbook. Fine-Tuning is sending the student to Med School.

---

## Scenario-Based Questions

### Q4: Your RAG system retrieves the wrong documents. How do you debug?
**Answer:**
1.  **Inspect Chunks**: Are they too small? Too big?
2.  **Check Embeddings**: Is the embedding model good for this domain? (e.g., Legal text might need a specialized model).
3.  **Hybrid Search**: Are you relying only on Vector Search? Add Keyword Search (BM25) to find exact matches.

### Q5: How do you handle "I don't know" in RAG?
**Answer:**
*   **Prompt Engineering**: Explicitly tell the model: "If the answer is not in the context, say 'I don't know'. Do not make things up."
*   **Threshold**: If the retrieval score (similarity) is low (e.g., < 0.7), don't even send it to the LLM. Just return "No relevant documents found."

---

## Behavioral / Role-Specific Questions

### Q6: A client wants to build a "Chat with your Data" app for 100GB of PDFs. What are the challenges?
**Answer:**
*   **Ingestion Time**: Processing 100GB takes time. Need a parallel processing pipeline (Spark/Ray).
*   **Cost**: Storing 100GB of vectors in Pinecone is expensive.
*   **Quality**: PDFs are messy (tables, images, multi-column). You need a good PDF parser (Unstructured.io, AWS Textract).
