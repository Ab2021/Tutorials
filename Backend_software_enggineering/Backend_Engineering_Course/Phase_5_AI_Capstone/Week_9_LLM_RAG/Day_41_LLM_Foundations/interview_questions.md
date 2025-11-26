# Day 41: Interview Questions & Answers

## Conceptual Questions

### Q1: What is a "Token"?
**Answer:**
*   **Definition**: The atomic unit of text for an LLM. It's roughly 0.75 words.
*   **Example**: "Hamburger" might be 1 token. "Ingenious" might be 2 tokens ("In", "genious").
*   **Importance**: You pay by the token. Context windows are limited by tokens (e.g., 8k, 128k).

### Q2: What does the `temperature` parameter do?
**Answer:**
*   **Mechanism**: It controls the randomness of the next-token selection.
*   **Low (0.0)**: Always pick the most likely token. (Good for code, JSON).
*   **High (1.0)**: Pick less likely tokens sometimes. (Good for creative writing).

### Q3: What is "Hallucination"?
**Answer:**
*   **Definition**: The model confidently states a fact that is false.
*   **Cause**: The model predicts the next word based on patterns, not truth. If it hasn't seen the answer, it makes up a plausible-sounding one.
*   **Mitigation**: RAG (Retrieval Augmented Generation), Grounding, Low Temperature.

---

## Scenario-Based Questions

### Q4: You need to extract structured data (JSON) from an email. How do you ensure the LLM returns valid JSON?
**Answer:**
1.  **Prompting**: "Output ONLY valid JSON. Do not include markdown blocks."
2.  **Few-Shot**: Give examples of the desired JSON format.
3.  **JSON Mode**: Use the API's `response_format={"type": "json_object"}` (OpenAI feature).
4.  **Validation**: Parse the output in Python. If it fails, retry the request with the error message.

### Q5: The user input is too long for the Context Window. What do you do?
**Answer:**
1.  **Truncate**: Cut off the end (bad).
2.  **Summarize**: Ask the LLM to summarize the text first (Map-Reduce).
3.  **RAG**: Chunk the text, store in Vector DB, retrieve only relevant chunks.

---

## Behavioral / Role-Specific Questions

### Q6: A stakeholder wants to use an LLM to make medical diagnoses. What is your advice?
**Answer:**
*   **High Risk**.
*   **Liability**: Hallucinations can kill.
*   **Regulation**: HIPAA compliance.
*   **Advice**: Use LLM as an *assistant* to a doctor (summarizing notes), not a decision maker. Keep a "Human in the Loop".
