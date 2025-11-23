# Day 96: Legal & Compliance Agents
## Core Concepts & Theory

### The High-Precision Domain

Legal AI is high-value but zero-tolerance for error.
*   **Tasks:** Contract Review, E-Discovery, Regulatory Compliance, Patent Search.
*   **Requirement:** Every claim must be cited. No hallucination allowed.

### 1. Contract Review & Extraction

*   **Extraction:** "What is the termination clause?" "What is the governing law?"
*   **Risk Analysis:** "Does this indemnity clause expose us to unlimited liability?"
*   **Redlining:** The agent suggests edits to protect the client.

### 2. E-Discovery (Electronic Discovery)

In lawsuits, lawyers must sift through millions of emails to find evidence.
*   **TAR (Technology Assisted Review):** Old ML method.
*   **LLM Agents:** Can understand nuance ("Find emails where Bob implies he knew about the fraud").
*   **Recall:** Must be near 100%. Missing a smoking gun is a disaster.

### 3. Regulatory Compliance

*   **Monitoring:** Watching the Federal Register for new laws.
*   **Mapping:** Mapping new laws to internal company policies.
*   **Gap Analysis:** "Do our current policies cover the new EU AI Act?"

### 4. Privacy & Confidentiality

Law firms handle highly sensitive data.
*   **On-Prem:** Many firms refuse to send data to OpenAI.
*   **Local Models:** Fine-tuned Llama-3 or Mistral running on secure servers.
*   **Data Retention:** Zero-day retention policies.

### Summary

Legal Agents are **Augmentation**, not Automation. They act as a "Junior Associate" who reads the documents and highlights issues for the Partner to review.
