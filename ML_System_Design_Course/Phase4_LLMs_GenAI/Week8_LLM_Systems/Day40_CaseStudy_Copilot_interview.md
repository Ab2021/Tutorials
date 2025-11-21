# Day 40: Case Study: Copilot - Interview Questions

> **Topic**: Coding Assistant Design
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Design an AI Coding Assistant (GitHub Copilot).
**Answer:**
*   **Goal**: Autocomplete code, Chat, Refactor.
*   **Latency**: < 200ms for autocomplete.
*   **Context**: Current file, Open tabs, Repo structure.

### 2. What is FIM (Fill-In-the-Middle)?
**Answer:**
*   Training objective: `<prefix> [MASK] <suffix>`.
*   Allows model to use context *after* the cursor.
*   Crucial for inserting code in middle of function.

### 3. How do you construct the prompt for completion?
**Answer:**
*   **Current File**: Text before/after cursor.
*   **Imports**: Signatures of imported functions.
*   **Similar Files**: Retrieve using Jaccard similarity or Embeddings.
*   **Language**: Add language tag.

### 4. How do you handle Privacy (PII/Secrets)?
**Answer:**
*   **Client-side filtering**: Regex to detect API keys, Emails. Block sending to server.
*   **Server-side**: Zero retention policy.

### 5. How do you reduce Latency?
**Answer:**
*   **Streaming**: Show tokens as they arrive.
*   **Speculative Decoding**.
*   **Small Model**: Use 1B-3B model for autocomplete, 70B for chat.
*   **Caching**: Prefix caching for common imports.

### 6. What is "Repo-level Context"?
**Answer:**
*   Understanding the whole codebase, not just one file.
*   **RAG**: Index code chunks. Retrieve relevant snippets.
*   **Knowledge Graph**: Class hierarchy, function calls.

### 7. How do you evaluate a Copilot?
**Answer:**
*   **Acceptance Rate**: % of suggestions accepted by user. (Gold standard).
*   **Retention**: % of accepted code still present after 10 mins.
*   **Latency**.

### 8. What is "Ghost Text"?
**Answer:**
*   The gray text overlay shown in IDE.
*   UX pattern. User presses Tab to accept.

### 9. How do you handle "Hallucinated APIs"?
**Answer:**
*   Model invents functions.
*   **Fix**: RAG (Retrieve valid APIs). Static Analysis (Check if function exists).

### 10. What is the architecture?
**Answer:**
*   **IDE Plugin**: Collects context, debounces keystrokes.
*   **Gateway**: Auth, Rate limit.
*   **Inference Service**: LLM.
*   **Telemetry**: Logs acceptance.

### 11. How do you handle "Debouncing"?
**Answer:**
*   Don't send request on every keystroke.
*   Wait for pause (e.g., 100ms) or specific triggers (Enter, Dot, Space).

### 12. What is "Prompt Engineering" for Code?
**Answer:**
*   Comment-driven development.
*   Providing type hints helps model.

### 13. How do you train a Code LLM?
**Answer:**
*   **Pre-training**: The Stack (GitHub data).
*   **Fine-tuning**: Commit history (Before -> After).
*   **Instruction Tuning**: "Write a function to..."

### 14. What is "StarCoder" / "CodeLlama"?
**Answer:**
*   Open source code models.
*   Trained on 80+ languages.
*   FIM support.

### 15. How do you handle License Compliance?
**Answer:**
*   Filter out GPL/AGPL code from training data.
*   "Reference Search": Check if generated code matches existing public code verbatim.

### 16. What is "Multi-line" vs "Single-line" completion?
**Answer:**
*   **Single**: Fast, low risk.
*   **Multi**: High value, high risk (distracting).
*   Heuristics to decide when to stop.

### 17. How do you personalize to the user's style?
**Answer:**
*   **In-context**: Include user's previous code in prompt.
*   **LoRA**: Fine-tune adapter on user's repo (Private Copilot).

### 18. What is "Semantic Search" for Code?
**Answer:**
*   Embed code snippets.
*   "Find function that connects to DB" -> Returns `db_connect()`.

### 19. How do you handle "Security Vulnerabilities"?
**Answer:**
*   Filter training data for CVEs.
*   Run security scanner on output.

### 20. What is the "Context Window" challenge in Code?
**Answer:**
*   Repos are huge.
*   Need long context (100k+) or smart retrieval.
