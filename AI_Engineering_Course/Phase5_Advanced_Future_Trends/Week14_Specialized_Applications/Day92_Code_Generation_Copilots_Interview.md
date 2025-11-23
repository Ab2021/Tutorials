# Day 92: Code Generation & Copilots
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the "FIM" objective and why is it needed?

**Answer:**
Fill-In-the-Middle. Standard Causal Language Modeling (CLM) predicts $P(Token_t | Tokens_{0:t-1})$. It ignores future tokens.
In coding, you often edit the middle of a file. The "Future" (closing braces, return statements) is highly relevant.
FIM rearranges the data: `Prefix + Suffix -> Middle`.

#### Q2: How do you handle "Repo-Level" context?

**Answer:**
You cannot feed the whole repo.
1.  **Jaccard Similarity:** Find files with similar imports/words.
2.  **Graph Traversal:** Follow `import` statements 1 level deep.
3.  **RepoMap:** Use Tree-Sitter to extract a skeleton (signatures only) of the most relevant files.

#### Q3: How do you prevent the model from generating GPL code?

**Answer:**
*   **Training:** Filter out GPL repositories.
*   **Inference:** Run a "Reference Checker". Hash the generated output (N-grams) and check against a database of known GPL code. If match found, block the output.

#### Q4: What is "Ghost Text" vs "Chat"?

**Answer:**
*   **Ghost Text:** Low latency (<100ms), single line completion, triggered automatically. (FIM).
*   **Chat:** High latency, multi-turn, explanation focused. (Standard Chat).

### Production Challenges

#### Challenge 1: Hallucinated Libraries

**Scenario:** Copilot suggests `import fast_json_parser`. You install it. It's malware.
**Root Cause:** The model memorized a typo or a malicious package name.
**Solution:**
*   **Package Verification:** The IDE extension should check if the package exists in PyPI/npm and has >N stars before suggesting it.

#### Challenge 2: The "Stop Token" Problem

**Scenario:** The model generates the code, then starts generating comments, then generates a whole new function, then a unit test.
**Root Cause:** Missing stop sequences.
**Solution:**
*   **Strict Stops:** Stop at `\n\n` (Function end) or `class` or `def`.
*   **Parser Check:** Stop when the AST is complete.

#### Challenge 3: Latency Jitter

**Scenario:** Sometimes it takes 50ms, sometimes 500ms. User turns it off.
**Root Cause:** Queueing at the server.
**Solution:**
*   **Debouncing:** Only request after the user stops typing for 300ms.
*   **Streaming:** Show the first token immediately.

### System Design Scenario: Enterprise Private Copilot

**Requirement:** Build a Copilot for a Bank. No code leaves the VPC.
**Design:**
1.  **Model:** Fine-tune StarCoder2-15B on the Bank's internal codebase (Cobol/Java).
2.  **Serving:** vLLM on on-prem GPUs (A100s).
3.  **Context:** Retrieval service indexing internal Confluence + Bitbucket.
4.  **IDE Plugin:** VSCode extension pointing to the internal API URL.

### Summary Checklist for Production
*   [ ] **Telemetry:** Track "Accept Rate" (how often user hits Tab). Target > 30%.
*   [ ] **Privacy:** Ensure no telemetry sends code snippets to the cloud.
*   [ ] **Debounce:** Tune the delay to balance responsiveness vs cost.
