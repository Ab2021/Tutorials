# Day 40 (Part 1): Advanced Copilot Case Study

> **Phase**: 6 - Deep Dive
> **Topic**: Coding Assistants
> **Focus**: FIM, Latency, and Privacy
> **Reading Time**: 60 mins

---

## 1. Training Objective: FIM

Standard Causal LM predicts Next Token. Copilot needs to insert code in the middle.

### 1.1 Fill-In-Middle (FIM)
*   **Data**: `def add(a, b): return a + b`
*   **Transform**: `<PRE> def add(a, b): <SUF> return a + b <MID> :`
*   **Inference**: User types `def add(a, b):`, cursor is here, `return a + b`.
*   Model sees `<PRE>...<SUF>...<MID>` and generates the missing colon.

---

## 2. Latency Constraints

### 2.1 The 50ms Budget
*   Typing speed: 100ms per keystroke.
*   Network RTT: 30ms.
*   Inference: Must be < 20ms.
*   **Solution**: Small models (Codex-Cushman, StarCoder-1B) running on Edge or close regions. Speculative decoding.

---

## 3. Tricky Interview Questions

### Q1: How to handle "Repo Context"?
> **Answer**:
> *   The file is small, but it imports 50 other files.
> *   **Jaccard Similarity**: Find files with similar imports/variable names.
> *   **Graph**: Use LSP (Language Server Protocol) to find Definition of functions used in current file. Add those definitions to prompt.

### Q2: Privacy: PII Redaction?
> **Answer**:
> *   Regex for Emails, IP addresses, API Keys.
> *   Replace with `<EMAIL>`, `<KEY>`.
> *   Run locally on client (VS Code extension) before sending to server.

### Q3: Client-Side Telemetry?
> **Answer**:
> *   **Acceptance Rate**: Did user press Tab?
> *   **Retention**: Did the code stay in the file 5 mins later? (Better metric).

---

## 4. Practical Edge Case: Ghost Text
*   **Problem**: Copilot suggests code that exists 5 lines below (Duplicate).
*   **Fix**: Suffix matching. If generation matches suffix, truncate.

