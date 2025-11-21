# Day 40: Case Study - Building a Copilot (Code Completion)

> **Phase**: 4 - LLMs & GenAI
> **Week**: 8 - LLM Systems
> **Focus**: Real-time Coding Assistant
> **Reading Time**: 60 mins

---

## 1. The Problem

**Goal**: Autocomplete code in VSCode.
**Constraint**: Latency < 50ms (Ghost text must appear instantly). Privacy (Code cannot leave premise for some clients).

---

## 2. The Architecture

### 2.1 The Model: FIM (Fill-In-the-Middle)
Standard LLMs predict the future. Copilots must look at the past (prefix) AND the future (suffix) to fill the cursor position.
*   **Training Data**: `<PRE> def add(a, b): <SUF> return a + b <MID> \n    """Adds two numbers"""`
*   **Inference**: Send Prefix and Suffix. Model generates Middle.

### 2.2 Context Management
*   **Files**: Not just the current file. Use "Jaccard Similarity" to find relevant snippets in other open tabs.
*   **Imports**: Resolve imports to understand types.

### 2.3 Serving Infrastructure
*   **Speculative Decoding**: Essential for latency.
*   **Quantization**: INT8 for throughput.
*   **Streaming**: Return characters as they are generated.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Privacy (PII/Secrets)
**Scenario**: Model suggests an AWS Key.
**Solution**:
*   **Pre-processing**: Regex scan for secrets/PII before sending to LLM.
*   **Post-processing**: Regex scan output.
*   **Entropy Filter**: High entropy strings (random characters) are likely secrets. Block them.

### Challenge 2: Repetition Loops
**Scenario**: `for i in range(10): print(i) print(i) print(i)...`
**Solution**:
*   **Repetition Penalty**: Penalize tokens that appeared recently.
*   **Stop Sequences**: Stop generation on `\n\n` or `def`.

---

## 4. Interview Preparation

### System Design Questions

**Q1: How do you train a model for FIM?**
> **Answer**: Take a code file. Randomly cut out a chunk. Move it to the end. Add special tokens `<PRE>`, `<SUF>`, `<MID>`. Train as a standard causal language model. This teaches the model to "infill".

**Q2: How do you handle "Repo-Level" context?**
> **Answer**: The context window is too small for the whole repo.
> 1.  **RAG**: Embed the repo chunks. Retrieve relevant chunks based on cursor location.
> 2.  **Knowledge Graph**: Build a graph of function calls/definitions. Traverse graph to find dependencies of the current function.

**Q3: Why is latency so critical for Copilot?**
> **Answer**: Human reaction time is ~200ms. If the ghost text appears after 500ms, the user has already typed the next character, breaking the flow. The system feels "laggy" and is turned off.

---

## 5. Further Reading
- [StarCoder: May the Source Be With You](https://arxiv.org/abs/2305.06161)
- [How Copilot Works (Reverse Engineering)](https://thakkarparth007.github.io/copilot-explorer/posts/copilot-internals)
