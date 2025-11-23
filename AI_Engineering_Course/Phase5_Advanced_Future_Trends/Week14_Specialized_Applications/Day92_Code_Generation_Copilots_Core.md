# Day 92: Code Generation & Copilots
## Core Concepts & Theory

### The Rise of AI Coding Assistants

Code generation is the "Killer App" of LLMs.
Models like Codex, StarCoder, and Llama-3-Code have transformed software engineering.
It's not just "Autocomplete"; it's "Intent-to-Code".

### 1. Fill-In-the-Middle (FIM)

Standard LLMs are Left-to-Right (Causal).
*   *Prompt:* `def add(a, b):` -> *Output:* `return a + b`
*   *Problem:* Editing code requires knowing the *suffix* (what comes after the cursor).
*   *Solution:* **FIM Training**.
    *   Format: `<PRE> prefix <SUF> suffix <MID> middle`
    *   The model learns to predict the missing middle part given the surrounding context.

### 2. Context Construction (The "Prompt")

A Copilot doesn't just see the cursor line. It sees:
*   **Current File:** The open tab.
*   **Recent Files:** Other tabs (Jaccard similarity).
*   **Imports:** Definitions of functions imported in the file.
*   **Cursor History:** Where you were editing 5 seconds ago.
*   **Repo Map:** A compressed graph of the entire repository structure.

### 3. Repository-Level Awareness

How do you fit a 1GB repo into a 32k context window?
*   **RAG:** Retrieve relevant snippets.
*   **RepoMap:** A tree-sitter based skeleton of the repo (Class names, Function signatures) without the bodies. This gives the model a "map" of the codebase.

### 4. Code-Specific Pre-training

*   **Data:** The Stack (GitHub), StackOverflow.
*   **Deduplication:** Crucial. Exact and Near-deduplication (MinHash) prevents the model from memorizing boilerplate or GPL code.
*   **PII Scrubbing:** Removing API keys and emails from training data.

### Summary

Building a Copilot is 10% Model and 90% Context Engineering. The goal is to provide the *most relevant* code snippets to the model with the lowest latency.
