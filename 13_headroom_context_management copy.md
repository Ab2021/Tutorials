# 🧠 Headroom Context Management: Architectural Masterclass
### Exhaustive Technical Breakdown of `chopratejas/headroom`

> [!IMPORTANT]
> This document is a comprehensive teardown of how the Headroom AI proxy handles Context Management. It covers the exact flow of data through the Rust core and Python proxy layers, the injection of CCR (Cached Context Reversible) mechanisms, the shaping of output tokens, and cross-agent memory handling. Use this to demonstrate deep mastery of Agentic AI system design.

---

## 🏗️ 1. The Interceptor Architecture: The Proxy Loop

Headroom operates as a transparent proxy (`headroom proxy --port 8787`) sitting between the developer's Agent (Claude Code, Cursor, Aider) and the foundational LLM API (Anthropic, OpenAI). Because it operates as a reverse proxy, it implements context management *without requiring any code changes* in the underlying agent.

### The Request Lifecycle (`headroom/proxy/server.py` & `helpers.py`):
When a payload (list of messages) arrives from the agent, it passes through several distinct architectural stages before hitting the LLM provider:

1.  **Ingestion & Auth (`auth_mode.py`)**: Validates the payload and securely proxies authentication (e.g., handling `copilot_macos_keychain.py` or Anthropic API keys).
2.  **Compression Decision (`compression_decision.py`)**: Evaluates if the payload *needs* compression. Small payloads bypass the heavy machinery.
3.  **Cross-Agent Memory Injection (`memory_injection.py`)**: Checks the local vector database for relevant past sessions or shared context across other agents and injects it into the system prompt.
4.  **Content Routing (`transforms/content_detector.rs`)**: Inspects the text to determine the optimal compression strategy (JSON, AST, or text).
5.  **Compression & CCR Stash**: Compresses the data, stores the original locally, and injects the CCR hash marker.
6.  **Output Shaping (`output_shaper.py`)**: Appends specific behavioral steering prompts to the system prompt to force the LLM to conserve output tokens.
7.  **Cache Alignment (`cache_control.rs`)**: Reorders and stabilizes the prompt structure to guarantee high KV cache hit rates at the provider level.

---

## 🗄️ 2. CCR: Cached Context Reversible (The Core Engine)

Standard RAG and context window truncation are *lossy*—once data is removed, the LLM can never access it. Headroom solves this using the **CCR (Compress-Cache-Retrieve)** paradigm, making the system "lossy on the wire, lossless end-to-end."

### The Storage Backends (`crates/headroom-core/src/ccr/backends/`)
The original, uncompressed context is stored in a high-performance backend, heavily optimized in Rust:
*   **`InMemoryCcrStore`**: A process-local, sharded `DashMap`. Excellent for single-developer local testing.
*   **`SqliteCcrStore`**: The production default. Runs in WAL (Write-Ahead Logging) mode with prepared statements and lazy TTL purging. It allows state persistence across proxy restarts.
*   **`RedisCcrStore`**: Opt-in backend for multi-worker environments. Crucial for enterprise deployments where requests might hit different load-balanced proxy instances.

### The Hashing & Injection Mechanism (`ccr/tool_injection.py` & `ccr/mod.rs`):
1.  **Hashing:** When a massive log file is intercepted, Headroom calculates a deterministic hash using **BLAKE3**. It takes the first 24 hex characters (`[a-f0-9]{24}`).
2.  **Stashing:** The raw log file is saved to the SQLite store using this 24-character hash as the key.
3.  **Marker Injection:** The compressed payload sent to the LLM is appended with a specific, immutable marker format: `<<ccr:{hash}>>` (e.g., `<<ccr:a1b2c3d4e5f6...>>`).
4.  **Tool Definitions:** Headroom secretly injects a new tool into the LLM's `tools` array: `headroom_retrieve`. The prompt explains: *"If you see a `<<ccr:HASH>>` marker and cannot answer the user's prompt because the text was compressed, call `headroom_retrieve(hash)`."*

### The Retrieval Loop (`ccr/response_handler.py`):
If the LLM calls `headroom_retrieve`:
1.  The LLM's response hits the Headroom proxy, *not* the user's agent.
2.  Headroom intercepts the tool call, extracts the requested hash.
3.  It fetches the original raw text from SQLite.
4.  It silently appends the raw text to the context window and re-prompts the LLM.
5.  The agent (e.g., Cursor) never knows this secondary loop happened. It simply receives the final, highly accurate answer.

---

## ⚙️ 3. Specialized Compression Strategies (The `transforms` Module)

Headroom does not rely on a generic text summarizer. It uses a **ContentRouter** to dispatch payloads to specialized engines built in Rust for maximum throughput.

### 1. SmartCrusher (JSON & Structured Data)
When the agent receives a massive JSON payload from a tool (e.g., an AWS API response), `SmartCrusher` kicks in.
*   It analyzes the JSON schema.
*   It drops highly repetitive keys.
*   It collapses massive arrays (e.g., `[{"id": 1, "val": "A"}, {"id": 2, "val": "B"}, ... 1000 more]` becomes `[{"id": 1, "val": "A"}, ... 999 omitted]`).
*   It preserves the *structure* so the LLM still understands the API contract.

### 2. CodeCompressor (AST Parsing)
When compressing source code, naive token dropping destroys syntax. `CodeCompressor`:
*   Parses the code into an Abstract Syntax Tree (AST).
*   Strips out non-functional elements like comments and docstrings.
*   If a file is provided merely for context (not being actively edited), it can collapse entire unmodified function bodies into signatures (e.g., `def calculate_tax(amount): ...`).

### 3. Kompress-v2-base (Semantic Text)
For raw prose, documentation, and conversational history, Headroom uses a specialized HuggingFace model (`chopratejas/kompress-v2-base`). This model is fine-tuned to extract the semantic essence of a paragraph, dropping filler words and redundant phrasing while retaining entities and relationships.

---

## 📉 4. Output Token Reduction & Shaping (`proxy/output_shaper.py`)

Reducing input tokens saves some money, but **output tokens are typically 5x more expensive** (especially on Claude 3.5 Sonnet / Opus). Headroom actively manages what the model *writes back*.

### Verbosity Steering (`verbosity_controller.py`)
Headroom dynamically appends steering instructions to the very end of the system prompt. Because it's at the end, it doesn't break the prefix caching.
*   *Prompt Injection:* "Be terse. Do not restate the context I just gave you. Do not use conversational preambles like 'Great, let me look at that'."

### Effort Routing & Inference Control
Agents often perform "routine" steps. For example, an agent might call a `read_file` tool just to check if a file exists.
*   **The Problem:** The LLM receives the file contents and often wastes 300 output tokens "thinking" about the file before deciding what to do next.
*   **The Solution:** Headroom detects when a turn is merely a routine tool resumption (e.g., a passing test result). It dynamically adjusts the LLM's inference parameters (or injects an "effort=low" directive) to prevent deep, unnecessary reasoning on that specific API call. When a *new* question or an *error* occurs, it restores full reasoning effort.

### Machine Learning Verbosity (`headroom learn`)
Developers don't explicitly declare their desired verbosity. Headroom's CLI includes a command: `headroom learn --verbosity`.
*   This mines local SQLite logs of past agent sessions.
*   It looks for behavioral cues: Did the user interrupt the LLM mid-generation? Did the user immediately execute a command without reading the 3-paragraph explanation?
*   It calculates an optimal verbosity score and applies it automatically to the proxy configuration.

---

## 🗂️ 5. KV Cache Alignment & Semantic Integrity

### The CacheAligner (`crates/headroom-core/src/cache_control.rs`)
LLM providers (like Anthropic) offer massive discounts for "Prompt Caching." However, this only works if the prefix of your prompt is *exactly identical* to a previous request.

Agents are notoriously bad at this; they constantly inject dynamic timestamps or slightly re-order tool definitions, destroying the KV cache.
*   **Stabilization:** Headroom acts as a "Cache Aligner." It intercepts the outgoing request, strips out volatile elements from the prefix, mathematically sorts the tool definitions alphabetically, and locks the system prompt into a rigid structure.
*   This guarantees that the bulk of the codebase context hits the provider's KV cache, cutting latency by up to 80%.

### Semantic Caching (`proxy/semantic_cache.py`)
Beyond provider-side KV caching, Headroom implements local semantic caching. If the agent asks a question whose embedding is semantically identical (cosine similarity > 0.98) to a question asked 10 minutes ago, Headroom intercepts the request and serves the cached answer locally, resulting in zero API cost and millisecond latency.

---

## 🌐 6. Cross-Agent Shared Memory (`memory_handler.py`)

A massive pain point in modern AI development is "Agent Amnesia." If you debug a complex React state issue using Claude Code, and then open Cursor the next day, Cursor doesn't know about the decisions Claude made.

*   **The Proxy Advantage:** Because Headroom sits beneath *all* of them at the network layer, it intercepts all context.
*   **Auto-Deduping Memory:** Headroom streams session summaries into a unified local vector database.
*   When Cursor is launched, Headroom's `memory_injection.py` surfaces the hard-won architectural decisions made by Claude Code yesterday, injecting them into Cursor's system prompt dynamically. 

This creates a unified, persistent "Developer Brain" that transcends the specific agent UI being used.
