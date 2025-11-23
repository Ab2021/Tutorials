# Day 74: Agentic Memory & Personalization
## Core Concepts & Theory

### The Stateless Problem

**LLMs are stateless:** Every call is a fresh start.
**Agents need memory:** To learn from mistakes, remember user preferences, and maintain long-running tasks.

### 1. Types of Memory

**Sensory Memory:**
- Raw inputs (Audio/Video buffer). Short retention.

**Short-Term Memory (Working Memory):**
- The Context Window.
- Holds the current conversation history.
- **Limit:** 128k - 1M tokens. Expensive and volatile.

**Long-Term Memory:**
- **Episodic:** Remembering past events ("We talked about X yesterday").
- **Semantic:** Remembering facts ("User likes Python").
- **Procedural:** Remembering how to do things ("To run code, use the exec tool").

### 2. Memory Architectures

**Vector DB as Memory:**
- Store past conversations as embeddings.
- Retrieve relevant past turns based on current query.
- **Pros:** Infinite capacity.
- **Cons:** Lossy retrieval (might miss context).

**MemGPT (OS for LLMs):**
- **Virtual Context Management:** Swaps data between "Main Memory" (Context Window) and "External Storage" (Vector DB/SQL).
- **Events:** System events trigger memory updates.

**Graph Memory:**
- Storing facts in a Knowledge Graph.
- `(User, Likes, Python)`, `(User, Role, Developer)`.
- **Pros:** Structured, explicit.

### 3. Personalization

**User Profile:**
- A structured summary of the user.
- **Fields:** Name, Role, Tech Stack, Tone Preference.
- **Update:** Auto-update based on conversation ("User mentioned they switched to Rust").

**Adaptive Tone:**
- If user is terse, be terse.
- If user is beginner, explain more.

### 4. Reflection & Learning

**Self-Correction:**
- Agent stores "Lessons Learned".
- Before taking action, retrieve "Related Lessons".
- "Last time I tried X, it failed. I should try Y."

### 5. Privacy & Forgetting

**Right to be Forgotten:**
- GDPR requires ability to delete user data.
- **Challenge:** If memory is mixed in a shared Vector DB, how to delete just one user's data?
- **Solution:** Tenant isolation (Metadata filtering).

### 6. Context Compression

**Summarization:**
- Instead of storing raw logs, store summaries.
- "User asked about X. I answered Y." -> "Discussed X."

**Entity Extraction:**
- Extract key entities and store in SQL.

### 7. Generative Agents (Simulacra)

**Stanford Paper:**
- Agents observing, planning, and reflecting.
- **Memory Stream:** A time-ordered list of observations.
- **Retrieval:** Recency, Importance, Relevance.

### 8. Summary

**Memory Strategy:**
1.  **Short-Term:** Use **Context Window** (Rolling buffer).
2.  **Long-Term:** Use **Vector DB** for episodic history.
3.  **Facts:** Use **SQL/Graph** for user profile.
4.  **Management:** Use **MemGPT** pattern to manage context.
5.  **Privacy:** Isolate data by **User ID**.

### Next Steps
In the Deep Dive, we will implement a Memory Module with Vector storage, a User Profile Builder, and a MemGPT-style context manager.
