# Day 48: Cognitive Architectures (Generative Agents)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the "Retrieval Function" in Generative Agents?

**Answer:**
It determines what memory is brought into the context window.
Formula: $Score = \alpha \cdot Recency + \beta \cdot Importance + \gamma \cdot Relevance$.
*   **Recency:** Decay factor (0.99^hours).
*   **Importance:** Rated by LLM (1-10). "Ate toast" (1), "Got married" (10).
*   **Relevance:** Cosine similarity to the current query.

#### Q2: How does MemGPT handle context limits?

**Answer:**
It uses a **Virtual Context Management** system.
*   It treats the context window as a "Main Memory" (RAM).
*   It treats the Vector DB as "Disk".
*   The LLM can call functions like `archival_memory_insert` and `archival_memory_search` to move data back and forth, just like an OS pages memory.

#### Q3: What is the "Reflection" step?

**Answer:**
It is a higher-order reasoning step.
*   *Input:* 100 low-level observations.
*   *Process:* Ask LLM "What patterns do you see?"
*   *Output:* High-level generalizations ("I am an introvert").
*   *Purpose:* Prevents the agent from drowning in noise and allows it to act consistently with its personality.

#### Q4: Why are Cognitive Architectures expensive?

**Answer:**
Every action requires multiple internal LLM calls (Retrieve -> Reflect -> Plan -> Act).
A single "turn" in a simulation might cost $0.10.

### Production Challenges

#### Challenge 1: Memory Dilution

**Scenario:** The agent has 10,000 memories. The retrieval fetches 5 random ones. The agent acts schizophrenic.
**Root Cause:** Poor retrieval tuning.
**Solution:**
*   **Hybrid Search:** Combine Vector Search (Semantic) with Keyword Search (BM25) and Time Filtering.
*   **Summary Boost:** Always include the "Agent Persona" summary in the context, regardless of retrieval.

#### Challenge 2: The "Truman Show" Effect

**Scenario:** In a simulation, agents start hallucinating events that didn't happen because one agent lied and others believed it.
**Root Cause:** Information cascades.
**Solution:**
*   **Ground Truth Oracle:** A master system that validates "physical" actions (e.g., "Did Alice actually give Bob the item?").

#### Challenge 3: Latency

**Scenario:** User says "Hi". Agent takes 30 seconds to respond because it's searching its childhood memories.
**Root Cause:** Over-engineering.
**Solution:**
*   **Fast Path:** If the input is a greeting, skip retrieval. Use a simple response.
*   **Async Reflection:** Do the "Reflection" and "Planning" steps in the background, not in the blocking path of the chat.

### System Design Scenario: NPC in a Video Game

**Requirement:** An NPC that remembers the player's actions forever.
**Design:**
1.  **Short-Term Memory:** Last 10 chat turns (Context).
2.  **Long-Term Memory:** Vector DB.
3.  **Trigger:** When the player leaves the zone, run a "Summarization" job to compress the interaction into a single memory ("Player helped me kill the wolf").
4.  **Retrieval:** When player returns, query DB for "Player".

### Summary Checklist for Production
*   [ ] **Importance Tuning:** Don't save everything. Filter out "I walked 1 step" logs.
*   [ ] **Pruning:** Delete old, low-importance memories to save storage/cost.
*   [ ] **Consistency:** Ensure the agent doesn't contradict its own past.
