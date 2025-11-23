# Day 41: Agent Memory Systems
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What are the different types of memory in an agent system?

**Answer:**
- **Short-Term:** Current session (last 10-20 turns). Sliding window.
- **Long-Term:** All past sessions. Stored in vector DB.
- **Working:** Current task context. Scratchpad in prompt.
- **Episodic:** Specific events with timestamps.
- **Semantic:** General facts/preferences about the user.
- **Use Case:** Short-term for context, long-term for "remember when...", semantic for personalization.

#### Q2: How do you implement memory retrieval?

**Answer:**
- **Recency:** Most recent memories (sorted by timestamp).
- **Relevance:** Most similar memories (vector search).
- **Importance:** High-priority memories (user-flagged, frequently accessed).
- **Hybrid:** Combine all three with weighted scoring.
- **Formula:** $\text{Score} = 0.5 \cdot \text{Relevance} + 0.3 \cdot \text{Recency} + 0.2 \cdot \text{Importance}$

#### Q3: What is memory consolidation and why is it necessary?

**Answer:**
- **Problem:** Too many memories → context overflow.
- **Consolidation:** Summarize or forget old memories.
- **Methods:**
  - **Summarization:** 100 messages → 1 summary paragraph.
  - **Forgetting:** Delete low-importance memories (not accessed in 30 days).
  - **Compression:** Merge similar memories.
- **Benefit:** Keeps memory manageable while retaining key information.

#### Q4: How do you extract semantic knowledge from conversations?

**Answer:**
- **Reflection:** After conversation, prompt LLM to extract learnings.
- **Prompt:** "What did I learn about the user? Extract preferences, habits, facts."
- **Storage:** Store as key-value pairs (e.g., `{"prefers_concise": true}`).
- **Use:** Include in future prompts for personalization.

#### Q5: What are the privacy concerns with agent memory?

**Answer:**
- **PII Storage:** Storing personal data raises privacy issues.
- **Data Leakage:** Memory from one user shouldn't leak to another.
- **Consent:** Users should control what's remembered.
- **Solutions:**
  - **User Isolation:** Strict access control per user.
  - **Encryption:** Encrypt memories at rest.
  - **Expiration:** Auto-delete after N days.
  - **User Control:** Allow users to view/delete memories.

---

### Production Challenges

#### Challenge 1: Memory Accuracy

**Scenario:** Agent "remembers" something the user never said.
**Root Cause:** LLM hallucinated during memory extraction.
**Solution:**
- **Store Exact Text:** Don't summarize. Store verbatim conversation.
- **Verification:** When recalling, show the original message for verification.
- **Confidence Scores:** Only store high-confidence extractions.

#### Challenge 2: Context Window Overflow

**Scenario:** User has 1000 past conversations. Can't fit all in context.
**Solution:**
- **Retrieval:** Only retrieve top 5 most relevant memories.
- **Summarization:** Summarize old conversations into 1-2 sentences.
- **Hierarchical:** Store summaries at multiple levels (message → conversation → week → month).

#### Challenge 3: Memory Staleness

**Scenario:** User's preferences change. Old memories are outdated.
**Example:** "User prefers Python" (2023) but now prefers Rust (2024).
**Solution:**
- **Recency Weighting:** Weight recent memories higher.
- **Expiration:** Mark old memories as "potentially outdated".
- **Conflict Resolution:** If conflicting memories, ask user to clarify.

#### Challenge 4: Cold Start

**Scenario:** New user has no memory. Agent can't personalize.
**Solution:**
- **Onboarding:** Ask initial questions to build memory.
  - "What programming languages do you use?"
  - "Do you prefer concise or detailed answers?"
- **Defaults:** Use sensible defaults until memory is built.
- **Progressive Learning:** Learn incrementally as user interacts.

#### Challenge 5: Multi-User Memory Isolation

**Scenario:** In a shared environment, User A's memory leaks to User B.
**Root Cause:** Insufficient access control.
**Solution:**
- **User ID Filtering:** Always filter by `user_id` in vector DB queries.
- **Separate Indexes:** Create one vector DB index per user (if feasible).
- **Audit Logs:** Log all memory access for debugging.

### Summary Checklist for Production
- [ ] **Storage:** Use **vector DB** for long-term memory.
- [ ] **Retrieval:** Combine **relevance + recency + importance**.
- [ ] **Consolidation:** **Summarize** old memories periodically.
- [ ] **Extraction:** Use **LLM reflection** to extract semantic knowledge.
- [ ] **Privacy:** **Encrypt** memories, **isolate** by user.
- [ ] **Expiration:** **Auto-delete** memories after **90 days** (or user-defined).
