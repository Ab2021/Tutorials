# Day 74: Agentic Memory & Personalization
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between Episodic and Semantic memory in Agents?

**Answer:**
- **Episodic:** Memory of specific events/experiences. "I tried to run the code at 10:00 PM and it failed." (Sequential, Time-based).
- **Semantic:** Memory of facts/concepts. "The user is a Python developer." "The API endpoint is /v1/chat." (General knowledge).
- **Implementation:** Episodic -> Vector DB logs. Semantic -> Knowledge Graph or SQL.

#### Q2: How does MemGPT solve the context window limit?

**Answer:**
- It treats the Context Window as "RAM" and the Vector DB as "Disk".
- It uses an OS-like paging mechanism to swap information in and out of the context window based on relevance.
- It allows the agent to explicitly call tools to `save_memory` or `search_memory`.

#### Q3: Why is "Retrieval" difficult for long-term memory?

**Answer:**
- **Relevance:** A simple cosine similarity search might return a conversation from 3 months ago that shares keywords but is irrelevant to the current context.
- **Recency:** You usually want the *latest* instruction ("I moved to New York"), not the old one ("I live in London"). Vector search doesn't inherently prioritize recency.

#### Q4: What are the privacy risks of Agent Memory?

**Answer:**
- **PII Retention:** Storing user chats forever means storing PII forever.
- **Leakage:** If the vector DB is shared, User A might retrieve User B's memory.
- **Right to Delete:** Hard to find and delete all vectors associated with a user if not properly tagged.

#### Q5: Explain "Generative Agents" (Simulacra).

**Answer:**
- A research paper where agents live in a sandbox town.
- Key mechanism: **Observation -> Memory Stream -> Retrieval -> Reflection -> Planning -> Action**.
- Agents form relationships and spread information organically through this memory architecture.

---

### Production Challenges

#### Challenge 1: The "Stale Profile"

**Scenario:** User says "I switched to Node.js". Agent continues to give Python code because the Profile says "Python Dev".
**Root Cause:** Profile update logic failed or wasn't triggered.
**Solution:**
- **Trigger:** Run an extraction step after *every* session or when specific keywords ("switched", "changed") are detected.
- **Conflict Resolution:** If new fact conflicts with old fact, overwrite and log the change.

#### Challenge 2: Memory Hallucination

**Scenario:** Agent "remembers" you promised to pay it $100.
**Root Cause:** LLM confabulation mixed with retrieved context.
**Solution:**
- **Source Citing:** Memory retrieval should return the *exact timestamp/message ID*.
- **Verification:** "I found a record of X in our chat history."

#### Challenge 3: Vector DB Cost

**Scenario:** Storing every single message for 1M users. Vector DB bill explodes.
**Root Cause:** Storing raw logs.
**Solution:**
- **Summarization:** Summarize daily chats and store only the summary.
- **TTL:** Delete episodic memory older than 1 year. Keep only semantic facts.

#### Challenge 4: Context Pollution

**Scenario:** Retrieving too much past memory confuses the model. "You said X last year, but Y today." Model tries to reconcile both.
**Root Cause:** Low precision retrieval.
**Solution:**
- **Re-ranking:** Re-rank retrieved memories by Recency and Relevance.
- **Prompting:** "Prioritize the most recent information."

#### Challenge 5: Cold Start

**Scenario:** New user has no memory. Agent feels generic.
**Root Cause:** Empty profile.
**Solution:**
- **Onboarding:** Ask 3 key questions at start ("What is your role?", "Preferred language?").
- **Defaults:** Use a "Default Persona" until enough data is gathered.

### System Design Scenario: Personal AI Tutor

**Requirement:** Tutor that remembers student's weak areas over a semester.
**Design:**
1.  **Memory:** Vector DB for all Q&A history.
2.  **Profile (SQL):** `{"weak_topics": ["Calculus", "Recursion"], "learning_style": "Visual"}`.
3.  **Update:** After each quiz, update `weak_topics`.
4.  **Retrieval:** When teaching a new concept, check if it relates to a `weak_topic`. "Since you struggled with Recursion, let's review that first."
5.  **Privacy:** Encrypt student data.

### Summary Checklist for Production
- [ ] **Architecture:** Hybrid **Vector (Episodic) + SQL (Semantic)**.
- [ ] **Privacy:** Implement **Tenant Isolation**.
- [ ] **Maintenance:** Implement **Summarization** to save space.
- [ ] **Updates:** Auto-update **User Profile** on changes.
- [ ] **Limits:** Set **TTL** on raw logs.
