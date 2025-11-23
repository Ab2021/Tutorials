# Day 60: Shared Memory & State Management in Swarms
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is "Stigmergy" in the context of AI Agents?

**Answer:**
Stigmergy is a mechanism of indirect coordination.
*   **Direct:** Agent A tells Agent B "Do this."
*   **Stigmergic:** Agent A leaves a "trace" in the environment (e.g., updates a file). Agent B sees the file changed and decides to act.
*   *Example:* In a coding swarm, the "Linter Agent" wakes up whenever a `.py` file is modified. It doesn't need to be told; it watches the environment.

#### Q2: How do you handle "Race Conditions" in Shared Memory?

**Answer:**
Two agents try to update the `summary` field at the same time.
*   **Strategy 1: Reducers.** (LangGraph). Define a merge function (e.g., concatenate both summaries).
*   **Strategy 2: Locking.** (Blackboard). Agent A locks the board, writes, unlocks. Agent B waits.
*   **Strategy 3: Versioning.** (Optimistic Concurrency). Agent B reads v1. Agent A writes v2. Agent B tries to write v2 but fails because v2 exists. Agent B must re-read and merge.

#### Q3: What is the difference between "Agent State" and "Global State"?

**Answer:**
*   **Agent State:** Private. The agent's conversation history, current thought process, and scratchpad.
*   **Global State:** Public. The Blackboard, the file system, the database.
*   *Design Rule:* Keep Agent State ephemeral. Persist Global State.

#### Q4: Why is "Event Sourcing" useful for Agents?

**Answer:**
Instead of storing just the *current* state ("Status: Done"), store the *sequence of events* ("Started", "Working", "Done").
*   **Debuggability:** You can replay the events to see *how* the swarm reached a bad state.
*   **Rollback:** You can revert the state by replaying events up to time T-1.

### Production Challenges

#### Challenge 1: The "Bloated" Blackboard

**Scenario:** Agents keep dumping data onto the blackboard. The context window fills up when you pass the blackboard to the next agent.
**Root Cause:** Lack of garbage collection.
**Solution:**
*   **Summarization:** A "Janitor Agent" periodically reads the blackboard and replaces old details with a summary.
*   **Segmentation:** Only pass relevant slices of the blackboard to each agent (e.g., Coder only sees `code`, not `marketing_plan`).

#### Challenge 2: Hallucinated Updates

**Scenario:** Agent A says "I updated the database" but didn't actually call the tool. Agent B reads the chat, thinks it's done, and proceeds.
**Root Cause:** Trusting natural language over system state.
**Solution:**
*   **Tool-Driven State:** The state only updates when the *Tool* returns success, not when the *Agent* says it did. The Blackboard should be updated by the Tool Output, not the Agent's thought.

#### Challenge 3: Infinite Reaction Loops

**Scenario:**
*   Agent A: "I fixed the bug." (Updates Board).
*   Agent B: (Sees update) "I verified it." (Updates Board).
*   Agent A: (Sees update) "Thanks." (Updates Board).
*   ...
**Root Cause:** Triggering on *any* change.
**Solution:**
*   **Filter Triggers:** Agent A only reacts to `status="FAILED"`. Agent B only reacts to `status="FIXED"`.

### System Design Scenario: Collaborative Coding Environment

**Requirement:** 3 Agents (Coder, Reviewer, Security) working on the same repo.
**Design:**
1.  **Shared State:** The Git Repository (File System).
2.  **Communication:** GitHub Pull Requests (Comments).
3.  **Workflow:**
    *   Coder pushes branch.
    *   Reviewer comments on PR (Stigmergy).
    *   Coder sees comment, fixes code.
    *   Security Agent scans PR, blocks merge if vulnerable.
4.  **Memory:** The "PR Description" acts as the Blackboard.

### Summary Checklist for Production
*   [ ] **Schema:** Define a strict Pydantic schema for the shared state.
*   [ ] **Reducers:** Define how parallel updates are merged.
*   [ ] **Janitor:** Implement a mechanism to clean up/summarize old state.
*   [ ] **Triggers:** Ensure agents only wake up for relevant state changes.
