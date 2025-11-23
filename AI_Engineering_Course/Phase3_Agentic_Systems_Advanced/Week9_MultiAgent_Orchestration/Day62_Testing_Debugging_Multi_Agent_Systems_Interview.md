# Day 62: Testing & Debugging Multi-Agent Systems
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How do you debug a "Deadlock" in a Multi-Agent System?

**Answer:**
*   **Symptoms:** The system hangs. No new messages.
*   **Cause:** Agent A is waiting for B. Agent B is waiting for A. Or both are waiting for a "User Input" that never comes.
*   **Fix:**
    *   **Timeouts:** Every wait must have a timeout.
    *   **Orchestrator Watchdog:** A background process that checks "Has anyone spoken in the last 60s?". If not, poke the Manager.

#### Q2: What is "Drift" in Agent Personas and how do you test for it?

**Answer:**
*   **Drift:** Over a long conversation, the "Pirate Agent" stops speaking like a pirate.
*   **Test:**
    *   **Metric:** "Persona Consistency Score".
    *   **Implementation:** Use an LLM Judge to grade every Nth message: "Does this sound like a pirate? Yes/No." Plot the score over time.

#### Q3: Why are "Regression Tests" difficult for Agents?

**Answer:**
Because Agents are non-deterministic. Even with `temperature=0`, floating point differences on GPUs can cause slight variations.
**Strategy:**
*   **Semantic Similarity:** Don't assert `output == "Hello"`. Assert `similarity(output, "Hello") > 0.9`.
*   **Functional Checks:** Assert `database.user_created == True`. Ignore the chat text.

#### Q4: How do you trace a request across multiple agents?

**Answer:**
**Correlation IDs.**
1.  Assign a `trace_id` to the incoming User Request.
2.  Every message passed between agents must include this `trace_id` in the metadata.
3.  The Logger aggregates all logs with that `trace_id` to reconstruct the flow.

### Production Challenges

#### Challenge 1: The "Heisenbug"

**Scenario:** The bug happens in production but disappears when you turn on "Debug Mode" (Logging).
**Root Cause:** Timing/Latency. Logging slows down the system, masking race conditions.
**Solution:**
*   **Async Logging:** Ensure logging is non-blocking (fire and forget).
*   **Deterministic Simulation:** Replay the exact inputs and random seeds in a controlled environment.

#### Challenge 2: Cost of Observability

**Scenario:** Logging every token of every prompt/response for 1M requests creates TBs of data and costs a fortune in storage/processing.
**Root Cause:** Verbosity.
**Solution:**
*   **Sampling:** Log 100% of errors, but only 1% of successes.
*   **Redaction:** Don't log the full prompt context (which might be huge). Log only the diff or the tool calls.

#### Challenge 3: Privacy in Logs

**Scenario:** Agents handle PII. Logs capture PII. Compliance violation.
**Root Cause:** Unfiltered logging.
**Solution:**
*   **PII Scrubbing:** Run a PII scrubber on the log stream *before* writing to disk. Replace emails with `[EMAIL]`.

#### Challenge 4: Visualizing Complexity

**Scenario:** You have 50 agents. The graph looks like a spaghetti monster. You can't understand it.
**Root Cause:** Too many nodes.
**Solution:**
*   **Hierarchical Views:** Collapse sub-graphs. Show "Sales Team" as one node. Click to expand into "Sales Manager", "SDR", "AE".

### System Design Scenario: Debugging Platform for Agents

**Requirement:** Build a dashboard to monitor a live swarm.
**Design:**
1.  **Ingestion:** Agents send events (JSON) to a Kafka topic.
2.  **Processing:** A worker consumes Kafka, scrubs PII, calculates metrics (Latency, Cost).
3.  **Storage:** Store traces in ClickHouse (for fast query) or MongoDB.
4.  **UI:**
    *   **Live Feed:** Scrolling chat of the swarm.
    *   **Graph View:** Active nodes light up.
    *   **Alerts:** "Agent X is stuck."

### Summary Checklist for Production
*   [ ] **Correlation IDs:** Implement everywhere.
*   [ ] **Timeouts:** No infinite waits.
*   [ ] **PII Scrubbing:** Protect the logs.
*   [ ] **Golden Set:** Have a regression suite of 50+ scenarios.
*   [ ] **Tracing:** Use a tool like LangSmith.
