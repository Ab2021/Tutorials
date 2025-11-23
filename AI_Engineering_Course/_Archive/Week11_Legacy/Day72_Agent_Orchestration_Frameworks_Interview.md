# Day 72: Agent Orchestration Frameworks
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why do we need an orchestration framework like LangGraph instead of just a `while` loop?

**Answer:**
- **State Management:** Frameworks handle passing state between steps automatically.
- **Observability:** Built-in tracing (LangSmith) to see exactly where the agent failed.
- **Resilience:** Features like checkpointing (saving state to DB) allow resuming after a crash or pausing for human input.
- **Complexity:** Managing 10 agents with `if/else` spaghetti code is unmaintainable. Graphs provide structure.

#### Q2: What is the difference between "ReAct" and "Plan-and-Solve" patterns?

**Answer:**
- **ReAct (Reason + Act):** Interleaved thinking and acting. "I should search Google. *Search*. I found X. Now I should calculate Y. *Calc*." Good for dynamic tasks.
- **Plan-and-Solve:** Generate a full plan first. "1. Search Google. 2. Calculate Y. 3. Write summary." Then execute. Good for complex tasks requiring global view.

#### Q3: How does AutoGen handle multi-agent communication?

**Answer:**
- **Conversable Agents:** Every agent is an object that can send/receive messages.
- **Group Chat Manager:** A special agent that selects the next speaker (Speaker Selection Policy: Round Robin, Random, or LLM-based).
- **Termination:** Agents converse until a termination condition is met (e.g., "TERMINATE" string found).

#### Q4: What are the risks of autonomous agents in production?

**Answer:**
- **Infinite Loops:** Agent gets stuck trying the same failed action forever.
- **Cost Runaway:** Agent calls GPT-4 1000 times in a loop.
- **Side Effects:** Agent deletes a file or sends an email by mistake.
- **Solution:** Human-in-the-loop, Timeouts, Budget Limits, Read-only tools.

#### Q5: Explain the concept of "Reflection" in agents.

**Answer:**
- **Self-Correction:** The agent generates an output, then is prompted to "Critique this output".
- **Refinement:** The agent uses the critique to generate a better version.
- **Benefit:** drastically improves quality for coding and writing tasks (e.g., AlphaCodium).

---

### Production Challenges

#### Challenge 1: The "Stuck" Agent

**Scenario:** Agent keeps searching Google for "current weather", gets an error, and tries again. Infinite loop.
**Root Cause:** No error handling or loop limit.
**Solution:**
- **Max Iterations:** Hard limit (e.g., 10 steps).
- **Error Parsing:** If tool fails, feed the error back to the LLM: "Tool failed with error X. Try a different approach."
- **Circuit Breaker:** Stop if the same tool is called 3 times with same args.

#### Challenge 2: Context Window Overflow

**Scenario:** Agent runs for 50 steps. The conversation history grows to 100k tokens. LLM crashes or forgets initial instruction.
**Root Cause:** Unbounded state.
**Solution:**
- **Summarization:** Periodically summarize the history (Memory compaction).
- **FIFO:** Drop oldest messages (Rolling window).
- **Scratchpad:** Keep only the "Current Plan" and "Latest Observation".

#### Challenge 3: Multi-Agent Coordination Fail

**Scenario:** Agent A (Coder) waits for Agent B (Reviewer). Agent B asks a question. Agent A thinks it's a code review. They talk past each other.
**Root Cause:** Ambiguous system prompts or handoffs.
**Solution:**
- **Structured Handoffs:** Use a "Router" to explicitly pass control.
- **Clear Roles:** "You are the Reviewer. You ONLY output 'APPROVE' or 'REJECT'."

#### Challenge 4: Debugging Black Boxes

**Scenario:** The agent output is wrong. You have no idea which of the 10 steps went wrong.
**Root Cause:** Lack of tracing.
**Solution:**
- **LangSmith / Arize Phoenix:** Trace every LLM call and Tool output.
- **Visual Graph:** Visualize the path taken through the state machine.

#### Challenge 5: Latency

**Scenario:** A simple request takes 45 seconds because the agent did 5 "Thought" steps.
**Root Cause:** Excessive reasoning.
**Solution:**
- **Compile:** For common paths, hardcode the steps (don't use LLM to decide next step).
- **Faster Models:** Use Groq/Llama-3 for the reasoning steps, GPT-4 only for the final generation.

### System Design Scenario: Customer Support Agent Swarm

**Requirement:** Handle refunds, tech support, and sales.
**Design:**
1.  **Triage Agent:** Classifies intent -> Routes to specific sub-agent.
2.  **Refund Agent:** Has access to Stripe API.
3.  **Tech Agent:** Has access to RAG (Docs).
4.  **Sales Agent:** Has access to Calendar (Booking).
5.  **Orchestrator:** LangGraph StateMachine manages the handoffs.
6.  **Human:** If any agent fails or confidence < 0.7, route to Human.

### Summary Checklist for Production
- [ ] **Framework:** Use **LangGraph** for control.
- [ ] **Limits:** Set **Max Iterations** and **Budget**.
- [ ] **Memory:** Implement **Summarization** for long horizons.
- [ ] **Tracing:** Use **LangSmith** to debug.
- [ ] **Safety:** **Human-in-the-loop** for sensitive actions.
