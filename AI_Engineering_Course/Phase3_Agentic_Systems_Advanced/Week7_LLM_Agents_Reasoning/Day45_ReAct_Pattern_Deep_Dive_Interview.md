# Day 45: ReAct Pattern Deep Dive (Reasoning + Acting)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the "Observation" step in ReAct?

**Answer:**
It is the output of the external tool, fed back into the LLM's context.
*   Crucially, the **LLM does not generate this**. The System generates it by running the code.
*   If the LLM generates the Observation, it is a hallucination.

#### Q2: How does ReAct handle "I don't know"?

**Answer:**
Ideally:
1.  Thought: "I need to search X."
2.  Action: Search(X)
3.  Observation: "No results."
4.  Thought: "Search failed. I cannot answer."
5.  Answer: "I don't know."
Without ReAct, the model might just guess.

#### Q3: What is the difference between ReAct and MRKL?

**Answer:**
They are very similar.
*   **MRKL (Modular Reasoning, Knowledge and Language):** A system architecture that routes queries to different "Expert Modules" (Calculator, Wiki, DB).
*   **ReAct:** A specific prompting strategy (Thought-Action-Observation loop) to implement such a system.

#### Q4: Why is ReAct slower than standard generation?

**Answer:**
It requires serial round-trips.
*   Standard: 1 LLM call.
*   ReAct: N LLM calls + N Tool calls.
*   Latency = Sum of all steps.

### Production Challenges

#### Challenge 1: The "Loop of Death"

**Scenario:**
Thought: "I need to check the weather."
Action: `Weather("NY")`
Observation: "Service Unavailable"
Thought: "I need to check the weather."
Action: `Weather("NY")`
...
**Root Cause:** The model doesn't know how to handle errors.
**Solution:**
*   **Max Retries:** If same action fails twice, force the model to try a different strategy.
*   **System Prompt:** "If a tool fails, try a different tool or give up. Do not retry endlessly."

#### Challenge 2: Context Pollution

**Scenario:** The search tool returns 5,000 words of HTML. The context fills up. The model forgets the original question.
**Root Cause:** Unfiltered observations.
**Solution:**
*   **Observation Truncation:** Limit observation to 500 chars.
*   **Summarizer:** Use a cheap LLM to summarize the observation before feeding it to the ReAct agent.

#### Challenge 3: Tool Hallucination

**Scenario:** Action: `SuperSecretTool("args")`. But you don't have that tool.
**Root Cause:** The model thinks it has tools it doesn't.
**Solution:**
*   **Strict Schema:** Use OpenAI Function Calling with `tool_choice="auto"`. The model is constrained to the provided schema.

### System Design Scenario: Customer Support Agent

**Requirement:** An agent that can refund orders and check status.
**Design:**
1.  **Tools:** `get_order(id)`, `refund_order(id)`.
2.  **Safety:** `refund_order` requires "Human Confirmation" (HITL).
3.  **ReAct Trace:**
    *   User: "Refund my order #123."
    *   Thought: "Check status first."
    *   Action: `get_order(123)` -> "Delivered".
    *   Thought: "It was delivered. I can refund."
    *   Action: `refund_order(123)` -> "Please confirm."
    *   Answer: "I can refund order #123. Proceed?"

### Summary Checklist for Production
*   [ ] **Timeout:** Hard limit on total execution time (e.g., 30s).
*   [ ] **Logging:** Log the full trace for debugging.
*   [ ] **Cost:** Monitor token usage per query (ReAct is expensive).
