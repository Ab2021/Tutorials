# Day 57: Multi-Agent Fundamentals
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: When should you use a Multi-Agent System vs. a Single Agent with Tools?

**Answer:**
*   **Single Agent:** Best for linear, low-complexity tasks (e.g., "Search for X and summarize"). It's cheaper and faster (fewer round trips).
*   **Multi-Agent:** Best for:
    *   **Separation of Concerns:** When the prompt for a single agent becomes too long/complex.
    *   **Diverse Roles:** When you need different "personas" (e.g., a Coder and a Lawyer) to collaborate.
    *   **Parallelism:** When sub-tasks can be done simultaneously.

#### Q2: What is the "Context Window" problem in Multi-Agent Systems?

**Answer:**
If Agent A and Agent B talk for 50 turns, their conversation history grows massive.
If you then pass this entire history to Agent C, you might blow the context limit (or just waste money).
**Solution:**
*   **Summarization:** Agent A summarizes the conversation before handing off to Agent C.
*   **Message Selection:** Only pass the last N messages.

#### Q3: Explain the "Manager-Worker" architecture.

**Answer:**
A central "Manager" (Orchestrator) LLM breaks down a high-level goal into sub-tasks.
It assigns these tasks to specific "Worker" agents (e.g., "Researcher", "Writer").
The Workers report back to the Manager.
The Manager aggregates the results and presents the final answer.
*   *Benefit:* Centralized control, easy to debug.
*   *Drawback:* The Manager is a bottleneck.

#### Q4: How do you prevent "Infinite Loops" between agents?

**Answer:**
*   **Scenario:** Agent A: "Here is the code." Agent B: "Thanks." Agent A: "You're welcome." Agent B: "Have a nice day."
*   **Solution:**
    *   **Max Turns:** Hard limit (e.g., 10 turns).
    *   **Termination Condition:** Instruct agents to output "TERMINATE" when the goal is achieved. The orchestrator listens for this string to stop the loop.

### Production Challenges

#### Challenge 1: The "Lazy" Manager

**Scenario:** The Manager agent just forwards the user's prompt to the Worker without breaking it down or adding context.
**Root Cause:** Weak system prompt or weak model.
**Solution:**
*   **Prompt Engineering:** "You are a Manager. You MUST break the task into at least 3 sub-tasks."
*   **Few-Shot:** Provide examples of good delegation.

#### Challenge 2: Lost in Translation (Handoff Failure)

**Scenario:** Triage Agent hands off to Tech Agent, but forgets to pass the User's User ID or the specific error message. Tech Agent asks "Who are you?"
**Root Cause:** Stateless handoff.
**Solution:**
*   **Structured Handoff:** The handoff output must be a JSON object containing `{"target_agent": "Tech", "context": {...}}`. The system ensures this context is injected into the target agent's memory.

#### Challenge 3: Cost Explosion

**Scenario:** A 5-agent swarm runs for 20 turns. 100 API calls. Cost: $5.00 per user query.
**Root Cause:** Uncontrolled chatter.
**Solution:**
*   **Cheaper Models:** Use GPT-4 for the Manager, but GPT-3.5/Haiku for the Workers.
*   **Single-Turn Tools:** Instead of a chat loop, treat workers as "Functions" that return a result in one turn.

#### Challenge 4: Inconsistent Personas

**Scenario:** The "Grumpy Critic" agent starts being nice after 5 turns.
**Root Cause:** Context dilution. The system prompt ("You are grumpy") is pushed out of the context window by recent messages.
**Solution:**
*   **System Prompt Reinforcement:** Re-inject the persona instructions at the end of the prompt (Recency Bias).

### System Design Scenario: Software Development Swarm

**Requirement:** Build a feature based on a ticket.
**Design:**
1.  **Product Manager (PM):** Reads ticket, writes Spec.
2.  **Architect:** Reads Spec, writes File Structure.
3.  **Coder:** Reads File Structure, writes Code.
4.  **Reviewer:** Reads Code, critiques.
5.  **Orchestration:** Sequential (PM -> Architect -> Coder -> Reviewer). If Reviewer rejects, loop back to Coder.

### Summary Checklist for Production
*   [ ] **Termination:** Implement strict termination conditions.
*   [ ] **Cost:** Monitor token usage per agent.
*   [ ] **Handoffs:** Ensure context is preserved during transfers.
*   [ ] **Models:** Mix and match models (Smart/Fast) to optimize ROI.
*   [ ] **Logging:** Log the conversation flow clearly (Who said what).
