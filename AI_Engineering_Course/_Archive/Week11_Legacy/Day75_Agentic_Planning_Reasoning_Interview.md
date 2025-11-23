# Day 75: Agentic Planning & Reasoning
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the main advantage of Tree of Thoughts (ToT) over Chain of Thought (CoT)?

**Answer:**
- **CoT:** Linear. If the model makes a mistake in step 2, the whole chain is ruined.
- **ToT:** Explores multiple branches. If one branch looks bad (low evaluation score), it backtracks or prunes it and follows a better branch.
- **Trade-off:** ToT is much slower and more expensive (requires multiple calls per step).

#### Q2: Explain the "Reflexion" framework.

**Answer:**
- It uses verbal reinforcement learning.
- Instead of updating weights (which is hard/impossible via API), the agent updates its *context* (memory) with a text-based critique of its past failure.
- "I failed because I forgot to import numpy. Next time I must import numpy."

#### Q3: When should you use ReAct vs Plan-and-Solve?

**Answer:**
- **ReAct:** Best for environments where the state changes or is unknown. (e.g., Browsing the web. You don't know what's on the page until you click).
- **Plan-and-Solve:** Best for static tasks where the steps are known in advance. (e.g., "Write a blog post about X"). It avoids the overhead of the ReAct loop.

#### Q4: What is "Inference-Time Compute"?

**Answer:**
- The idea that spending more compute *during generation* (by thinking longer, searching trees, or verifying answers) yields better results than just training a larger model.
- **Example:** OpenAI o1 (Strawberry).

#### Q5: How do you evaluate a Planning Agent?

**Answer:**
- **Success Rate:** Did it achieve the goal?
- **Steps Taken:** Was it efficient? (10 steps vs 50 steps).
- **Cost:** How many tokens?
- **Benchmarks:** AgentBench, GAIA (General AI Assistants benchmark).

---

### Production Challenges

#### Challenge 1: The "Hallucinated Plan"

**Scenario:** Agent plans to "Use the TimeTravel tool" which doesn't exist.
**Root Cause:** Model doesn't know its own tools.
**Solution:**
- **RAG for Tools:** Inject available tool definitions into the system prompt.
- **Validation:** Check plan against valid tool list before execution.

#### Challenge 2: Getting Stuck in Details

**Scenario:** Agent spends 50 steps optimizing the font size of a chart instead of finishing the report.
**Root Cause:** Lack of high-level direction.
**Solution:**
- **Hierarchical Planning:** A "Manager" agent sets the goal ("Make a chart"), a "Worker" agent executes. The Manager interrupts if the Worker takes too long.

#### Challenge 3: Error Propagation

**Scenario:** Step 1 returns wrong data. Step 2 uses it. Step 10 fails.
**Root Cause:** No validation between steps.
**Solution:**
- **Unit Tests:** Agent writes a test for Step 1's output.
- **Human Review:** Pause after major steps.

#### Challenge 4: Latency of ToT

**Scenario:** Tree of Thoughts takes 5 minutes to generate a response. User leaves.
**Root Cause:** Exponential branching factor.
**Solution:**
- **Monte Carlo Tree Search (MCTS):** Smarter search than BFS.
- **Fast Evaluator:** Use a small model to score states.
- **Async:** Show partial progress.

#### Challenge 5: Infinite Loops in ReAct

**Scenario:** Thought -> Action -> Same Observation -> Thought -> Same Action...
**Root Cause:** Model doesn't realize it's repeating itself.
**Solution:**
- **History Check:** If `current_action in past_3_actions`, force a change.
- **Temperature:** Increase temperature to force diversity.

### System Design Scenario: Autonomous Coding Agent

**Requirement:** Given a GitHub issue, fix the bug and open a PR.
**Design:**
1.  **Explore:** ReAct loop to explore codebase and reproduce bug.
2.  **Plan:** Generate a plan ("Modify file X, Add test Y").
3.  **Code:** Write code.
4.  **Verify:** Run tests. If fail -> Reflexion loop ("Why did test fail?").
5.  **Submit:** Use GitHub API to open PR.

### Summary Checklist for Production
- [ ] **Method:** Start with **ReAct**, upgrade to **ToT** if needed.
- [ ] **Recovery:** Implement **Reflexion** for error handling.
- [ ] **Efficiency:** Use **Parallel Execution** for independent steps.
- [ ] **Safety:** Limit **Max Steps**.
- [ ] **Evaluation:** Use **Benchmarks** (GAIA) to measure improvement.
