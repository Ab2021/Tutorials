# Day 36: Agentic AI Fundamentals
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between a chatbot and an agent?

**Answer:**
- **Chatbot:** Responds to user queries. Passive. Cannot take actions beyond generating text.
- **Agent:** Autonomous system that can perceive, reason, plan, and act. Can use tools, execute code, query databases.
- **Example:** ChatGPT (chatbot) vs. ChatGPT with Plugins (agent).

#### Q2: Explain the ReAct pattern.

**Answer:**
- **ReAct:** Reasoning + Acting. The agent alternates between thinking (reasoning) and doing (acting).
- **Loop:** Thought -> Action -> Observation -> Thought -> ...
- **Benefit:** Makes the agent's reasoning explicit and interpretable. Allows for error correction.
- **Example:** "Thought: I need to find X. Action: search[X]. Observation: Result. Thought: Now I can answer."

#### Q3: What are the main challenges in building reliable agents?

**Answer:**
- **Hallucination:** Agent might generate incorrect tool calls or reasoning.
- **Error Propagation:** One wrong action can derail the entire plan.
- **Cost:** Multiple LLM calls per task (expensive).
- **Latency:** Sequential tool calls add delay.
- **Safety:** Agent might take harmful actions if not properly constrained.

#### Q4: How do you implement memory in an LLM agent?

**Answer:**
- **Short-Term:** Include recent conversation turns in the prompt (sliding window).
- **Long-Term:** Store past experiences in a vector database. Retrieve relevant memories based on similarity to the current query.
- **Working Memory:** Use a "scratchpad" section in the prompt for intermediate reasoning.

#### Q5: What is the difference between zero-shot and few-shot planning?

**Answer:**
- **Zero-Shot:** Agent plans without examples. Relies on the model's pre-trained knowledge.
- **Few-Shot:** Provide examples of task decomposition in the prompt. Agent learns the pattern and applies it to new tasks.
- **Trade-off:** Few-shot is more reliable but uses more tokens.

---

### Production Challenges

#### Challenge 1: Agent Gets Stuck in Loops

**Scenario:** Your agent keeps calling the same tool repeatedly without making progress.
**Example:** `search["X"]` -> No results -> `search["X"]` -> No results -> ...
**Solution:**
- **Max Iterations:** Limit the number of steps (e.g., 10).
- **Reflection:** After each failure, prompt the agent to reflect and try a different approach.
- **State Tracking:** Detect repeated states and force a different action.

#### Challenge 2: Tool Call Parsing Errors

**Scenario:** The agent generates `search[Paris weather` (missing closing bracket). Your parser fails.
**Solution:**
- **Robust Parsing:** Use regex with error handling. If parsing fails, prompt the agent to retry.
- **Structured Output:** Use OpenAI's function calling API instead of free-form text.
- **Validation:** Check that all required parameters are present before executing.

#### Challenge 3: Cost Explosion

**Scenario:** A simple task requires 20 LLM calls. Your costs skyrocket.
**Solution:**
- **Caching:** Cache tool results. If the same query appears, reuse the result.
- **Smaller Models:** Use GPT-3.5 for simple reasoning, GPT-4 only for complex tasks.
- **Batch Processing:** Group multiple queries and process them together.

#### Challenge 4: Safety and Sandboxing

**Scenario:** Your agent has access to a `run_code` tool. A user asks it to delete files.
**Solution:**
- **Sandboxing:** Run code in an isolated environment (Docker container, VM).
- **Approval Gates:** Require human approval for dangerous actions (delete, network requests).
- **Input Validation:** Block dangerous commands (rm -rf, DROP TABLE).

#### Challenge 5: Evaluating Agent Performance

**Scenario:** How do you measure if your agent is "good"?
**Metrics:**
- **Task Success Rate:** % of tasks completed successfully.
- **Efficiency:** Average number of steps to complete a task.
- **Cost:** Average cost per task (number of LLM calls).
- **Latency:** Average time to complete a task.
**Benchmark:** Create a test set of tasks with ground-truth solutions. Measure success rate.

### Summary Checklist for Production
- [ ] **Max Iterations:** Set a limit (e.g., **10 steps**).
- [ ] **Reflection:** Add **self-critique** after failures.
- [ ] **Parsing:** Use **structured output** (function calling).
- [ ] **Caching:** Cache **tool results**.
- [ ] **Safety:** **Sandbox** dangerous tools.
- [ ] **Monitoring:** Track **success rate** and **cost**.
