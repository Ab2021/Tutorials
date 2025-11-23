# Day 77: Agent Evaluation & Simulation
## Core Concepts & Theory

### The Evaluation Crisis

**Problem:** How do you know if your agent is "good"?
- **Unit Tests:** Work for code, but agents are non-deterministic.
- **Vibe Check:** "It feels smarter" is not a metric.
- **Benchmarks:** MMLU tests knowledge, not *agency* (tool use, planning).

### 1. Agent Benchmarks

**AgentBench:**
- Comprehensive framework evaluating agents on 8 environments (OS, DB, Knowledge Graph, etc.).

**GAIA (General AI Assistants):**
- Focuses on tasks that are conceptually simple for humans but hard for AI (e.g., "Find the cheapest flight to Paris next Tuesday").
- **Metric:** Success Rate.

**SWE-bench:**
- Evaluates coding agents on real GitHub issues.
- **Metric:** Pass Rate on unit tests.

### 2. Simulation Environments

**Concept:**
- Test agents in a sandbox before deploying to production.
- **WebArena:** Simulated web browsing environment.
- **OSWorld:** Simulated Operating System control.

**Simulacra:**
- Simulating *users*.
- Create a "User Agent" with a specific persona and goal ("I am an angry customer").
- Have the "User Agent" talk to the "Support Agent".

### 3. LLM-as-a-Judge for Agents

**Concept:**
- Use a stronger model (GPT-4) to grade the trajectory of a weaker model.
- **Criteria:**
  - **Efficiency:** Did it take too many steps?
  - **Safety:** Did it try to delete files?
  - **Correctness:** Did it solve the user request?

### 4. Trajectory Evaluation

**Concept:**
- Don't just evaluate the final answer. Evaluate the *path* taken.
- **Good Path:** Search -> Read -> Answer.
- **Bad Path:** Search -> Search -> Search -> Give Up.

### 5. Evals as Code

**Frameworks:**
- **DeepEval / Ragas:** Python libraries for defining LLM test cases.
- **LangSmith:** Trace-based evaluation.

### 6. Contamination

**Risk:**
- If your agent was trained on the benchmark questions (e.g., StackOverflow), it memorized the answer.
- **Solution:** Use private, dynamic evaluation sets.

### 7. Unit Testing Agents

**Deterministic Mocks:**
- Mock the tools.
- "If agent calls `get_weather`, return `20C`."
- Verify the agent handles the `20C` correctly.

### 8. Summary

**Evaluation Strategy:**
1.  **Unit Tests:** Mock tools to test logic.
2.  **Integration Tests:** Run against a **Simulator**.
3.  **Benchmarks:** Run on **GAIA** or **SWE-bench**.
4.  **Online Eval:** Monitor **User Feedback** and **Step Count**.
5.  **Judge:** Use **GPT-4** to grade trajectories.

### Next Steps
In the Deep Dive, we will implement a simple Agent Eval Harness, a User Simulator, and a Trajectory Scorer.
