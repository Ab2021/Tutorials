# Day 54: Evaluation of Tool-Using Agents
## Core Concepts & Theory

### Why is Evaluating Agents Hard?

Evaluating a chatbot is hard. Evaluating an agent that *does things* is harder.
*   **Non-Determinism:** Agents might take different paths to the same solution.
*   **Side Effects:** You can't just run an agent that deletes files 100 times to test it.
*   **Multi-Step Reasoning:** If the agent fails at step 5, was it because of a bad retrieval at step 1 or a logic error at step 4?

### 1. The Evaluation Hierarchy

We evaluate agents at three levels:

1.  **Unit Testing (Tools):** Does the tool work? (Deterministic).
2.  **Trajectory Evaluation (Process):** Did the agent choose the right tools in the right order?
3.  **End-to-End Evaluation (Outcome):** Did the agent solve the user's problem?

### 2. Trajectory Evaluation

A **Trajectory** is the sequence of `(Thought, Action, Observation)` tuples.
We use **LLM-as-a-Judge** to grade this trajectory.
*   *Criteria:*
    *   **Efficiency:** Did it loop unnecessarily?
    *   **Tool Selection:** Did it use `search` when it should have used `calculator`?
    *   **Hallucination:** Did it invent tool arguments?

### 3. Benchmarks

*   **AgentBench:** A comprehensive framework for evaluating agents across different environments (OS, DB, Knowledge Graph).
*   **ToolBench:** Specifically focuses on instruction following for tool use.
*   **MMLU:** General knowledge (less relevant for agents).
*   **HotpotQA:** Multi-hop reasoning (relevant for RAG agents).

### 4. Simulation & Sandboxing

To evaluate agents safely, we need **Simulators**.
*   **WebArena:** A simulated web environment for testing browser agents.
*   **Mock APIs:** Instead of calling the real Stripe API, call a mock that returns deterministic responses.
*   **VCR (Record/Replay):** Record the interactions of a successful run and replay them to test for regressions.

### 5. Metrics

*   **Success Rate (SR):** % of tasks completed.
*   **Pass@K:** Probability of success if allowed K attempts.
*   **Steps to Solution:** Average number of turns. Lower is usually better.
*   **Tool Error Rate:** % of tool calls that raised an exception.

### 6. The "Golden Dataset"

You cannot evaluate without ground truth.
*   **Dataset:** A list of `(Goal, Expected_Result)` pairs.
*   *Example:* `("What is 2+2?", "4")`, `("Book a meeting with John", "Meeting booked")`.
*   **Creation:** Use human annotators or GPT-4 to generate synthetic test cases.

### 7. RAGAS for Agents

RAGAS (Retrieval Augmented Generation Assessment) is standard for RAG. For agents, we extend it:
*   **Tool Selection Accuracy:** (Selected Tools / Optimal Tools).
*   **Argument Quality:** Did the arguments match the schema?

### Summary

Evaluation is the difference between a demo and a product. For agents, "it looks good" is not enough. We need rigorous, automated pipelines that test the agent's reasoning, tool use, and safety in a controlled environment.
