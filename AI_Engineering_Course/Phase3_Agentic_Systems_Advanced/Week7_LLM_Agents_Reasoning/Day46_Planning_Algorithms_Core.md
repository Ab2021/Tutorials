# Day 46: Planning Algorithms (RAP, LLM+P)
## Core Concepts & Theory

### The Planning Problem

ReAct is "Greedy". It decides the next step based only on the current state. It doesn't look ahead.
For complex tasks (e.g., "Book a multi-city flight with constraints"), greedy agents get stuck.
**Planning** involves creating a sequence of actions *before* executing them.

### 1. LLM+P (LLM + Classical Planner)

Proposed by Liu et al. (2023).
LLMs are bad at long-horizon planning. Classical Planners (PDDL - Planning Domain Definition Language) are perfect at it (Solvers).
**Workflow:**
1.  **Translation:** LLM translates the user's natural language request into PDDL (Problem file).
2.  **Solving:** A classical solver (e.g., Fast Downward) finds the optimal plan.
3.  **Translation:** LLM translates the PDDL plan back into natural language or executable actions.

### 2. RAP (Reasoning via Planning)

Treats the LLM as both the World Model and the Planner.
*   **Simulation:** The LLM predicts the outcome of an action ("If I do X, what happens?").
*   **MCTS (Monte Carlo Tree Search):** Explore the tree of future states.
    *   Select -> Expand -> Simulate -> Backpropagate.
*   This allows the agent to "imagine" failures before they happen.

### 3. DEPS (Describe, Explain, Plan, Select)

A multi-step prompting strategy for planning.
1.  **Describe:** Describe the current state.
2.  **Explain:** Explain why the goal is not yet met.
3.  **Plan:** Generate multiple plans.
4.  **Select:** Choose the best one.

### 4. Hierarchical Planning

Break the goal into Sub-goals.
*   **High-Level Planner:** "Go to the kitchen."
*   **Low-Level Controller:** "Move forward 1m", "Turn left".
*   This abstraction reduces the search space.

### Summary

Planning bridges the gap between "Chatbots" and "Robots". It allows agents to solve tasks that require looking 10 steps ahead, avoiding dead ends that a greedy ReAct agent would fall into.
