# Day 47: Reflection & Self-Correction (Reflexion)
## Core Concepts & Theory

### The Learning Loop

Humans learn from mistakes. Standard LLM Agents (ReAct) do not. If they fail, they fail.
**Reflexion** (Shinn et al., 2023) adds a "Verbal Reinforcement Learning" loop.
Instead of updating weights (which is expensive), the agent updates its **Memory** (Context).

### 1. The Reflexion Architecture

1.  **Actor:** Tries to solve the task (e.g., ReAct).
2.  **Evaluator:** Checks if the solution is correct (Test cases, Reward Model).
3.  **Self-Reflection:** If failed, the agent analyzes *why*. "I failed because I used the wrong tool."
4.  **Memory:** The reflection is stored.
5.  **Retry:** The Actor tries again, but this time the Prompt includes the Reflection.

### 2. Verbal Reinforcement

*   **Standard RL:** Update weights $\theta$ to maximize reward.
*   **Verbal RL:** Update context $C$ (Reflections) to maximize reward.
*   It turns "Optimization" into "Prompt Engineering".

### 3. Self-Correction

Even without a full loop, agents can self-correct in real-time.
*   **Critique-and-Refine:**
    *   Generate Draft.
    *   Critique Draft ("Too verbose").
    *   Refine Draft.
*   **Constitutional AI:** "Critique this response based on the rule: Be Harmless."

### 4. Recursive Criticism and Improvement (RCI)

A prompting strategy:
1.  Generate Output.
2.  Critique Output.
3.  Improve Output based on Critique.
Repeat $N$ times.

### Summary

Reflexion allows agents to improve over time (within an episode or across episodes) without fine-tuning. It is the key to **Robustness**.
