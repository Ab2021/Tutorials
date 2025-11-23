# Day 44: Tree of Thoughts (ToT) & Graph of Thoughts (GoT)
## Core Concepts & Theory

### Beyond Linear Reasoning

CoT is linear (Step 1 -> Step 2 -> Step 3).
Real problem solving is non-linear. We explore branches, backtrack, and merge ideas.
*   "Let's try Approach A... dead end. Backtrack. Try Approach B."

### 1. Tree of Thoughts (ToT)

Proposed by Yao et al. (2023).
**Concept:** Treat reasoning as a search problem over a tree.
*   **Nodes:** Partial solutions (Thoughts).
*   **Edges:** Steps.
*   **Search Algorithm:** BFS (Breadth-First Search) or DFS (Depth-First Search).

**The Loop:**
1.  **Thought Decomposition:** Break the problem into steps.
2.  **Thought Generator:** Given the current state, generate $k$ possible next steps.
3.  **State Evaluator:** Score each new state (Value Function). "Is this promising?"
4.  **Search:** Keep the best states, discard the rest.

### 2. Graph of Thoughts (GoT)

Proposed by Besta et al. (2023).
**Concept:** Reasoning is a DAG (Directed Acyclic Graph).
*   **Aggregation:** You can combine three different thoughts into a new, better thought.
*   **Loops:** You can refine a thought recursively.
*   **Operations:** `Generate`, `Aggregate`, `Refine`, `KeepBest`.

### 3. Algorithm of Thoughts (AoT)

Instead of using an external Python script to manage the search (like ToT), AoT tries to teach the LLM to run the search algorithm *inside* its own context window.
*   *Prompt:* "Explore 3 paths. If path 1 fails, backtrack to start and try path 2."

### 4. Comparison

| Method | Topology | Use Case | Cost |
| :--- | :--- | :--- | :--- |
| **CoT** | Line | Simple Logic | Low |
| **CoT-SC** | Parallel Lines | Ambiguity | Medium |
| **ToT** | Tree | Creative Writing, Planning | High |
| **GoT** | Graph | Summarization, Sorting | Very High |

### Summary

ToT and GoT turn the LLM into a **Search Engine for Ideas**. They are essential for tasks like "Write a novel" or "Plan a marketing campaign" where there is no single right answer, but a vast space of possibilities to explore.
