# Day 44: Tree of Thoughts (ToT) & Graph of Thoughts (GoT)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the main bottleneck of Tree of Thoughts?

**Answer:**
**Cost and Latency.**
*   ToT might require 100+ LLM calls to solve one problem (Breadth * Depth * Evaluation).
*   It is too slow for real-time applications.
*   It is best used for offline tasks (e.g., generating training data, solving complex puzzles, strategic planning).

#### Q2: How does the "Evaluator" work in ToT?

**Answer:**
The Evaluator is often the LLM itself (Self-Evaluation).
*   *Prompt:* "Look at this partial solution. Is it possible to reach 24 from here? Answer 'Sure', 'Likely', or 'Impossible'."
*   *Challenge:* LLMs are often overconfident or bad at judging their own partial work.
*   *Fix:* Use a rigorous external verifier (e.g., a Python script) if possible, or train a specific Reward Model.

#### Q3: Difference between DFS and BFS in ToT?

**Answer:**
*   **DFS (Depth-First):** Explores one path to the end. Good if solutions are dense (easy to find). Memory efficient.
*   **BFS (Breadth-First):** Explores all options at current step. Good if solutions are rare or if you need the *optimal* (shortest) solution. Memory intensive.

#### Q4: Can ToT be done in a single prompt?

**Answer:**
Yes, **Algorithm of Thoughts (AoT)** attempts this.
*   *Prompt:* "I want you to simulate a DFS search. Try option A. If it fails, output [BACKTRACK] and try option B."
*   *Limit:* The context window limits how deep the tree can go. The Python-controlled ToT has infinite memory (external state).

### Production Challenges

#### Challenge 1: Infinite Loops

**Scenario:** The model generates the same thought over and over. "Try 4+9... Try 9+4..."
**Root Cause:** Lack of history awareness.
**Solution:**
*   **Deduplication:** The Controller must hash each state and prevent visiting the same state twice.
*   **Temperature:** Increase temperature slightly to encourage diversity.

#### Challenge 2: The "Bad Evaluator" Problem

**Scenario:** The Evaluator gives high scores to bad paths. The search explores garbage.
**Root Cause:** Evaluation is harder than generation.
**Solution:**
*   **Rubric:** Give the Evaluator a strict checklist.
*   **Consistency:** Ask the Evaluator 3 times and take the average score.

#### Challenge 3: Context Fragmentation

**Scenario:** In GoT, aggregating Node A and Node Z requires fitting both into the context.
**Root Cause:** Limited context window.
**Solution:**
*   **Hierarchical GoT:** Summarize A and Z individually before combining.
*   **RAG:** Retrieve only the relevant parts of A and Z.

### System Design Scenario: Novel Writing Assistant

**Requirement:** Write a coherent 50-page mystery novel.
**Design:**
1.  **ToT for Plotting:**
    *   Root: "Murder at the Manor".
    *   Branches: 3 different killers.
    *   Depth: Clues, Red Herrings, Reveal.
    *   *Selection:* User picks the best plot outline.
2.  **GoT for Character Arcs:**
    *   Node: Character A's timeline.
    *   Node: Character B's timeline.
    *   Edge: Interaction/Conflict.
    *   *Check:* Ensure consistency (A can't be in two places).
3.  **CoT for Drafting:**
    *   Write Chapter 1 based on the plan.

### Summary Checklist for Production
*   [ ] **Budget:** Set a max number of tokens/calls per problem.
*   [ ] **Pruning:** Aggressively prune low-score branches.
*   [ ] **Cache:** Cache thought generation to save money on retries.
