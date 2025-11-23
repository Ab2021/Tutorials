# Day 61: Consensus Protocols (Voting, Debate)
## Core Concepts & Theory

### The "Wisdom of Crowds" (for Agents)

LLMs hallucinate. But different LLMs (or the same LLM with different personas) tend to hallucinate *differently*.
**Consensus Protocols** leverage this by aggregating multiple opinions to reach a higher-confidence truth.

### 1. Voting Mechanisms

*   **Majority Vote:** 3 agents generate an answer. If 2 say "A" and 1 says "B", pick "A".
*   **Weighted Vote:** The "Senior Engineer" agent's vote counts for 2x.
*   **Unanimity:** All agents must agree (Critical for safety/security).

### 2. Debate (Multi-Turn Consensus)

Voting is static. Debate is dynamic.
*   **Round 1:** Agent A says "X". Agent B says "Y".
*   **Round 2:** Agent A reads "Y", critiques it, and maybe updates to "X'". Agent B reads "X", critiques it.
*   **Convergence:** After N rounds, agents usually converge on a single answer that is better than either started with.

### 3. Roles in Consensus

*   **Proposer:** Generates the initial solution.
*   **Critique:** Checks for errors.
*   **Judge/Moderator:** Decides when the debate is over and selects the winner.

### 4. Self-Consistency (MoE - Mixture of Experts)

You don't need multiple *agents*. You can just sample the same model 5 times (Temperature > 0.7).
*   **Sample-and-Marginalize:** Generate 5 reasoning paths. Pick the most common final answer.
*   **Why it works:** There are many paths to the right answer, but usually only one "right answer". Hallucinations are scattered.

### 5. Verification (The "Checker" Pattern)

*   **Generator:** "The answer is 42."
*   **Verifier:** "Is the answer 42? Let me check."
*   **Refiner:** "The Verifier said 42 is wrong. I will try again."

### 6. Challenges

*   **Groupthink:** Agents might agree on a wrong answer if the bias is strong in the base model.
*   **Cost:** 5 agents = 5x cost.
*   **Latency:** Debate takes time.

### Summary

Consensus protocols trade **Compute for Accuracy**. They are essential for high-stakes domains (Medical, Legal, Code) where a single hallucination is unacceptable.
