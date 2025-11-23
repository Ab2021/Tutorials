# Day 43: Chain of Thought (CoT) & Self-Consistency
## Core Concepts & Theory

### The Reasoning Gap

Standard LLMs (Zero-Shot) are good at pattern matching but bad at multi-step logic.
*   *Prompt:* "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many does he have?"
*   *Zero-Shot:* "Roger has 11 balls." (Often correct, but fragile).
*   *Complex Math:* Fails easily.

### 1. Chain of Thought (CoT)

Proposed by Wei et al. (2022).
**Idea:** Force the model to generate intermediate reasoning steps before the final answer.
*   *Prompt:* "Roger has 5 balls. 2 cans * 3 balls/can = 6 balls. 5 + 6 = 11. Answer: 11."
*   *Mechanism:* The reasoning steps act as a "scratchpad". The model attends to its own previous tokens to guide the next step. It reduces the computational burden of doing everything in one forward pass.

### 2. Zero-Shot CoT

"Let's think step by step."
*   Proposed by Kojima et al. (2022).
*   Just adding this magic phrase triggers the model to generate its own reasoning chain without few-shot examples.

### 3. Self-Consistency

Proposed by Wang et al. (2022).
**Idea:** CoT is greedy (single path). It might hallucinate a step.
**Algorithm:**
1.  Sample `k` different CoT paths (using high temperature).
2.  Extract the final answer from each.
3.  **Majority Vote:** The most common answer wins.
*   *Analogy:* Asking 10 experts to think through a problem and taking the consensus.

### 4. Least-to-Most Prompting

Break the problem down explicitly.
1.  **Decomposition:** "To solve X, I need to solve Y and Z first."
2.  **Sequential Solving:** Solve Y. Use answer to solve Z. Use answer to solve X.

### Summary

Reasoning is not "Magic"; it is **Compute**. CoT trades more inference tokens (time/cost) for higher accuracy. It is the foundation of all Agentic AI.
