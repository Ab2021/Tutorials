# Day 34: Safe Reinforcement Learning

## 1. The Safety Problem
Standard RL maximizes reward without considering **safety constraints**.
**Problems:**
*   Dangerous exploration (robotics, autonomous vehicles).
*   Violating constraints (budget, ethical guidelines).
*   Unintended behaviors (negative side effects).

## 2. Constrained RL
Formalize safety as **constraints**:
$$ \max_\pi \mathbb{E}[R] \quad \text{s.t.} \quad \mathbb{E}[C_i] \leq d_i $$
where $C_i$ are cost functions (e.g., collision risk, energy usage).

## 3. Safe Exploration
*   **Safety Layer:** Filter unsafe actions before execution.
*   **Model-Based Safe RL:** Use a learned model to predict constraint violations.
*   **Conservative Policy Updates:** PPO with additional safety constraints.

## 4. Constrained Policy Optimization (CPO)
Extends TRPO with safety constraints:
$$ \max_\pi J(\pi) \quad \text{s.t.} \quad D_{KL}(\pi_{old} || \pi) \leq \delta, \quad C(\pi) \leq d $$
*   Guarantees constraint satisfaction during training.

## 5. Reward Design for Safety
*   **Negative Rewards for Violations:** Penalize unsafe behavior.
*   **Shaping:** Guide agent away from unsafe regions.
*   **Inverse RL:** Learn safe rewards from expert demonstrations.

### Key Takeaways
*   Safe RL ensures agents respect constraints during learning and execution.
*   Constrained RL formalizes safety as optimization constraints.
*   Critical for real-world deployment (robotics, healthcare, autonomous systems).
