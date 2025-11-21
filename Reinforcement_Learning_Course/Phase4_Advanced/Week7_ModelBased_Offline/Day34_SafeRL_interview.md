# Day 34 Interview Questions: Safe RL

## Q1: What is Safe Reinforcement Learning?
**Answer:**
Safe RL ensures agents respect safety constraints during learning and execution.
**Challenges:**
*   Dangerous exploration (can't crash a real car while learning).
*   Constraint violations (budget, ethical, safety limits).
*   Unintended behaviors (reward hacking, side effects).

## Q2: What is Constrained RL?
**Answer:**
Formalizes safety as optimization constraints:
$$ \max_\pi \mathbb{E}[R] \quad \text{s.t.} \quad \mathbb{E}[C_i] \leq d_i $$
where $C_i$ are cost functions (e.g., collision risk).
**Example:** CPO (Constrained Policy Optimization) extends TRPO with safety constraints.

## Q3: What is RLHF (Reinforcement Learning from Human Feedback)?
**Answer:**
RLHF aligns AI systems (e.g., ChatGPT) with human preferences:
1. **Supervised Fine-Tuning:** Train on human demonstrations.
2. **Reward Modeling:** Learn a reward model from human preference comparisons.
3. **RL Optimization:** Use PPO to optimize the policy w.r.t. the learned reward.

## Q4: How do you ensure safe exploration?
**Answer:**
*   **Safety Layer:** Filter unsafe actions before execution using a model or rules.
*   **Conservative Updates:** Use small policy updates (PPO, TRPO) to avoid drastic behavior changes.
*   **Model-Based Prediction:** Use a learned model to predict constraint violations before taking actions.
*   **Offline Pretraining:** Learn from safe data first, then fine-tune cautiously online.

## Q5: What is reward hacking?
**Answer:**
**Reward hacking:** The agent finds unintended ways to maximize the reward specification that violate the true intent.
**Example:** A cleaning robot pushes trash under the rug to appear clean.
**Solutions:**
*   Better reward design (include penalties for side effects).
*   Human in the loop (RLHF).
*   Inverse RL (learn rewards from demonstrations).
