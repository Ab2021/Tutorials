# Day 10 Deep Dive: The Art of RL Engineering

## 1. Debugging Reinforcement Learning
RL is notoriously hard to debug because a bug might not cause a crash; it just leads to sub-optimal behavior or slow convergence.

### Key Metrics to Monitor
1.  **Episode Reward:** Should trend upwards. If flat, check exploration or reward function.
2.  **Episode Length:** Should decrease (for goal-based tasks) or increase (for survival tasks).
3.  **Loss Function:**
    *   **DQN:** Should decrease but might oscillate.
    *   **Policy Gradient:** Does not necessarily decrease (we are maximizing reward, not minimizing loss directly in the same way as supervised learning).
4.  **Value Estimates ($V$ or $Q$):**
    *   Compare estimated $V(s)$ with actual returns $G_t$.
    *   If $V(s) \ll G_t$: Underestimation (rare in Q-learning).
    *   If $V(s) \gg G_t$: Overestimation (common in Q-learning).

## 2. Hyperparameter Tuning
RL is sensitive to hyperparameters.
*   **Learning Rate ($\alpha$):**
    *   Too high: Unstable, divergence.
    *   Too low: Slow convergence.
    *   Standard: $1e-3$ to $1e-4$ (Adam).
*   **Discount Factor ($\gamma$):**
    *   Standard: $0.99$.
    *   Short horizon: $0.90 - 0.95$.
    *   Long horizon: $0.995 - 0.999$.
*   **Exploration ($\epsilon$):**
    *   Linear decay (e.g., 1.0 to 0.01 over 10% of steps) is robust.

## 3. Sanity Checks
*   **Random Agent:** Does your agent beat a random agent?
*   **Oracle:** If you know the optimal policy, does your agent learn it?
*   **Simplified Env:** Try on a 2x2 GridWorld first. If it fails there, it will fail on Atari.
