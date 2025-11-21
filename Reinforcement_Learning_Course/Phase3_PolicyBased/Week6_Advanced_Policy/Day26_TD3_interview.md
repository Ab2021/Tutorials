# Day 26 Interview Questions: TD3

## Q1: What are the three improvements in TD3 over DDPG?
**Answer:**
1. **Clipped Double Q-Learning:** Uses two critics, takes the minimum for the target to reduce overestimation.
2. **Delayed Policy Updates:** Updates the actor less frequently than the critics (e.g., every 2 critic updates) to avoid overfitting to noisy Q-values.
3. **Target Policy Smoothing:** Adds clipped noise to the target action to make Q-values robust to perturbations.

## Q2: Why does TD3 use the minimum of two Q-values?
**Answer:**
Q-learning tends to **overestimate** values due to the max operator in the Bellman equation.
By maintaining two independent critics and taking the minimum:
$$ y = r + \gamma \min(Q'_1(s', a'), Q'_2(s', a')) $$
we get a **conservative (pessimistic) estimate** that reduces overestimation bias.
This is similar to Double DQN but uses two networks instead of decoupling selection and evaluation.

## Q3: Why delay policy updates in TD3?
**Answer:**
The actor's gradient depends on the critic's Q-values:
$$ \nabla_\theta J = \nabla_a Q(s, a)|_{a=\mu(s)} \nabla_\theta \mu(s) $$
If the critic's estimates are noisy (which they are early in training or immediately after an update), the actor will learn a poor policy.
**Solution:** Update the critic multiple times to stabilize Q-estimates before updating the actor.
Typical: Update actor every 2 critic updates.

## Q4: What is Target Policy Smoothing?
**Answer:**
Adding noise to the target action:
$$ a' = \mu'(s') + \text{clip}(\epsilon, -c, c), \quad \epsilon \sim \mathcal{N}(0, \sigma) $$
**Why?** A deterministic target policy can create sharp, narrow peaks in the Q-function. If the Q-function overfits to these specific actions, small errors can cause large value overestimates.
Adding noise "smooths out" the Q-function, making it more robust.

## Q5: How does TD3 compare to DDPG and SAC?
**Answer:**
| Feature | DDPG | TD3 | SAC |
|---------|------|-----|-----|
| **Stability** | Low | High | Very High |
| **Sample Efficiency** | Moderate | Good | Excellent |
| **Complexity** | Low | Medium | High |
| **Policy Type** | Deterministic | Deterministic | Stochastic |
| **Exploration** | Added Noise | Added Noise | Entropy Max |

*   **TD3** is a strict improvement over DDPG with minimal added complexity.
*   **SAC** is more sample-efficient but requires tuning entropy temperature.

## Q6: What are the typical hyperparameters for TD3?
**Answer:**
*   **Policy Update Frequency:** 2 (update actor every 2 critic updates).
*   **Target Policy Noise:** $\sigma = 0.2$.
*   **Noise Clip:** $c = 0.5$.
*   **Discount Factor:** $\gamma = 0.99$.
*   **Learning Rate:** $3 \times 10^{-4}$ for both actor and critics.
*   **Soft Update Rate:** $\tau = 0.005$.
*   **Batch Size:** 256.
