# Day 25 Interview Questions: DDPG

## Q1: Why can't we use DQN for continuous action spaces?
**Answer:**
DQN requires computing $\max_a Q(s, a)$ over all actions.
For discrete actions, we can evaluate $Q$ for each action and take the max.
For continuous actions (e.g., $a \in \mathbb{R}^d$), there are infinitely many actions. We cannot enumerate and evaluate all of them.
DDPG solves this by learning a deterministic policy $\mu(s)$ that directly outputs the action, and uses the policy gradient to optimize it.

## Q2: What is the Deterministic Policy Gradient theorem?
**Answer:**
The DPG theorem states that for a deterministic policy $\mu_\theta(s)$:
$$ \nabla_\theta J = \mathbb{E}[\nabla_a Q(s, a)|_{a=\mu(s)} \nabla_\theta \mu_\theta(s)] $$
**Intuition:**
*   $\nabla_a Q$ tells us how to change the action to increase value.
*   $\nabla_\theta \mu$ tells us how to change the parameters to produce that action.
*   Chain rule connects them.

## Q3: Why does DDPG use soft target updates instead of periodic hard updates?
**Answer:**
DDPG operates in continuous action spaces, where Q-values can change rapidly.
Periodic hard updates (like DQN) can cause instability.
**Soft updates** $\theta' \leftarrow \tau \theta + (1 - \tau) \theta'$ with small $\tau \approx 0.001$ make the target network change smoothly, reducing oscillations and improving stability.

## Q4: How does DDPG handle exploration?
**Answer:**
A deterministic policy has no inherent randomness, so it cannot explore.
DDPG adds **action noise** during training:
$$ a = \mu_\theta(s) + \mathcal{N}(0, \sigma^2) $$
*   Often uses **Ornstein-Uhlenbeck (OU) noise** for temporally correlated exploration (better for physical systems).
*   The noise variance $\sigma$ is decayed over time.
*   During evaluation, no noise is added.

## Q5: What are the main limitations of DDPG?
**Answer:**
1. **Overestimation Bias:** The critic overestimates Q-values (like DQN).
2. **Hyperparameter Sensitivity:** Very sensitive to learning rates, noise levels, and network architectures.
3. **Sample Efficiency:** Not as sample-efficient as more modern methods (TD3, SAC).
4. **Brittleness:** Can diverge or perform poorly with slight hyperparameter changes.

## Q6: How do TD3 and SAC improve upon DDPG?
**Answer:**
*   **TD3:**
    *   Uses **Clipped Double Q-Learning** (two critics, take min) to reduce overestimation.
    *   **Delayed policy updates** (update actor less frequently than critic).
    *   **Target policy smoothing** (add noise to target actions).
*   **SAC:**
    *   **Stochastic policy** instead of deterministic.
    *   **Entropy maximization** for automatic exploration.
    *   **Automatic temperature tuning** for the entropy coefficient.
    *   Much more stable and sample-efficient than DDPG.
