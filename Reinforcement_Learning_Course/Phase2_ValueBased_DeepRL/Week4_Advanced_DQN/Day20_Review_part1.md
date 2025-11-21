# Day 20 Deep Dive: Why we need Policy Gradients

## 1. The Limits of Value-Based Methods
We have spent 10 days on Value-Based RL (DQN and friends). They are powerful but have limitations:
1.  **Continuous Action Spaces:** As seen in Day 17 (NAF), handling continuous actions requires hacks (quadratic assumption) or expensive optimization.
2.  **Stochastic Policies:** Value-based methods are inherently deterministic ($a = \arg\max Q$). They cannot learn optimal stochastic policies (e.g., Rock-Paper-Scissors).
3.  **Stability:** Minimizing MSBE (Bellman Error) is not the same as maximizing Reward. Small errors in Q-values can lead to bad policies.

## 2. The Policy Gradient Theorem
Instead of learning $Q(s, a)$ and deriving $\pi$, why not learn $\pi_\theta(a|s)$ directly?
$$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] $$
The gradient is:
$$ \nabla_\theta J(\theta) = \mathbb{E} [\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s, a)] $$
This is the foundation of Phase 3 (Policy-Based Methods).

## 3. Preview of Phase 3
*   **REINFORCE:** The simplest PG algorithm.
*   **Actor-Critic:** Combining PG with Value Learning (Best of both worlds).
*   **PPO (Proximal Policy Optimization):** The industry standard (used by OpenAI, DeepMind).
*   **SAC (Soft Actor-Critic):** The state-of-the-art for continuous control.
