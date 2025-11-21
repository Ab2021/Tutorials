# Day 23 Interview Questions: PPO

## Q1: What problem does PPO solve?
**Answer:**
Vanilla Policy Gradients can take destructively large policy updates if the learning rate is too high, leading to catastrophic performance collapse.
PPO constrains the policy update by clipping the probability ratio, ensuring that the new policy doesn't deviate too much from the old policy. This makes training more stable and sample-efficient.

## Q2: Explain the PPO clipped objective.
**Answer:**
PPO uses a clipped surrogate objective:
$$ L^{CLIP} = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)] $$
where $r_t = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ is the probability ratio.
*   If $A_t > 0$ (good action), we want to increase $\pi(a|s)$, but clip $r_t$ at $1 + \epsilon$ to prevent overshooting.
*   If $A_t < 0$ (bad action), we want to decrease $\pi(a|s)$, but clip $r_t$ at $1 - \epsilon$.
*   This constrains policy changes to a trust region.

## Q3: Why can PPO reuse data for multiple epochs?
**Answer:**
PPO uses **importance sampling** via the probability ratio $r_t = \frac{\pi_\theta}{\pi_{\theta_{old}}}$.
This allows us to train on data collected from an older policy, as long as the policies aren't too different (ensured by clipping).
We can perform multiple gradient updates (epochs) on the same batch of data, making PPO more sample-efficient than on-policy methods like A2C.

## Q4: What is the difference between PPO and TRPO?
**Answer:**
*   **TRPO:** Enforces a hard KL constraint using second-order optimization (conjugate gradient + Fisher Information Matrix). Theoretically sound but complex and slow.
*   **PPO:** Uses a first-order clipped objective to approximate the trust region. Simpler, faster, and works just as well in practice.
*   PPO is the modern standard because it's easier to implement and tune.

## Q5: What are typical hyperparameters for PPO?
**Answer:**
*   **Clip Epsilon ($\epsilon$):** 0.2 (range: 0.1-0.3).
*   **Epochs:** 10 (range: 3-10).
*   **Mini-batch Size:** 64 (range: 32-256).
*   **GAE Lambda ($\lambda$):** 0.95 (range: 0.9-0.99).
*   **Learning Rate:** 3e-4 (often with linear annealing).
*   **Discount ($\gamma$):** 0.99.

## Q6: Why is gradient clipping used in PPO?
**Answer:**
Even with the clipped objective, gradients can occasionally explode due to the ratio $r_t$.
Gradient clipping (e.g., `clip_grad_norm_(params, 0.5)`) limits the magnitude of gradients, preventing unstable updates.
This is especially important when training on diverse environments or with large networks.
