# Day 21 Deep Dive: Taming the Variance

## 1. The Variance Problem
The gradient estimate in REINFORCE is unbiased but has **high variance**.
$$ \nabla J \approx \nabla \log \pi(a|s) G_t $$
*   $G_t$ depends on the entire trajectory (hundreds of stochastic steps).
*   If $G_t$ varies wildly (e.g., sometimes +100, sometimes -100), the gradient will swing wildly.
*   This forces us to use very small learning rates, making training slow.

## 2. Baselines
We can subtract a **baseline** $b(s)$ from the return without changing the expected gradient.
$$ \nabla J = \mathbb{E} [\nabla \log \pi(a|s) (G_t - b(s))] $$
*   **Proof:** $\mathbb{E} [\nabla \log \pi(a|s) b(s)] = b(s) \sum \nabla \pi(a|s) = b(s) \nabla \sum \pi(a|s) = b(s) \nabla (1) = 0$.
*   **Optimal Baseline:** The value function $V(s)$ is a good choice for $b(s)$.
    *   If $G_t > V(s)$: The action was better than expected. Increase prob.
    *   If $G_t < V(s)$: The action was worse than expected. Decrease prob.
    *   This term $(G_t - V(s))$ is called the **Advantage**.

## 3. Continuous Action Spaces
Policy Gradients shine in robotics.
We usually parameterize the policy as a Gaussian distribution:
$$ \pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)) $$
*   The network outputs the mean $\mu$ and (log) std dev $\sigma$.
*   Action $a = \mu + \sigma \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$.
*   The log-probability is differentiable with respect to $\mu$ and $\sigma$.

## 4. Deterministic Policy Gradient (DPG)
Standard PG learns a stochastic policy $\pi(a|s)$.
DPG (Silver et al., 2014) learns a deterministic policy $\mu_\theta(s)$.
$$ \nabla J = \mathbb{E} [\nabla_a Q(s, a) |_{a=\mu(s)} \nabla_\theta \mu_\theta(s)] $$
*   This is the core of **DDPG** (Deep Deterministic Policy Gradient).
*   It is more sample efficient because we don't need to integrate over the action space.
