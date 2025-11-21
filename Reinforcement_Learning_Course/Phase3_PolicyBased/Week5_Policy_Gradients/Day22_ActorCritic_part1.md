# Day 22 Deep Dive: GAE and Asynchronous Training

## 1. Generalized Advantage Estimation (GAE)
The standard advantage $A(s, a) = r + \gamma V(s') - V(s)$ (1-step TD) is biased but low variance.
The Monte Carlo advantage $A(s, a) = G_t - V(s)$ is unbiased but high variance.
**GAE** interpolates between them using $\lambda$.
$$ A^{GAE}(s_t, a_t) = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l} $$
where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.
*   $\lambda = 0$: Pure 1-step TD (low variance, high bias).
*   $\lambda = 1$: Monte Carlo (high variance, low bias).
*   Typically $\lambda = 0.95$ for a good tradeoff.

## 2. A3C: Asynchronous Advantage Actor-Critic
**Key Idea:** Run multiple agents in parallel, each with its own environment.
*   Each agent collects experience and computes gradients.
*   Gradients are asynchronously applied to a shared global network.
*   This breaks correlation (like a replay buffer, but for on-policy methods).

**Advantage over A2C:**
*   A2C is synchronous (wait for all workers).
*   A3C updates the global network asynchronously (faster, but noisier).
*   In practice, A2C is more stable and GPU-friendly.

## 3. Entropy Regularization
To encourage exploration, we add an entropy bonus to the loss:
$$ L_{total} = L_{actor} + c_1 L_{critic} - c_2 H(\pi) $$
where $H(\pi) = -\sum \pi(a|s) \log \pi(a|s)$.
*   High entropy = uniform distribution (exploration).
*   Low entropy = peaked distribution (exploitation).
*   Typically $c_2 = 0.01$.

## 4. Shared vs. Separate Networks
*   **Shared:** One network with two heads (Actor + Critic). Faster, fewer parameters, but can conflict.
*   **Separate:** Two networks. More stable, but doubles memory.
*   Most modern implementations use shared networks with careful tuning.
