# Day 23 Deep Dive: PPO Variants and Hyperparameters

## 1. PPO-Penalty vs. PPO-Clip
There are two versions of PPO:
*   **PPO-Penalty:** Adds a KL divergence penalty to the loss.
    $$ L(\theta) = \mathbb{E}[r_t(\theta) A_t - \beta \cdot D_{KL}(\pi_{\theta_{old}} || \pi_\theta)] $$
    *   $\beta$ is adaptive (increases if KL is too large, decreases if too small).
    *   More complex to tune.
*   **PPO-Clip:** Uses the clipped objective (most popular).
    $$ L(\theta) = \mathbb{E}[\min(r_t(\theta) A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)] $$
    *   Simpler, no adaptive hyperparameters.
    *   Works better in practice.

## 2. Critical Hyperparameters
*   **Clip Epsilon ($\epsilon$):** Typically 0.1-0.3. Smaller = more conservative updates.
*   **Epochs:** Number of gradient steps per batch of data. Usually 3-10.
*   **Mini-batch Size:** Divide the collected batch into mini-batches for SGD. Typically 64-256.
*   **GAE Lambda ($\lambda$):** 0.95-0.99 for advantage estimation.
*   **Discount ($\gamma$):** 0.99 for most tasks.
*   **Learning Rate:** 2.5e-4 to 3e-4 (with annealing).

## 3. Value Function Clipping
Some implementations also clip the value function loss to prevent large updates:
$$ L^{V,CLIP} = \max((V - V_{target})^2, (V_{clip} - V_{target})^2) $$
where $V_{clip} = V_{old} + \text{clip}(V - V_{old}, -\epsilon, \epsilon)$.
This is less common but can improve stability.

## 4. PPO vs. TRPO
*   **TRPO (Trust Region Policy Optimization):** Solves a constrained optimization problem to enforce the KL constraint exactly. Requires second-order optimization (Fisher Information Matrix).
*   **PPO:** First-order approximation of TRPO. Easier to implement, faster, and works just as well.
*   PPO is the practical choice for most applications.
