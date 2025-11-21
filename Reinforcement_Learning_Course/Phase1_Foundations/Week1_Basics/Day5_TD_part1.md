# Day 5 Deep Dive: TD Learning Nuances

## 1. Bias-Variance Tradeoff
*   **Monte Carlo (MC):**
    *   **Bias:** Zero (unbiased estimate of expected return).
    *   **Variance:** High (depends on random outcomes of entire episode).
*   **TD Learning:**
    *   **Bias:** Non-zero (bootstraps from current biased estimates). Bias reduces over time as estimates improve.
    *   **Variance:** Low (depends only on one step of randomness: $R_{t+1}$ and $S_{t+1}$).
*   **Result:** TD usually converges faster than MC because low variance allows for more stable updates, even if they are slightly biased initially.

## 2. Convergence Properties
*   **TD(0):** Converges to $V_{\pi}$ for any fixed policy $\pi$ if the step-size parameter $\alpha$ decreases appropriately (Robbins-Monro conditions: $\sum \alpha = \infty, \sum \alpha^2 < \infty$).
*   **Q-Learning:** Converges to $Q_*$ with probability 1, assuming all state-action pairs are visited infinitely often and step-sizes decay.
*   **SARSA:** Converges to $Q_{\pi}$ (where $\pi$ is the policy being followed). If $\pi$ becomes greedy over time (GLIE), SARSA converges to $Q_*$.

## 3. Maximization Bias
Q-Learning uses the `max` operator: $R + \gamma \max_a Q(S', a)$.
*   **Problem:** $\max$ is a convex function. $\mathbb{E}[\max(X_1, X_2)] \ge \max(\mathbb{E}[X_1], \mathbb{E}[X_2])$.
*   **Result:** Q-Learning tends to **overestimate** Q-values because it treats noise as signal (picking the "lucky" high value).
*   **Solution:** Double Q-Learning (covered in Day 12) uses two separate Q-tables to decouple selection and evaluation.

## 4. n-Step TD
We can bridge the gap between TD(0) (1-step) and MC ($\infty$-step) using n-step TD.
$$ G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}) $$
This allows us to trade off bias and variance by tuning $n$.
