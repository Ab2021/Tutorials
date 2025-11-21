# Day 4 Deep Dive: Advanced Monte Carlo Methods

## 1. First-Visit vs. Every-Visit MC
When an agent visits the same state $s$ multiple times in a single episode, how do we calculate the return?

### First-Visit MC
*   **Method:** Only count the return following the *first* time $s$ is visited in an episode.
*   **Properties:**
    *   **Unbiased:** The returns are i.i.d. samples of the true expectation.
    *   **Convergence:** Converges to $V_{\pi}(s)$ by the Law of Large Numbers.

### Every-Visit MC
*   **Method:** Count the return following *every* visit to $s$.
*   **Properties:**
    *   **Biased:** Returns within an episode are correlated (not i.i.d.).
    *   **Consistent:** The bias vanishes as the number of episodes $\to \infty$.
    *   **Variance:** Often has lower variance than First-Visit because it uses more data.

## 2. Off-Policy Learning & Importance Sampling
How can we learn about the optimal policy $\pi_*$ while behaving according to a different, exploratory policy $\mu$ (e.g., random)?
*   **Target Policy ($\pi$):** The policy we want to learn (usually greedy).
*   **Behavior Policy ($\mu$):** The policy used to generate data (usually $\epsilon$-greedy).

### Importance Sampling
To estimate $V_{\pi}$ using data from $\mu$, we must weight returns by the probability ratio:
$$ \rho_{t:T-1} = \frac{\prod_{k=t}^{T-1} \pi(A_k|S_k)}{\prod_{k=t}^{T-1} \mu(A_k|S_k)} $$
*   **Ordinary Importance Sampling:** Simple average of weighted returns. Unbiased but **infinite variance** (unstable).
*   **Weighted Importance Sampling:** Weighted average. Biased but consistent, and much lower variance.

## 3. Bias-Variance Tradeoff
*   **Monte Carlo:** Zero Bias (estimates converge to true value), High Variance (sum of many random rewards).
*   **Dynamic Programming:** Bias (bootstraps from estimates), Zero Variance (in the update step itself, though values change).
*   **TD Learning (Day 5):** Will sit in the middle.
