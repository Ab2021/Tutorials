# Day 6 Deep Dive: Advanced Eligibility Traces

## 1. Watkin's Q($\lambda$)
Standard TD($\lambda$) is on-policy (like SARSA). How do we combine traces with Q-Learning (off-policy)?
**Watkin's Q($\lambda$)** handles this:
*   Behave according to $\mu$ (e.g., $\epsilon$-greedy).
*   Update traces normally as long as we take the greedy action.
*   **Cut the trace:** If we take a *non-greedy* (exploratory) action, we set all traces to zero ($E_t(s) = 0$).
*   **Reason:** The lookahead into the future is broken because we deviated from the target policy (greedy). We can't backup rewards from a non-greedy path to update the greedy policy's value.

## 2. Accumulating vs. Replacing Traces
*   **Accumulating Trace:** $E_t(s) \leftarrow E_t(s) + 1$.
    *   If a state is visited frequently, the trace can grow very large (potentially $> 1/(1-\gamma\lambda)$).
    *   Can cause instability.
*   **Replacing Trace:** $E_t(s) \leftarrow 1$.
    *   Resets the trace to 1 regardless of current value.
    *   Usually performs better and is more stable.

## 3. True Online TD($\lambda$)
Traditional TD($\lambda$) is an approximation of the "offline" $\lambda$-return (updating at the end of the episode).
**True Online TD($\lambda$)** is an exact online implementation that matches the offline equivalence perfectly. It's slightly more complex but generally superior.

## 4. Dutch Traces
A variant used in modern linear RL (like in True Online TD).
$$ E_t(s) \leftarrow \gamma \lambda E_{t-1}(s) + (1 - \alpha \gamma \lambda E_{t-1}(s) \phi(s)^T \phi(s)) \phi(s) $$
(Don't worry too much about the math, just know that "traces" have evolved).
