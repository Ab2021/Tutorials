# Day 3 Deep Dive: Advanced Dynamic Programming

## 1. Generalized Policy Iteration (GPI)
Policy Iteration (PI) completely evaluates a policy before improving it. Value Iteration (VI) does just one step of evaluation before improving.
**GPI** is the general idea that we can interleave evaluation and improvement at any granularity.
*   We maintain a policy $\pi$ and a value function $V$.
*   We push $V$ towards $V_{\pi}$ (Evaluation).
*   We push $\pi$ to be greedy w.r.t $V$ (Improvement).
Eventually, they converge to the optimal $V_*$ and $\pi_*$.

## 2. Asynchronous DP
Standard DP algorithms require a full sweep over the state space (Synchronous DP). This is slow and requires two copies of the value function (old and new).
**Asynchronous DP** updates states in any order, using the most recent values available.
*   **In-Place DP:** Update $V(s)$ directly in memory. This usually converges faster because updated values are immediately used for neighbors.
*   **Prioritized Sweeping:** Update states with the largest **Bellman Error** ($|\text{NewVal} - \text{OldVal}|$) first. This focuses computation on "interesting" states where the value is changing rapidly.

## 3. The Curse of Dimensionality
DP methods are computationally expensive.
*   Complexity per iteration: $O(|S|^2 |A|)$ (or $O(|S||A|)$ if transitions are sparse).
*   If the state is defined by $d$ variables each with $k$ values, $|S| = k^d$.
*   Exponential growth makes DP intractable for complex problems (e.g., images, robotics).
*   **Solution:** Approximate DP (using function approximation like Neural Networks) or Sampling methods (Monte Carlo / TD).

## 4. Bootstrapping
DP methods **bootstrap**: they update estimates based on other estimates.
$$ V(s) \leftarrow R + \gamma V(s') $$
Here, $V(s)$ is updated using the current guess of $V(s')$. This is crucial for efficiency but can introduce bias if the initial estimates are wrong (though it washes out in the limit for tabular DP).
