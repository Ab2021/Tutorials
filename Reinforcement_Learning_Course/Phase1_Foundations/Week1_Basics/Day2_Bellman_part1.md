# Day 2 Deep Dive: Bellman Optimality & Contraction Mapping

## 1. The Bellman Optimality Equation
While the Expectation Equation tells us the value of a *fixed* policy, the **Optimality Equation** tells us the value of the *best possible* policy.

### The Optimal Value Function $V_*(s)$
$$ V_*(s) = \max_{\pi} V_{\pi}(s) $$
It satisfies the recursive relationship:
$$ V_*(s) = \max_{a} \sum_{s', r} P(s', r | s, a) [r + \gamma V_*(s')] $$
Notice the **max** operator. This means: "The value of a state under the optimal policy is the value of taking the *best* action and then acting optimally thereafter."

### The Optimal Action-Value Function $Q_*(s, a)$
$$ Q_*(s, a) = \sum_{s', r} P(s', r | s, a) [r + \gamma \max_{a'} Q_*(s', a')] $$

## 2. Linearity vs. Non-Linearity
*   **Expectation Equation:** Linear system of equations. Can be solved directly as $V = R + \gamma P V \implies V = (I - \gamma P)^{-1} R$ (if state space is small).
*   **Optimality Equation:** Non-linear due to the $\max$ operator. Cannot be solved with simple matrix inversion. We must use iterative methods (Value Iteration, Q-Learning).

## 3. Contraction Mapping Theorem
Why does iteratively applying the Bellman update converge?
The Bellman operator $T$ is a **contraction mapping** in the $L_\infty$ norm, provided $\gamma < 1$.
$$ ||TV - TU||_\infty \le \gamma ||V - U||_\infty $$
This means that every time we apply the update, the distance between our current value estimate $V$ and the true optimal value $V_*$ shrinks by a factor of $\gamma$.
By the **Banach Fixed Point Theorem**, this guarantees:
1.  **Existence:** A unique fixed point $V_*$ exists.
2.  **Convergence:** Iterative application converges to $V_*$ from *any* initial guess.

## 4. Greedy Policies
Once we have $V_*$ (or $Q_*$), the optimal policy is deterministic and greedy:
$$ \pi_*(s) = \arg\max_a Q_*(s, a) $$
If there are multiple actions with the same max value, any mixture of them is optimal.
