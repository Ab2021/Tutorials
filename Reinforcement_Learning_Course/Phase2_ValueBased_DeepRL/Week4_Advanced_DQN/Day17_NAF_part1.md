# Day 17 Deep Dive: The Geometry of NAF

## 1. Quadratic Approximation
NAF approximates the Q-function as a quadratic function of action $a$ near the optimal action $\mu(s)$.
$$ Q(s, a) \approx Q(s, \mu(s)) - \frac{1}{2} (a - \mu(s))^T P(s) (a - \mu(s)) $$
*   This is equivalent to a **second-order Taylor expansion** of the true Q-function around the maximum.
*   **Limitation:** It cannot represent multi-modal Q-functions (e.g., if going Left is good and Right is good, but staying Middle is bad). NAF will try to average them and pick Middle (which is bad).

## 2. Relation to Model-Based RL (iLQG)
The structure of NAF is very similar to the value function in **Linear-Quadratic Regulators (LQR)** and **iLQG**.
*   In LQR, we assume linear dynamics and quadratic cost. The optimal Value function is *exactly* quadratic.
*   NAF can be seen as a "Model-Free LQR" that learns the quadratic structure directly from data without knowing the dynamics matrices $A$ and $B$.

## 3. Imaginary Actions
Since we can calculate the max analytically, we don't need to actually *sample* the optimal action to train on it.
We can perform off-policy updates using the analytic target:
$$ y = r + \gamma V(s') $$
This makes NAF very sample efficient compared to methods that need to run an optimization loop (like Cross-Entropy Method) to find the target action.
