# Day 7 Deep Dive: The Dangers of Function Approximation

## 1. Semi-Gradient Methods
In true Gradient Descent, we minimize error on a fixed target.
$$ \mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla J(\mathbf{w}) $$
In TD learning, the target $U_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w})$ *depends on the weights* we are updating.
If we took the full gradient:
$$ \nabla \frac{1}{2} (U_t - \hat{v}(S_t, \mathbf{w}))^2 = -(U_t - \hat{v}(S_t, \mathbf{w})) (\nabla \hat{v}(S_t) - \gamma \nabla \hat{v}(S_{t+1})) $$
This extra term $\gamma \nabla \hat{v}(S_{t+1})$ is complex and often ignored.
**Semi-Gradient:** We pretend the target is a constant (like in Supervised Learning).
$$ \mathbf{w} \leftarrow \mathbf{w} + \alpha [U_t - \hat{v}(S_t)] \nabla \hat{v}(S_t) $$
This simplifies computation and still converges for linear on-policy cases.

## 2. The Deadly Triad
Reinforcement Learning algorithms can become unstable and diverge (values $\to \infty$) when three elements are combined:
1.  **Function Approximation:** Generalizing across states (not tabular).
2.  **Bootstrapping:** Updating estimates based on other estimates (TD/DP).
3.  **Off-Policy Training:** Training on a distribution of transitions different from the target policy.

**Why?**
*   FA introduces errors.
*   Bootstrapping propagates these errors.
*   Off-policy learning means we might update states rarely visited by the target policy, where the approximation error is high. This error then bleeds back into the frequently visited states via bootstrapping, creating a positive feedback loop of error.

## 3. Convergence Guarantees
*   **Linear MC:** Converges (No bootstrapping).
*   **Linear TD(0):** Converges (On-policy).
*   **Linear Q-Learning:** **Can Diverge** (Off-policy + Bootstrapping + FA).
*   **DQN:** Uses Experience Replay and Target Networks to mitigate these issues (effectively making it look more like Supervised Learning).
