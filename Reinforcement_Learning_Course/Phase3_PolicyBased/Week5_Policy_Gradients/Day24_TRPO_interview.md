# Day 24 Interview Questions: TRPO

## Q1: What is the main idea behind TRPO?
**Answer:**
TRPO constrains policy updates using KL divergence to ensure the new policy doesn't deviate too much from the old policy.
$$ \max_\theta \mathbb{E}[\frac{\pi_\theta}{\pi_{\theta_{old}}} A] \quad \text{s.t.} \quad D_{KL}(\pi_{\theta_{old}} || \pi_\theta) \leq \delta $$
This guarantees **monotonic improvement**: the new policy is at least as good as the old one.

## Q2: What is the Fisher Information Matrix?
**Answer:**
The Fisher Information Matrix measures the curvature of the KL divergence in parameter space:
$$ F_{ij} = \mathbb{E}[\frac{\partial \log \pi}{\partial \theta_i} \frac{\partial \log \pi}{\partial \theta_j}] $$
It's used to compute the **natural gradient**, which is the direction of steepest ascent in the distribution space (not parameter space).
Natural gradients are invariant to reparameterization of the policy.

## Q3: How does TRPO compute the policy update?
**Answer:**
TRPO uses a **constrained optimization** approach:
1. Compute the policy gradient $g$.
2. Use **Conjugate Gradient** to solve $F^{-1} g$ (natural gradient direction).
3. Compute step size: $\sqrt{\frac{2\delta}{g^T F^{-1} g}}$.
4. Perform a **line search** to ensure the KL constraint is satisfied.
5. Update parameters along the natural gradient direction.

## Q4: What is Conjugate Gradient and why is it used?
**Answer:**
Conjugate Gradient (CG) is an iterative method to solve $Ax = b$ for large matrices.
In TRPO, we need to compute $F^{-1} g$, but:
*   Computing $F$ explicitly is $O(n^2)$ in memory and $O(n^3)$ in time.
*   Inverting $F$ is even more expensive.
CG solves $F x = g$ using only **Hessian-vector products** $F v$, which can be computed efficiently via automatic differentiation.
CG typically converges in ~10 iterations.

## Q5: Why is PPO preferred over TRPO?
**Answer:**
*   **Simplicity:** PPO uses a first-order clipped objective, no second-order optimization.
*   **Speed:** PPO is 2-3x faster than TRPO.
*   **Ease of Implementation:** TRPO requires CG and line search, adding complexity.
*   **Performance:** In practice, PPO performs just as well as TRPO.
TRPO is important for theoretical understanding, but PPO is the practical choice.

## Q6: What is monotonic improvement in TRPO?
**Answer:**
TRPO guarantees that each policy update **never decreases** performance:
$$ J(\pi_{new}) \geq J(\pi_{old}) $$
This is achieved by constraining the KL divergence, which ensures the approximation error in the surrogate loss is small.
Monotonic improvement makes TRPO very safe for real-world robotics, where a bad policy update could damage hardware.
