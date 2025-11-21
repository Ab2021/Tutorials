# Day 24 Deep Dive: Natural Gradients and Monotonic Improvement

## 1. Why Natural Gradients?
Standard gradient descent treats parameter space uniformly.
**Problem:** A small change in $\theta$ can cause a large or small change in the policy distribution, depending on the parameterization.
**Natural Gradient:** Accounts for the geometry of the probability distribution space.
$$ \tilde{g} = F^{-1} g $$
where $F$ is the Fisher Information Matrix.
*   This gives the direction of steepest ascent in the **distribution space**, not parameter space.
*   Invariant to reparameterization.

## 2. Monotonic Improvement Theorem
TRPO guarantees that the new policy is at least as good as the old one:
$$ J(\pi_{new}) \geq J(\pi_{old}) $$
**Proof Sketch:**
*   Surrogate loss: $L(\theta) = \mathbb{E}[r_t(\theta) A_t]$.
*   KL constraint ensures the policies are "close enough" for the approximation to hold.
*   This makes TRPO extremely safe for real-world applications (robotics).

## 3. Conjugate Gradient Method
To avoid computing $F^{-1}$ (which requires $O(n^3)$ operations for $n$ parameters):
*   Use **Conjugate Gradient (CG)** to solve $F x = g$ iteratively.
*   Only requires computing **Hessian-vector products** $F v$, which can be done efficiently using automatic differentiation.
*   CG converges in $\sim 10$ iterations for typical problems.

## 4. Why TRPO is Rarely Used
*   **Complexity:** Requires second-order optimization, CG, line search.
*   **Computational Cost:** ~2-3x slower than PPO per update.
*   **Code Complexity:** Hard to implement and debug.
*   **PPO is Good Enough:** In practice, PPO's first-order approximation works just as well with much less complexity.
