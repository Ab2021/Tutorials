# Day 9 (Part 1): Advanced Optimization Theory

> **Phase**: 6 - Deep Dive
> **Topic**: Reaching the Minimum Fast
> **Focus**: Second-Order Methods, Constraints, and Geometry
> **Reading Time**: 60 mins

---

## 1. Second-Order Methods

SGD uses Gradient (Slope). Newton's Method uses Hessian (Curvature).

### 1.1 Newton's Method
*   Update: $x_{new} = x - H^{-1} \nabla f(x)$.
*   **Pros**: Converges quadratically (extremely fast) near minimum.
*   **Cons**: Computing/Inverting Hessian ($N \times N$) is $O(N^3)$. Impossible for Deep Learning.

### 1.2 Quasi-Newton (BFGS / L-BFGS)
*   **Idea**: Approximate the inverse Hessian using gradient history.
*   **L-BFGS**: Limited memory version. Stores last $m$ updates.
*   **Usage**: Scipy optimizers, fine-tuning small models.

---

## 2. Constrained Optimization

Minimize $f(x)$ subject to $g(x) = 0$.

### 2.1 Lagrange Multipliers
*   Construct Lagrangian: $\mathcal{L}(x, \lambda) = f(x) + \lambda g(x)$.
*   Solve $\nabla \mathcal{L} = 0$.
*   **Intuition**: At the optimum, the gradient of the objective is parallel to the gradient of the constraint.

### 2.2 KKT Conditions
*   Generalization for inequality constraints ($g(x) \le 0$).
*   Includes "Complementary Slackness": Either $\lambda = 0$ or $g(x) = 0$.

---

## 3. Geometry of Loss Surfaces

### 3.1 Saddle Points
*   Gradient is 0, but it's not a minimum. (Curvature is positive in one direction, negative in another).
*   **High Dimensions**: Saddle points are exponentially more common than local minima.
*   **Escape**: SGD with noise/momentum escapes saddle points. Newton's method is attracted to them (needs modification).

### 3.2 Convexity
*   **Definition**: Line segment between any two points lies above the curve.
*   **Property**: Local minimum = Global minimum.
*   **DL**: Neural Nets are highly non-convex, yet SGD finds good solutions. Why? (Open research: Mode Connectivity).

---

## 4. Tricky Interview Questions

### Q1: Why don't we use L-BFGS for training Transformers?
> **Answer**:
> 1.  **Stochasticity**: L-BFGS relies on accurate gradient estimates (full batch). It fails with noisy mini-batch gradients.
> 2.  **Cost**: Even approximating Hessian vector products is expensive compared to Adam.
> 3.  **Non-convexity**: Second-order methods can get stuck in saddle points more easily without careful damping.

### Q2: Explain Momentum physically.
> **Answer**: A ball rolling down a hill.
> *   **Gradient**: Acceleration (Force).
> *   **Momentum**: Velocity.
> *   Accumulates speed in consistent directions, dampens oscillations in zig-zag directions.

### Q3: What is the "Learning Rate Warmup"?
> **Answer**: Starting with small LR and increasing it.
> *   **Reason**: Early gradients are huge (random weights). Large LR causes divergence. Warmup allows weights to settle into a stable "valley" before accelerating.

---

## 5. Practical Edge Case: Exploding Gradients
*   **Signs**: Loss becomes `NaN`. Weights become huge.
*   **Fix**: Gradient Clipping (`torch.nn.utils.clip_grad_norm_`). Rescales the gradient vector if its norm exceeds a threshold (e.g., 1.0).

