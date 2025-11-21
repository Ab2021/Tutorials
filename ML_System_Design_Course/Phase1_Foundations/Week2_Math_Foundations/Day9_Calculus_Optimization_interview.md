# Day 9: Calculus & Optimization - Interview Questions

> **Topic**: Learning Dynamics
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is a Derivative? What does it represent in ML?
**Answer:**
*   Rate of change of a function. Slope of the tangent line.
*   **ML**: Tells us how to change weights to decrease error (Gradient).

### 2. What is a Partial Derivative?
**Answer:**
*   Derivative with respect to one variable, treating others as constants.
*   $\frac{\partial f}{\partial x}$.

### 3. Explain the Gradient. How is it related to the direction of steepest ascent?
**Answer:**
*   Vector of partial derivatives. $\nabla f = [\frac{\partial f}{\partial x_1}, ...]$.
*   Points in the direction of **steepest ascent**.
*   We move in the opposite direction ($-\nabla f$) to minimize loss.

### 4. What is the Chain Rule? Why is it crucial for Backpropagation?
**Answer:**
*   Rule for differentiating composite functions. $(f(g(x)))' = f'(g(x)) g'(x)$.
*   **Backprop**: Neural nets are nested functions $f_L(...f_1(x))$. Chain rule allows us to compute gradients layer by layer from output to input.

### 5. What is a Convex Function? Why is convexity important in optimization?
**Answer:**
*   A bowl-shaped function. Line segment between any two points lies above the curve.
*   **Importance**: Any local minimum is a **Global Minimum**. Optimization is easy and guaranteed to converge.

### 6. Explain Gradient Descent.
**Answer:**
*   Iterative algorithm to find minimum.
*   $w_{new} = w_{old} - \eta \nabla Loss(w)$.
*   Step down the hill.

### 7. What is the difference between Batch, Stochastic (SGD), and Mini-batch Gradient Descent?
**Answer:**
*   **Batch**: Uses ALL data for one step. Stable but slow. Memory heavy.
*   **SGD**: Uses ONE sample. Noisy but fast. Escapes local minima.
*   **Mini-batch**: Uses K samples (e.g., 32). Best of both worlds. SIMD friendly.

### 8. What is the Learning Rate? What happens if it's too high or too low?
**Answer:**
*   Step size ($\eta$).
*   **Too High**: Overshoots minimum. Diverges.
*   **Too Low**: Converges very slowly. Gets stuck in local minima.

### 9. What is a Local Minimum vs Global Minimum?
**Answer:**
*   **Local**: Lowest point in a neighborhood.
*   **Global**: Lowest point in the entire domain.
*   Deep Learning loss landscapes are non-convex (many local minima), but usually local minima are "good enough".

### 10. What is a Saddle Point? Why is it a problem for optimization?
**Answer:**
*   Point where gradient is zero, but it's a min in one direction and max in another.
*   **Problem**: Gradients are small (plateau), slowing down training. More common than bad local minima in high dimensions.

### 11. Explain Momentum in optimization.
**Answer:**
*   Accumulates a moving average of past gradients.
*   "Velocity". Helps plow through saddle points and dampen oscillations in ravines.
*   $v_t = \gamma v_{t-1} + \eta \nabla L$.

### 12. What is the Jacobian Matrix?
**Answer:**
*   Matrix of all first-order partial derivatives of a vector-valued function.
*   If $f: R^n \to R^m$, Jacobian is $m \times n$.

### 13. What is the Hessian Matrix?
**Answer:**
*   Square matrix of second-order partial derivatives.
*   Describes the **curvature** of the function.
*   Used in Newton's Method.

### 14. Explain Newton's Method for optimization.
**Answer:**
*   Second-order method. Uses Hessian (Curvature) to jump directly to the minimum (assuming quadratic function).
*   $x_{new} = x - H^{-1} \nabla f$.
*   Fast convergence but computing $H^{-1}$ is $O(N^3)$ (too expensive for Deep Learning).

### 15. What is Constrained Optimization?
**Answer:**
*   Minimizing $f(x)$ subject to constraints $g(x) = 0$ or $h(x) \le 0$.
*   Example: SVM (Minimize weights s.t. margin $\ge 1$).

### 16. What are Lagrange Multipliers?
**Answer:**
*   Technique to solve constrained optimization.
*   Convert to unconstrained problem: $L(x, \lambda) = f(x) + \lambda g(x)$.
*   Solve $\nabla L = 0$.

### 17. Explain the concept of "Vanishing Gradients".
**Answer:**
*   In deep networks, gradients are multiplied by chain rule. If derivatives are small (< 1, e.g., Sigmoid), the product goes to zero exponentially.
*   Early layers stop learning.
*   **Fix**: ReLU, Residual Connections, Batch Norm.

### 18. Explain the concept of "Exploding Gradients".
**Answer:**
*   Gradients > 1 multiply to infinity. Weights become NaN.
*   **Fix**: Gradient Clipping.

### 19. What is the role of the Activation Function in terms of calculus/gradients?
**Answer:**
*   Introduces **Non-linearity**. Without them, a deep net is just a single linear layer.
*   Must be differentiable (almost everywhere) for Backprop.

### 20. How does L2 Regularization affect the gradient update?
**Answer:**
*   Adds $\lambda w^2$ to Loss.
*   Gradient becomes $\nabla L + 2\lambda w$.
*   Update: $w = w - \eta (\nabla L + 2\lambda w) = w(1 - 2\eta\lambda) - \eta \nabla L$.
*   **Weight Decay**: Shrinks weights towards zero at every step.
