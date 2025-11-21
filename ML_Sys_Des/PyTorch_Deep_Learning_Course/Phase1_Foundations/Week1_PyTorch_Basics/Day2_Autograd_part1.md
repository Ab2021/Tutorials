# Day 2: Autograd - Deep Dive

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: VJP, Hooks, and Higher-Order Derivatives

## 1. Vector-Jacobian Product (VJP) Explained

Let's rigorously understand why we compute VJP.
Consider a function $f: \mathbb{R}^n \to \mathbb{R}^m$.
The Jacobian $J$ is $m \times n$.
If we have a scalar loss $L$, we want $\nabla_x L$ (size $n$).
By Chain Rule:
$$ \nabla_x L = (\nabla_y L)^T \cdot J $$
Here, $\nabla_y L$ is the gradient of the loss with respect to the output of $f$. Let's call this vector $v$ (size $m$).
So we need to compute $v^T J$.

**Why not compute J?**
If $n=1000$ and $m=1000$, $J$ has $1,000,000$ entries.
But $v^T J$ is just a vector of size 1000.
Computing $J$ explicitly is wasteful. We can compute the product directly.

**Example**:
$y = Ax$. $J = A$.
$v^T J = v^T A$.
We never need to construct a Jacobian matrix for element-wise ops like ReLU; we just multiply by the local derivative (0 or 1).

## 2. Higher-Order Derivatives (Hessians)

Sometimes we need the second derivative (e.g., for Newton's Method or Gradient Penalty in GANs).
PyTorch supports this via `create_graph=True`.

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3

# First Derivative
grad_1 = torch.autograd.grad(y, x, create_graph=True)[0]
# grad_1 = 3x^2 = 12

# Second Derivative
grad_2 = torch.autograd.grad(grad_1, x)[0]
# grad_2 = 6x = 12
```

When `create_graph=True`, the gradient computation itself is added to the graph, allowing you to backpropagate through the backpropagation!

## 3. Hooks: Debugging the Backward Pass

Hooks allow you to inspect or modify gradients as they flow through the graph. This is invaluable for debugging **Vanishing/Exploding Gradients**.

```python
v = torch.tensor([1.0, 2.0], requires_grad=True)

def log_grad(grad):
    print(f"Gradient at v: {grad}")
    if grad.norm() > 1.0:
        return grad.clamp(-1, 1) # Gradient Clipping inside the graph!

# Register hook
handle = v.register_hook(log_grad)

y = v * 2
y.sum().backward()
# Output: Gradient at v: tensor([2., 2.])
```

## 4. In-Place Operations and Version Counter

PyTorch protects you from shooting yourself in the foot.
Every tensor has a `_version` counter.
*   `x = x + 1` -> New tensor.
*   `x += 1` -> In-place. Increments version.

If you save `x` for backward (e.g., in `y = x * w`), and then modify `x` in-place (`x += 1`), and *then* call `y.backward()`, PyTorch will see that the version of `x` has changed since it was saved.
**Error**: `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`.

## 5. Forward Mode Autograd (New in PyTorch)

Standard Backprop is **Reverse Mode** (Output to Input). Efficient when Inputs > Outputs (Loss is scalar).
**Forward Mode** (Input to Output) computes Jacobian-Vector Products (JVP). Efficient when Outputs > Inputs.

```python
import torch.autograd.forward_ad as fwAD

x = torch.tensor(1.0)
tangent = torch.tensor(1.0) # Direction

with fwAD.dual_level():
    dual_x = fwAD.make_dual(x, tangent)
    y = dual_x ** 2
    jvp = fwAD.unpack_dual(y).tangent
    # jvp = 2x * 1 = 2
```
Useful for computing Jacobians row-by-row or for specific sensitivity analysis.
