# Day 2: Autograd & Computational Graphs - Theory & Implementation

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: Automatic Differentiation, Jacobian Matrices, and Dynamic Graphs

## 1. Theoretical Foundation: Calculus of Deep Learning

### The Goal
Training a neural network is an optimization problem: Find parameters $\theta$ that minimize a loss function $L(\theta)$. We use Gradient Descent:
$$ \theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta) $$
To do this, we need the gradient $\nabla_\theta L$.

### The Chain Rule
For a composite function $z = f(g(x))$, the derivative is:
$$ \frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} $$
where $y = g(x)$.

In Deep Learning, we have a massive composite function:
$$ L = Loss(Layer_N(...Layer_1(x)...)) $$
Backpropagation is simply the repeated application of the Chain Rule from the output (Loss) back to the input (Weights).

### The Jacobian Matrix
When functions map vectors to vectors ($\vec{y} = f(\vec{x})$ where $\vec{x} \in \mathbb{R}^n, \vec{y} \in \mathbb{R}^m$), the derivative is the **Jacobian Matrix** $J \in \mathbb{R}^{m \times n}$:
$$ J_{ij} = \frac{\partial y_i}{\partial x_j} $$

PyTorch Autograd does **not** compute the full Jacobian (it would be too large). Instead, it computes the **Vector-Jacobian Product (VJP)**.
Given a gradient vector $v$ from the next layer (upstream gradient), it computes:
$$ v^T J $$
This propagates the gradient backward efficiently.

## 2. The Computational Graph (DAG)

PyTorch builds a **Directed Acyclic Graph (DAG)** to record operations.
*   **Nodes**: Tensors (Data).
*   **Edges**: Functions (Operations like `Mul`, `Add`, `ReLU`).

### Dynamic vs Static Graphs
*   **Static (TensorFlow v1)**: Define the graph structure once, then compile and run.
    *   *Pros*: Easy to optimize (fusion), serialize.
    *   *Cons*: Hard to debug, no Python control flow.
*   **Dynamic (PyTorch)**: The graph is built **on-the-fly** as code executes.
    *   *Pros*: Use Python `if/else`, `for` loops. Debug with `pdb`.
    *   *Cons*: Overhead of rebuilding graph every iteration (solved by `torch.compile`).

## 3. Implementation: `requires_grad`

The flag `requires_grad=True` tells PyTorch: "Track every operation on this tensor."

```python
import torch

# Weights (Leaf nodes)
w = torch.tensor([1.0, 2.0], requires_grad=True)
x = torch.tensor([3.0, 4.0]) # Data (No grad needed)

# Forward Pass (Building the Graph)
y = w * x          # MulBackward
z = y.sum()        # AddBackward

print(z) 
# tensor(11., grad_fn=<SumBackward0>)
# grad_fn is the link to the graph
```

## 4. The Backward Pass

Calling `.backward()` triggers the traversal of the graph.

```python
# Backward Pass
z.backward()

# Gradients are accumulated in .grad
print(w.grad) 
# dz/dw = d(w*x)/dw = x
# Result: [3.0, 4.0]
```

**Important**: Gradients accumulate!
If you call `z.backward()` again, `w.grad` will become `[6.0, 8.0]`.
You must zero them out: `w.grad.zero_()` or `optimizer.zero_grad()`.

## 5. Controlling the Flow

### `torch.no_grad()`
Disables graph building. Used for Inference to save memory.
```python
with torch.no_grad():
    y = w * x
    # y.requires_grad is False. No graph stored.
```

### `detach()`
Cuts the graph. Returns a new tensor that shares storage but has no history.
Useful for Truncated Backpropagation through Time (TBPTT) in RNNs.

```python
y = w * x
z = y.detach() # z is a leaf now. Grads won't flow back to w through z.
```

## 6. Advanced: Custom Autograd Functions

You can define your own differentiable operations by subclassing `torch.autograd.Function`.

```python
class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input) # Save for backward
        return input ** 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # d(x^2)/dx = 2x
        # Chain rule: grad_output * local_derivative
        return grad_output * (2 * input)

# Usage
sq = MySquare.apply
out = sq(torch.tensor(3.0, requires_grad=True))
out.backward()
```
