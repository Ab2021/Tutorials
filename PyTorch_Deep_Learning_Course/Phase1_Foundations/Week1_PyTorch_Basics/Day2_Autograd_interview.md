# Day 2: Autograd - Interview Questions

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: Autograd, Math, and Graphs

### 1. Explain the difference between Reverse Mode and Forward Mode Automatic Differentiation.
**Answer:**
*   **Reverse Mode (Backprop)**: Propagates gradients from Output to Input. Computes Vector-Jacobian Product ($v^T J$). Efficient when Outputs (1, Loss) << Inputs (Millions of params). Complexity proportional to Inputs.
*   **Forward Mode**: Propagates gradients from Input to Output. Computes Jacobian-Vector Product ($J v$). Efficient when Inputs << Outputs.

### 2. Why does PyTorch accumulate gradients by default?
**Answer:**
*   To support cases where a single optimization step depends on multiple backward passes.
*   **Example 1**: RNNs (Backprop through time).
*   **Example 2**: Gradient Accumulation (Simulating large batch sizes on small VRAM).
*   **Example 3**: Multi-task learning (Summing gradients from different losses).

### 3. What is the "Vanishing Gradient" problem and how does Autograd help diagnose it?
**Answer:**
*   In deep networks, gradients can become tiny ($< 10^{-5}$) as they multiply through layers (Chain Rule), especially with Sigmoid/Tanh.
*   Diagnose using **Hooks** (`register_hook`) to print gradient norms at each layer during backward pass.
*   Fix using ReLU, BatchNorm, or Residual Connections.

### 4. Can you differentiate through an `argmax` or `round` operation?
**Answer:**
*   No. These are step functions (discontinuous or zero derivative almost everywhere).
*   Gradient is 0 or undefined.
*   **Workaround**: Use "Straight-Through Estimator" (STE) or Softmax (differentiable approximation).

### 5. How does PyTorch handle control flow (if/else) in the graph?
**Answer:**
*   PyTorch uses **Dynamic Graphs**.
*   The graph is constructed *during* execution.
*   If `x > 0`, the graph includes the "True" branch. If `x < 0` in the next iteration, the graph includes the "False" branch.
*   This makes it trivial to support variable length sequences or logic.

### 6. What is the computational complexity of Backpropagation?
**Answer:**
*   Time: $O(N)$ where $N$ is the number of operations in the forward pass. (Roughly 2x forward pass cost).
*   Space: $O(N)$ to store intermediate activations (feature maps) needed for gradient computation.

### 7. What is "Gradient Checkpointing" and when should you use it?
**Answer:**
*   A technique to trade Compute for Memory.
*   Instead of storing *all* intermediate activations, we drop some and re-compute them during the backward pass.
*   Reduces memory usage from $O(N)$ to $O(\sqrt{N})$.
*   Use when training very deep models (LLMs) that don't fit in VRAM.

### 8. Why do we need `retain_graph=True`?
**Answer:**
*   By default, PyTorch frees the graph buffers after `.backward()` to save memory.
*   If you need to call `.backward()` multiple times on the same graph (e.g., GANs: update Generator, then Discriminator using same output), you must retain the graph.

### 9. What is a "Leaf Variable" in the context of `requires_grad`?
**Answer:**
*   A tensor that is at the beginning of the graph (created by user, not by an operation).
*   Usually Weights or Inputs.
*   Only leaf tensors with `requires_grad=True` will have their `.grad` attribute populated. Intermediate gradients are freed to save memory (unless `retain_grad()` is called).

### 10. How does `torch.autograd.grad` differ from `loss.backward()`?
**Answer:**
*   `loss.backward()`: Computes gradients for *all* leaf tensors and stores them in `.grad` attribute. Returns `None`.
*   `torch.autograd.grad(outputs, inputs)`: Computes gradients of `outputs` w.r.t specific `inputs`. Returns the gradients as a tuple. Does not populate `.grad`. Functional approach.

### 11. Explain the "Stop Gradient" operation.
**Answer:**
*   `x.detach()` in PyTorch.
*   Prevents gradients from flowing back past this point.
*   Used in:
    *   Freezing layers (Transfer Learning).
    *   Target networks in DQN (Reinforcement Learning).
    *   Truncated BPTT in RNNs.

### 12. What is the derivative of ReLU at x=0? How does PyTorch handle it?
**Answer:**
*   Mathematically undefined (subgradient is $[0, 1]$).
*   PyTorch convention: Derivative is 0 at x=0. (Or 0.5 in some libraries, but usually 0 or 1).

### 13. Why is the Jacobian matrix sparse for Convolutional layers?
**Answer:**
*   In a Conv layer, each output pixel depends only on a small window (kernel size) of input pixels.
*   Most entries in the Jacobian (relationship between any output and any input) are zero.
*   This is why Convolutions are efficient.

### 14. What happens if you set `requires_grad=False` on the weights of a model?
**Answer:**
*   The weights are treated as constants.
*   Gradients are not computed for them.
*   Optimizer will not update them.
*   Used for "Freezing" a backbone.

### 15. How do you implement a custom Loss function?
**Answer:**
*   Just write it using PyTorch operations (`+`, `-`, `*`, `exp`). Autograd handles the rest.
*   Only need to write custom `Function` if using non-PyTorch ops (e.g., calling C++ code or Numpy).

### 16. What is the "Re-parameterization Trick" (VAE)?
**Answer:**
*   Sampling $z \sim N(\mu, \sigma)$ is stochastic and non-differentiable.
*   Trick: $z = \mu + \sigma \cdot \epsilon$ where $\epsilon \sim N(0, 1)$.
*   Now $z$ is a deterministic function of $\mu, \sigma$ (differentiable) and stochasticity is external ($\epsilon$).

### 17. What is the difference between `model.eval()` and `torch.no_grad()`?
**Answer:**
*   `model.eval()`: Changes behavior of layers (Dropout disabled, BatchNorm uses running stats). Does NOT disable gradients.
*   `torch.no_grad()`: Disables gradient computation. Does NOT change layer behavior.
*   Usually used together during inference.

### 18. How does PyTorch handle complex numbers in Autograd?
**Answer:**
*   PyTorch supports complex tensors.
*   Gradients are computed using Wirtinger Calculus (treating $z$ and $\bar{z}$ as independent).

### 19. What is "Gradient Clipping"?
**Answer:**
*   Rescaling gradients if their norm exceeds a threshold.
*   Prevents Exploding Gradients.
*   `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

### 20. Can you backpropagate through an Integer tensor?
**Answer:**
*   No. Integers are discrete.
*   Gradients require continuous domains.
*   PyTorch tensors must be floating point (`float32`, `float64`) to have `requires_grad=True`.
