# Day 20: Mobile Optimization - Deep Dive

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: Pruning, Distillation, and ExecuTorch

## 1. Model Pruning

Removing weights to make the model smaller/faster.

### Unstructured Pruning
Set individual weights to 0.
*   Result: Sparse Matrix.
*   **Problem**: GPUs/CPUs are designed for dense matrices. Sparse matrices are often *slower* unless sparsity > 90%.

### Structured Pruning
Remove entire **Channels** or **Filters**.
*   Result: A smaller dense matrix.
*   **Benefit**: Immediate speedup on all hardware.
*   **Criteria**: L1 Norm (remove filters with small weights).

```python
import torch.nn.utils.prune as prune

# Prune 30% of connections in a layer (Unstructured)
prune.l1_unstructured(model.conv1, name="weight", amount=0.3)

# Remove pruning re-parametrization (make it permanent)
prune.remove(model.conv1, 'weight')
```

## 2. Knowledge Distillation

Training a small **Student** to mimic a large **Teacher**.
Loss = $\alpha \cdot L_{CE}(y, \hat{y}_{student}) + (1-\alpha) \cdot L_{KL}(y_{teacher}, \hat{y}_{student})$
*   **Soft Targets**: The teacher's output probabilities contain information about class similarity (e.g., "Dog" is similar to "Cat", not "Car").
*   **Temperature ($T$)**: Softens the probability distribution. $p_i = \frac{\exp(z_i/T)}{\sum \exp(z_j/T)}$.

## 3. ExecuTorch (PyTorch Edge)

The new stack for on-device AI (Beta).
Replaces `PyTorch Mobile`.
1.  **Export**: `torch.export` (Sound Graph Capture).
2.  **Compile**: Lower to Edge IR.
3.  **Runtime**: Lightweight C++ runtime (Nano-sized).

## 4. Neural Architecture Search (NAS)

How MobileNetV3 was found.
*   **Search Space**: Kernel sizes, expansion ratios, depths.
*   **Controller**: RL agent or Evolutionary algorithm proposing architectures.
*   **Reward**: Accuracy - $\lambda \times$ Latency.
*   **Hardware-Aware**: Measures latency on the *actual device* (Pixel 4, iPhone) during search.
