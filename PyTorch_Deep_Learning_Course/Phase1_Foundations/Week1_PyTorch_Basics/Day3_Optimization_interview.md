# Day 3: Optimization - Interview Questions

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: SGD, Adam, and Convergence

### 1. Why is SGD preferred over Batch Gradient Descent for Deep Learning?
**Answer:**
*   **Computational Efficiency**: Computing gradient over 1M images is impossible in RAM. SGD uses mini-batches (e.g., 32).
*   **Generalization**: The noise in SGD helps escape sharp local minima and find flatter minima, which generalize better.
*   **Redundancy**: Data is often redundant. Gradient from 1000 images is a good approximation of gradient from 1M images.

### 2. Explain the difference between Adam and AdamW.
**Answer:**
*   **Adam**: Adds L2 regularization to the Loss function. Due to adaptive learning rates, the regularization strength varies per parameter implicitly.
*   **AdamW**: Decouples weight decay from the gradient update. Applies weight decay directly to the weights ($\theta \leftarrow \theta(1 - \lambda)$) *before* the Adam update. This is mathematically correct for adaptive optimizers.

### 3. What is "Momentum" intuitively?
**Answer:**
*   It simulates a heavy ball rolling down the loss surface.
*   It builds up velocity in directions with consistent gradients (speeding up convergence).
*   It resists changing direction due to noise (dampening oscillations).

### 4. Why do we need Learning Rate Schedulers?
**Answer:**
*   **Early training**: High LR helps traverse the landscape quickly and escape bad local minima.
*   **Late training**: We are near the minimum. High LR causes oscillation around the bottom. We need to decay LR to settle into the minimum.

### 5. What is a "Saddle Point" and why is it a problem?
**Answer:**
*   A point where gradient is zero, but it's a min in some directions and max in others.
*   In high dimensions, saddle points are exponentially more common than local minima.
*   SGD can get stuck (slow down) near saddle points because gradients are small. Noise helps escape.

### 6. How does RMSprop work?
**Answer:**
*   Root Mean Square Propagation.
*   It divides the learning rate by the moving average of the root mean squared gradients.
*   Effect: Parameters with large gradients get smaller effective LR (preventing divergence). Parameters with small gradients get larger effective LR (speeding up learning).

### 7. What is the "Lookahead" optimizer?
**Answer:**
*   "k steps forward, 1 step back".
*   It maintains a set of "slow weights". It runs a standard optimizer (like Adam) for k steps ("fast weights"), then updates the slow weights towards the fast weights.
*   Improves stability.

### 8. Can you use L-BFGS for training Deep Nets?
**Answer:**
*   Generally No.
*   It requires computing/approximating the Hessian (or storing history of size $O(Params)$).
*   Too expensive for large models.
*   Also, L-BFGS works best for full-batch optimization, not mini-batch (stochastic) settings.

### 9. What is "Warmup" in learning rate schedules?
**Answer:**
*   Starting with a very low LR and linearly increasing it for a few epochs.
*   Reason: In the beginning, gradients are huge and unstable. A high LR might throw the weights into a bad region. Warmup allows statistics (Adam moments) to stabilize.

### 10. What is the "Sharpness" of a minimum?
**Answer:**
*   Curvature of the loss function at the minimum (Eigenvalues of Hessian).
*   **Sharp Minima**: High curvature. Small perturbation in weights leads to huge loss increase. Bad generalization.
*   **Flat Minima**: Low curvature. Robust to perturbations. Good generalization.

### 11. How do you choose the Batch Size?
**Answer:**
*   **Small Batch (32)**: More noise, better generalization (flat minima), slower training (less parallelism).
*   **Large Batch (8k)**: Less noise, faster training (GPU saturation), but tends to converge to sharp minima (generalization gap).
*   **Linear Scaling Rule**: If you double batch size, double learning rate.

### 12. What is "Gradient Accumulation"?
**Answer:**
*   Simulating a large batch size by running multiple forward/backward passes before `optimizer.step()`.
*   Useful when VRAM is limited.

### 13. Why do we initialize biases to 0?
**Answer:**
*   Symmetry breaking is handled by weights.
*   Biases just shift the activation. 0 is a neutral starting point.
*   Exception: For the final classification layer with imbalanced data, initializing bias to $\log(prior)$ helps convergence.

### 14. What is "Mode Collapse" in optimization?
**Answer:**
*   Usually in GANs.
*   The generator finds one output that tricks the discriminator and produces *only* that output.
*   Optimization fails to find the diverse distribution.

### 15. Explain "Polyak Averaging" (SWA).
**Answer:**
*   Stochastic Weight Averaging.
*   Averaging the weights of the model at different points in the trajectory (e.g., end of each epoch).
*   The averaged solution often lands in the center of a flat minimum, improving generalization.

### 16. What happens if Learning Rate is too high?
**Answer:**
*   Loss oscillates or diverges (NaN).
*   Weights update too much, overshooting the valley.

### 17. What happens if Learning Rate is too low?
**Answer:**
*   Training is extremely slow.
*   Might get stuck in a suboptimal local minimum or saddle point.

### 18. How does `optimizer.zero_grad(set_to_none=True)` differ from `zero_grad()`?
**Answer:**
*   `set_to_none=True` sets `.grad` to `None` instead of a tensor of zeros.
*   Saves memory and slightly faster (skips memset).
*   PyTorch treats `None` grad as zero.

### 19. What is the "Ill-conditioning" of the Hessian?
**Answer:**
*   When the ratio of largest to smallest eigenvalue (condition number) is huge.
*   The valley is a long, narrow canyon.
*   SGD bounces back and forth across the canyon walls instead of moving along the floor.
*   Momentum helps fix this.

### 20. Why is Adam sometimes worse than SGD?
**Answer:**
*   Adam generalizes worse on some tasks (like Image Classification with ResNet).
*   It might converge too fast to a sharp minimum.
*   State-of-the-art results on ImageNet often use SGD + Momentum. Transformers prefer AdamW.
