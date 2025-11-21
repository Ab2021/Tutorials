# Day 18: Optimization - Interview Questions

> **Topic**: Training Dynamics
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Explain SGD vs Adam. When to use which?
**Answer:**
*   **SGD**: Simple. Generalizes well. Needs tuning. Good for CV.
*   **Adam**: Adaptive LR. Fast convergence. Good default. Good for NLP/RL.

### 2. How does Adam work?
**Answer:**
*   Combines **Momentum** (First moment) and **RMSProp** (Second moment - variance).
*   Keeps exponentially decaying average of past gradients ($m$) and squared gradients ($v$).
*   Update: $w = w - \eta \cdot m / (\sqrt{v} + \epsilon)$.

### 3. What is Learning Rate Warmup?
**Answer:**
*   Start with low LR, increase linearly to target LR, then decay.
*   **Why**: Early gradients are unstable. Warmup stabilizes training start.

### 4. What is Learning Rate Decay (Scheduler)?
**Answer:**
*   Reduce LR as training progresses.
*   **Step Decay**: Drop by 10x every 30 epochs.
*   **Cosine Decay**: Smooth curve to zero.
*   Helps settle into sharp minima.

### 5. Explain Gradient Clipping.
**Answer:**
*   If norm of gradient vector > Threshold (e.g., 1.0), scale it down.
*   Prevents **Exploding Gradients** (common in RNNs).

### 6. What is Weight Decay? How is it different from L2 Regularization?
**Answer:**
*   **L2**: Adds term to Loss.
*   **Weight Decay**: Subtracts term directly from weight update.
*   Equivalent for SGD. Different for Adam (AdamW fixes this).

### 7. What is the difference between Batch Norm and Layer Norm?
**Answer:**
*   **Batch Norm**: Normalizes across the **Batch** dimension (for each feature). Depends on Batch Size. Bad for RNNs.
*   **Layer Norm**: Normalizes across the **Feature** dimension (for each sample). Independent of Batch Size. Good for RNNs/Transformers.

### 8. What is RMSProp?
**Answer:**
*   Divides gradient by running average of its magnitude.
*   Adapts LR per parameter. High gradient -> Low LR. Low gradient -> High LR.

### 9. Why do we need Bias terms in Neural Networks?
**Answer:**
*   Allows shifting the activation function left or right.
*   Without bias, line always goes through origin ($y = wx$).

### 10. What is Early Stopping?
**Answer:**
*   Monitor Validation Loss.
*   Stop training when Val Loss stops improving (or starts increasing).
*   Prevents overfitting.

### 11. What is the "Lottery Ticket Hypothesis"?
**Answer:**
*   A large network contains a smaller sub-network (winning ticket) that, if trained in isolation, would match the performance of the full network.
*   Justifies Pruning.

### 12. What is Mode Collapse in GANs?
**Answer:**
*   Generator produces only one type of output (e.g., only generates "Shoe" regardless of noise input).
*   Discriminator gets stuck.

### 13. Explain the concept of "Saddle Points" in high dimensions.
**Answer:**
*   Most critical points are saddle points, not local minima.
*   Gradient is zero, but Hessian has positive and negative eigenvalues.

### 14. What is Polyak Averaging?
**Answer:**
*   Keep a moving average of the weights during training.
*   Use averaged weights for inference.
*   Often leads to better generalization.

### 15. How does Batch Size affect Generalization?
**Answer:**
*   **Small Batch**: Noisy gradients acts as noise injection (Regularization). Flatter minima. Better generalization.
*   **Large Batch**: Sharp minima. Worse generalization (unless tuned carefully).

### 16. What is Gradient Accumulation?
**Answer:**
*   Simulate large batch size on small GPU.
*   Run forward/backward for N mini-batches, summing gradients. Update weights once.

### 17. What is Mixed Precision Training?
**Answer:**
*   Use FP16 (Half precision) for math, FP32 (Single) for master weights.
*   Reduces memory by 50%, speeds up math (Tensor Cores).

### 18. What is the difference between Adam and AdamW?
**Answer:**
*   **AdamW** decouples Weight Decay from the gradient update.
*   Corrects L2 regularization implementation in Adam. Essential for Transformers.

### 19. What is Lookahead Optimizer?
**Answer:**
*   "k steps forward, 1 step back".
*   Maintains "Slow weights" and "Fast weights". Interpolates. Stable.

### 20. How do you debug a network that isn't learning?
**Answer:**
*   Overfit a single batch.
*   Check input normalization.
*   Check output activation (Sigmoid vs Softmax).
*   Check Loss function.
*   Visualize gradients.
