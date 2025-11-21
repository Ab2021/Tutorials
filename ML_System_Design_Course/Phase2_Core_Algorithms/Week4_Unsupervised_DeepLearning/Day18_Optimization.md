# Day 18: Deep Learning Optimization

> **Phase**: 2 - Core Algorithms
> **Week**: 4 - Unsupervised & Deep Learning
> **Focus**: Training Dynamics & Stability
> **Reading Time**: 50 mins

---

## 1. Backpropagation: Under the Hood

Backprop is the engine of learning. It computes $\nabla_\theta L$ efficiently.

### 1.1 Computational Graph
*   Forward Pass: Compute outputs and store intermediate values (activations).
*   Backward Pass: Traverse graph in reverse. Multiply local gradients (Chain Rule).
*   **Memory Cost**: We must store activations from the forward pass to compute gradients. This is why training uses more VRAM than inference.

---

## 2. Stabilizing Training

Deep networks are notoriously unstable.

### 2.1 Batch Normalization
*   **Idea**: Normalize the inputs of each layer to have mean 0, var 1. Then scale and shift ($\gamma, \beta$).
*   **Benefit**:
    *   Prevents internal covariate shift.
    *   Allows higher learning rates.
    *   Acts as a regularizer.
*   **Gotcha**: Behaves differently in Train (using batch stats) vs. Eval (using running stats).

### 2.2 Dropout
*   **Idea**: Randomly zero out neurons during training.
*   **Interpretation**: Training an ensemble of $2^N$ thinned networks.
*   **Scaling**: At test time, multiply weights by $(1-p)$ to match expected magnitude. (Or scale up during training, which is standard now).

---

## 3. Real-World Challenges & Solutions

### Challenge 1: The Generalization Gap
**Scenario**: Train Loss 0.01, Val Loss 0.5. Massive overfitting.
**Solution**:
*   **Data Augmentation**: Create new samples (rotations, flips).
*   **Weight Decay (L2 Regularization)**.
*   **Label Smoothing**: Don't target [0, 1]. Target [0.1, 0.9]. Prevents overconfidence.

### Challenge 2: Learning Rate Tuning
**Scenario**: Loss oscillates (LR too high) or decreases too slowly (LR too low).
**Solution**:
*   **LR Schedulers**: Cosine Decay, ReduceLROnPlateau.
*   **Warmup**: Start low, ramp up, then decay.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: Why does Batch Normalization allow higher learning rates?**
> **Answer**: It ensures that layer activations don't explode or vanish. It smooths the optimization landscape (Lipschitz continuity), making the gradients more reliable and predictive, allowing larger steps without diverging.

**Q2: Explain the difference between Adam and SGD with Momentum.**
> **Answer**:
> *   **SGD+Momentum**: Uses a single global learning rate. Adds a velocity term to smooth direction.
> *   **Adam**: Adaptive. Keeps a separate learning rate for *each parameter*. Scales the step size by the inverse of the gradient variance (Second moment). Good for sparse data or complex landscapes.

**Q3: Why do we need to zero_grad() in PyTorch?**
> **Answer**: PyTorch accumulates gradients by default (`grad += new_grad`). This is useful for RNNs or virtual large batches. But for standard training, we must clear old gradients before the next backward pass, otherwise they sum up and lead to incorrect updates.

---

## 5. Further Reading
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
- [Visualizing Optimizers](https://www.deeplearning.ai/ai-notes/optimization/)
