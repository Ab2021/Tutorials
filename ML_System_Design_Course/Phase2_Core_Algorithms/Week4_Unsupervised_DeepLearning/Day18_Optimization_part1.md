# Day 18 (Part 1): Advanced Optimization & Normalization

> **Phase**: 6 - Deep Dive
> **Topic**: Training Stability
> **Focus**: AdamW, Normalization Layers, and Schedulers
> **Reading Time**: 60 mins

---

## 1. Adam vs. AdamW

Why did everyone switch to AdamW in 2017?

### 1.1 The Bug in Adam
*   **L2 Regularization**: Adds $\lambda w^2$ to Loss. Gradient is $2\lambda w$.
*   **Weight Decay**: Updates weights: $w = w - \eta \lambda w$.
*   **In SGD**: These are identical.
*   **In Adam**: Adam scales gradients by $1/\sqrt{v}$. If you add L2 to the gradient, it gets scaled! This means weight decay is weaker for parameters with large gradients.
*   **AdamW**: Decouples weight decay. Apply the gradient update first, *then* decay the weights directly. $w = w - \eta (\text{AdamStep}) - \eta \lambda w$.

---

## 2. Normalization Layers

### 2.1 Batch Norm (BN)
*   Normalizes across the Batch dimension $(N, C, H, W) \to (\cdot, C, H, W)$.
*   **Pros**: Allows higher LR.
*   **Cons**: Depends on Batch Size. Fails for RNNs.

### 2.2 Layer Norm (LN)
*   Normalizes across the Feature dimension $(N, C, H, W) \to (N, \cdot, \cdot, \cdot)$.
*   **Pros**: Independent of Batch Size. Works for RNNs/Transformers.
*   **Cons**: Slightly worse than BN for CNNs.

### 2.3 RMSNorm (Root Mean Square Norm)
*   Like Layer Norm, but skips mean subtraction. Just scales by variance.
*   **Used in**: Llama, T5. Faster.

---

## 3. Tricky Interview Questions

### Q1: Why does Adam need Bias Correction?
> **Answer**:
> *   $m_0 = 0, v_0 = 0$.
> *   First update: $m_1 = 0.9 m_0 + 0.1 g = 0.1 g$.
> *   This is biased towards 0 (too small).
> *   Correction: $\hat{m}_t = m_t / (1 - \beta_1^t)$. At $t=1$, divide by $0.1$, scaling it back to $1.0 g$.

### Q2: Explain Cosine Annealing with Warm Restarts (SGDR).
> **Answer**:
> *   **Annealing**: Decrease LR following a cosine curve (smooth drop).
> *   **Restart**: Suddenly reset LR to max.
> *   **Why?**: Helps escape local minima. The sudden jump pushes the model out of a sharp basin into a broader (better generalizing) basin.

### Q3: Why does Batch Norm fail with small batch sizes?
> **Answer**:
> *   BN estimates mean/var from the batch.
> *   If $N=2$, the estimate is extremely noisy. The model learns to rely on the specific statistics of those 2 examples (overfitting/instability).
> *   **Fix**: Group Norm or Layer Norm.

---

## 4. Practical Edge Case: Gradient Accumulation
*   **Scenario**: You want Batch Size 128, but GPU fits 16.
*   **Code**:
    ```python
    optimizer.zero_grad()
    for i, (x, y) in enumerate(loader):
        loss = model(x, y) / 8  # Normalize loss
        loss.backward()
        if (i + 1) % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()
    ```
*   **Trap**: Don't forget to divide loss by accumulation steps! Otherwise gradients sum up to 8x magnitude.

