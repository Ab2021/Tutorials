# Day 8 Deep Dive: Advanced Training Techniques

## 1. Label Smoothing
**Problem:** One-hot targets $[0, 1, 0]$ force the model to be extremely confident (logits $\to \infty$), leading to overfitting.

**Solution:** Soften targets.
$$ y_{new} = (1 - \epsilon) y_{onehot} + \frac{\epsilon}{K} $$
*   Example ($\epsilon=0.1, K=3$): $[0, 1, 0] \to [0.033, 0.933, 0.033]$.
*   Prevents over-confidence and improves calibration.

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

## 2. MixUp and CutMix

### MixUp
Linear interpolation of two images and their labels.
$$ x' = \lambda x_i + (1-\lambda) x_j $$
$$ y' = \lambda y_i + (1-\lambda) y_j $$
*   $\lambda \sim \text{Beta}(\alpha, \alpha)$.
*   Encourages linear behavior in-between classes.

### CutMix
Patches a region of one image onto another.
*   Labels are mixed proportional to the area.
*   Forces model to look at less discriminative parts.

## 3. Gradient Clipping
Prevents exploding gradients (common in RNNs, but also deep CNNs).
*   If $||g|| > \text{threshold}$, scale $g$ down.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 4. Batch Size vs Learning Rate
**Linear Scaling Rule:** When increasing batch size by $k$, increase LR by $k$.
*   Large batches provide more accurate gradient estimates, allowing larger steps.
*   **Warmup:** Start with small LR and linearly increase to target LR over first few epochs. Crucial for large batch training.

## 5. Normalization Layers

### Batch Normalization (BN)
Normalizes across batch dimension $(N, H, W)$.
*   Dependent on batch size (bad for small batches).

### Layer Normalization (LN)
Normalizes across channel dimension $(C, H, W)$.
*   Independent of batch size. Used in Transformers.

### Group Normalization (GN)
Divides channels into groups and normalizes.
*   Good compromise for detection/segmentation (small batches).

## Summary
Advanced techniques like Label Smoothing, MixUp, and Warmup are standard in SOTA training recipes (e.g., "Bag of Tricks for Image Classification").
