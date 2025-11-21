# Day 8: Training CNNs

## 1. Loss Functions
The objective function that the network minimizes.

### Cross-Entropy Loss (Classification)
Standard for multi-class classification.
$$ L = -\sum_{c=1}^C y_c \log(p_c) $$
*   $y_c$: Ground truth (one-hot).
*   $p_c$: Predicted probability (Softmax).

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
```

### Focal Loss (Class Imbalance)
Down-weights easy examples to focus on hard ones.
$$ FL(p_t) = -(1 - p_t)^\gamma \log(p_t) $$
*   $\gamma$: Focusing parameter (e.g., 2).

## 2. Optimization Algorithms

### SGD (Stochastic Gradient Descent)
Updates weights based on gradient of a mini-batch.
$$ w_{t+1} = w_t - \eta \nabla L(w_t) $$

### SGD with Momentum
Accumulates past gradients to smooth updates and accelerate convergence.
$$ v_{t+1} = \mu v_t - \eta \nabla L(w_t) $$
$$ w_{t+1} = w_t + v_{t+1} $$

### Adam (Adaptive Moment Estimation)
Combines Momentum and RMSProp (adaptive learning rates).
*   Maintains running average of gradients ($m_t$) and squared gradients ($v_t$).
*   **Standard choice:** `lr=3e-4` or `1e-3`.

### AdamW (Adam with Weight Decay)
Decouples weight decay from gradient update.
*   **Best practice** for modern Transformers and CNNs.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

## 3. Regularization
Prevents overfitting.

### Dropout
Randomly zeroes out neurons during training with probability $p$.
*   Forces network to learn redundant representations.
*   **Inference:** Scale weights by $(1-p)$ (handled automatically by PyTorch).

### Weight Decay (L2 Regularization)
Adds penalty term $\lambda ||w||^2$ to loss.
*   Keeps weights small.

### Data Augmentation
Artificially expands dataset.
*   **Geometric:** Flip, Rotate, Scale, Crop.
*   **Color:** Jitter Brightness, Contrast, Saturation.
*   **Advanced:** MixUp, CutMix.

```python
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])
```

## 4. Learning Rate Schedules
Adjusting LR during training is crucial.

### Step Decay
Reduce LR by factor $\gamma$ every $N$ epochs.

### Cosine Annealing
Smoothly decreases LR following a cosine curve.
*   Often used with Warmup.

### One Cycle Policy
Increase LR to max, then decrease.
*   Super-convergence.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

## Summary
Training a CNN requires choosing the right Loss (CrossEntropy), Optimizer (AdamW), Regularization (Dropout, Augmentation), and Scheduler (Cosine) to ensure convergence and generalization.
