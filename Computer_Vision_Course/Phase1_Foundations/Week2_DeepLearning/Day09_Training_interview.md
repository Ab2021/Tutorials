# Day 9 Interview Questions: Training Techniques

## Q1: Explain mixup and why it improves generalization.
**Answer:**

**Mixup:** Linear interpolation between training examples.

$$ \tilde{x} = \lambda x_i + (1-\lambda) x_j $$
$$ \tilde{y} = \lambda y_i + (1-\lambda) y_j $$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$, typically $\alpha \in [0.1, 0.4]$.

**Why it works:**
1. **Smooths decision boundaries:** Encourages linear behavior between classes
2. **Reduces overfitting:** Trains on infinite virtual examples
3. **Improves calibration:** Better confidence estimates
4. **Robustness:** More stable to adversarial examples

**Results:** Typically 1-2% accuracy improvement on ImageNet.

**Trade-off:** Slightly slower convergence (need more epochs).

## Q2: What is transfer learning and when should you use it?
**Answer:**

**Transfer Learning:** Use pre-trained model weights as initialization.

**Strategies:**

1. **Feature Extraction (Small dataset, similar domain):**
   - Freeze all layers except classifier
   - Train only final layer
   - Fast, prevents overfitting

2. **Fine-tuning (Medium dataset):**
   - Freeze early layers
   - Train later layers + classifier
   - Balance between speed and adaptation

3. **Full fine-tuning (Large dataset):**
   - Unfreeze all layers
   - Use lower learning rate
   - Best performance but slower

**When to use:**
- **Always** if dataset < 10K images
- **Usually** if dataset < 100K images
- **Optional** if dataset > 1M images

**Domain similarity matters:**
- Similar (ImageNet → CIFAR-10): Great results
- Different (ImageNet → Medical): Still helpful (low-level features)

## Q3: Compare different data augmentation techniques.
**Answer:**

| Technique | Type | Benefit | When to Use |
|-----------|------|---------|-------------|
| **Horizontal Flip** | Geometric | Simple, effective | Almost always |
| **Random Crop** | Geometric | Translation invariance | Images with context |
| **Color Jitter** | Photometric | Illumination robustness | Natural images |
| **Cutout** | Occlusion | Robustness to occlusion | Object recognition |
| **Mixup** | Mixing | Smooth boundaries | Classification |
| **CutMix** | Mixing | Localization + mixing | Detection/Segmentation |
| **AutoAugment** | Learned | Task-specific | When you have compute |

**Implementation priority:**
1. Flip + Crop (baseline)
2. Color jitter
3. Cutout/Random erasing
4. Mixup or CutMix
5. AutoAugment (if resources allow)

## Q4: Explain knowledge distillation and its benefits.
**Answer:**

**Knowledge Distillation:** Train small student to mimic large teacher.

**Loss:**
$$ L = \alpha L_{hard} + (1-\alpha) L_{soft} $$

**Hard loss:** Cross-entropy with true labels
$$ L_{hard} = -\sum_c y_c \log(p_c^{student}) $$

**Soft loss:** KL divergence with teacher (temperature $T$)
$$ L_{soft} = T^2 \cdot KL(softmax(\frac{z^{teacher}}{T}), softmax(\frac{z^{student}}{T})) $$

**Why temperature $T > 1$:**
- Softens probability distribution
- Reveals relative probabilities (dark knowledge)
- Example: $[0.9, 0.05, 0.05]$ → $[0.6, 0.2, 0.2]$ at $T=3$

**Benefits:**
1. **Model compression:** 10× smaller model, 95% of accuracy
2. **Faster inference:** Deploy on mobile/edge
3. **Better than training from scratch:** Teacher provides richer signal

**Typical:** $\alpha=0.3, T=3$

## Q5: What is label smoothing and why does it help?
**Answer:**

**Label Smoothing:** Soften one-hot labels.

**Hard labels:** $y = [0, 0, 1, 0, 0]$
**Soft labels:** $y = [0.025, 0.025, 0.9, 0.025, 0.025]$

**Formula:**
$$ y_i^{LS} = (1 - \epsilon) y_i + \frac{\epsilon}{K} $$

Typically $\epsilon = 0.1$.

**Why it helps:**
1. **Prevents overconfidence:** Model doesn't push logits to infinity
2. **Better calibration:** Confidence matches accuracy
3. **Regularization:** Encourages model to be less certain
4. **Robustness:** More stable predictions

**Results:** 0.5-1% accuracy improvement, better calibration.

**Trade-off:** Slightly lower training accuracy (but better validation).

## Q6: Explain learning rate warm-up and why it's important.
**Answer:**

**Warm-up:** Gradually increase LR from 0 to target over first few epochs.

**Linear warm-up:**
$$ \eta_t = \eta_{max} \cdot \frac{t}{T_{warmup}} $$

**Why it's important:**
1. **Large batch training:** Prevents divergence with large batches
2. **Adam optimizer:** Prevents bias in early iterations
3. **Transfer learning:** Allows model to adapt before aggressive updates
4. **Stability:** Gradients more reliable after a few iterations

**Typical:** 5-10 epochs warm-up, then cosine decay.

**Without warm-up:** Model may diverge or get stuck in poor local minimum.

## Q7: What is the difference between dropout and batch normalization?
**Answer:**

**Dropout:**
- **Purpose:** Regularization
- **Mechanism:** Randomly zero activations (p=0.5)
- **Training:** Stochastic
- **Inference:** Deterministic (scale by keep_prob)
- **Effect:** Prevents co-adaptation
- **Where:** Fully connected layers

**Batch Normalization:**
- **Purpose:** Stabilize training
- **Mechanism:** Normalize to mean=0, std=1
- **Training:** Uses batch statistics
- **Inference:** Uses running statistics
- **Effect:** Reduces internal covariate shift
- **Where:** After conv/linear, before activation

**Can use both:**
```python
x = conv(x)
x = batch_norm(x)
x = relu(x)
x = dropout(x, p=0.2)  # Lower p with BN
```

**Modern trend:** BN is more common, dropout less used in CNNs.

## Q8: Implement a learning rate scheduler with warm-up.
**Answer:**

```python
import math

class WarmupCosineScheduler:
    """Warm-up + cosine annealing scheduler."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, 
                 base_lr, min_lr=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warm-up
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr

# Usage
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, 
                                 total_epochs=100, base_lr=0.001)

for epoch in range(100):
    train_epoch(model, train_loader)
    lr = scheduler.step()
    print(f"Epoch {epoch}, LR: {lr:.6f}")
```

## Q9: How to handle class imbalance in training?
**Answer:**

**Problem:** Unequal class distribution (e.g., 90% class A, 10% class B).

**Solutions:**

**1. Weighted Loss:**
```python
# Calculate class weights
class_counts = torch.tensor([9000, 1000])  # Class A, B
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()  # Normalize

criterion = nn.CrossEntropyLoss(weight=class_weights.cuda())
```

**2. Oversampling:**
```python
from torch.utils.data import WeightedRandomSampler

# Sample weights (inverse of class frequency)
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

**3. Focal Loss:**
$$ FL(p_t) = -(1-p_t)^\gamma \log(p_t) $$

Focuses on hard examples.

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        
        return focal_loss.mean()
```

**4. Data Augmentation:** More augmentation for minority class.

## Q10: What is gradient accumulation and when to use it?
**Answer:**

**Gradient Accumulation:** Simulate large batch by accumulating gradients over multiple mini-batches.

**Why:**
- GPU memory limited (can't fit large batch)
- Want large effective batch size for stability

**How it works:**
1. Forward pass on mini-batch
2. Backward pass (accumulate gradients)
3. Repeat N times
4. Update weights
5. Zero gradients

**Effective batch size:** `mini_batch_size × accumulation_steps`

**Implementation:**
```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, labels) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps  # Normalize
    loss.backward()  # Accumulate gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update weights
        optimizer.zero_grad()  # Reset gradients
```

**Trade-off:**
- **Pro:** Same results as large batch, less memory
- **Con:** Slower (no parallelism across accumulation steps)

**When to use:**
- Training large models (BERT, GPT)
- Limited GPU memory
- Reproducing results with specific batch size
