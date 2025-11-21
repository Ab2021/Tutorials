# Day 10 Interview Questions: Week 2 Review

## Q1: Explain the complete forward and backward pass in a CNN.
**Answer:**

**Forward Pass:**
```
Input (3×224×224)
    ↓ Conv1 (64 filters, 3×3)
Feature maps (64×224×224)
    ↓ ReLU
Activated (64×224×224)
    ↓ MaxPool (2×2)
Pooled (64×112×112)
    ↓ ... (more layers)
    ↓ Global Average Pool
Vector (512×1×1)
    ↓ Flatten
Vector (512)
    ↓ FC (512→10)
Logits (10)
    ↓ Softmax
Probabilities (10)
```

**Backward Pass:**
```
Loss gradient
    ↓ Softmax gradient: dL/dz = p - y
FC gradient: dL/dW = a^T · dL/dz
    ↓ Backprop through FC
Pool gradient: Route to max positions
    ↓ Backprop through pool
ReLU gradient: Mask where input > 0
    ↓ Backprop through ReLU
Conv gradient: dL/dW = input * dL/doutput (convolution)
    ↓ Continue to input
```

**Key equations:**
- Conv forward: $Z = X * W + b$
- Conv backward: $\frac{\partial L}{\partial W} = X * \frac{\partial L}{\partial Z}$

## Q2: Compare ResNet-50 vs EfficientNet-B0 for production.
**Answer:**

| Aspect | ResNet-50 | EfficientNet-B0 |
|--------|-----------|-----------------|
| **Parameters** | 25.6M | 5.3M (4.8× fewer) |
| **FLOPs** | 4.1B | 390M (10.5× fewer) |
| **Accuracy** | 76.2% | 77.1% (0.9% better) |
| **Inference (GPU)** | ~10ms | ~8ms |
| **Inference (CPU)** | ~100ms | ~60ms |
| **Training time** | Faster | Slower (complex ops) |
| **Transfer learning** | Excellent | Good |
| **Pretrained weights** | Widely available | Available |

**When to use:**
- **ResNet-50:** 
  - Need fast training
  - Transfer learning priority
  - Established baseline
  
- **EfficientNet-B0:**
  - Mobile/edge deployment
  - Limited compute budget
  - Want best accuracy/size ratio

**Recommendation:** Start with ResNet-50 (easier), switch to EfficientNet if deployment constraints require it.

## Q3: Design a training strategy for a small dataset (<5K images).
**Answer:**

**Challenge:** High risk of overfitting.

**Strategy:**

**1. Transfer Learning (Critical):**
```python
# Use pre-trained model
model = torchvision.models.resnet50(pretrained=True)

# Freeze early layers
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

**2. Aggressive Data Augmentation:**
```python
transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),  # If applicable
    T.RandomRotation(30),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.5),
])
```

**3. Strong Regularization:**
- Dropout: 0.5
- Weight decay: 1e-3
- Label smoothing: 0.1
- Early stopping: patience=20

**4. Training Protocol:**
- Small batch size: 16-32 (more updates)
- Lower learning rate: 1e-4
- More epochs: 200-300
- Cross-validation: 5-fold

**5. Ensemble:**
Train 3-5 models with different:
- Random seeds
- Augmentation strategies
- Architectures (ResNet, EfficientNet, VGG)

Average predictions for final output.

## Q4: Explain batch normalization's effect during training vs inference.
**Answer:**

**Training:**
```python
# Compute batch statistics
mean = x.mean(dim=[0, 2, 3])  # Per channel
var = x.var(dim=[0, 2, 3])

# Normalize
x_norm = (x - mean) / sqrt(var + eps)

# Scale and shift (learnable)
y = gamma * x_norm + beta

# Update running statistics (momentum=0.1)
running_mean = 0.9 * running_mean + 0.1 * mean
running_var = 0.9 * running_var + 0.1 * var
```

**Inference:**
```python
# Use running statistics (no batch dependency)
x_norm = (x - running_mean) / sqrt(running_var + eps)
y = gamma * x_norm + beta
```

**Key differences:**
1. **Statistics source:** Batch (train) vs Running (inference)
2. **Deterministic:** Stochastic (train) vs Deterministic (inference)
3. **Batch size:** Any (train) vs Can be 1 (inference)

**Common mistake:** Forgetting `model.eval()` → uses batch stats at inference → inconsistent predictions.

## Q5: How to debug a model that's not learning?
**Answer:**

**Systematic debugging checklist:**

**1. Data Issues:**
```python
# Visualize batch
for images, labels in train_loader:
    plt.imshow(images[0].permute(1, 2, 0))
    print(f"Label: {labels[0]}")
    break

# Check label distribution
print(np.bincount(all_labels))

# Verify normalization
print(f"Mean: {images.mean()}, Std: {images.std()}")
```

**2. Loss Issues:**
```python
# Check initial loss
# Should be -log(1/num_classes) for classification
expected_loss = -np.log(1 / num_classes)
print(f"Expected: {expected_loss}, Actual: {initial_loss}")

# Monitor loss
if loss == nan:
    print("NaN loss - check learning rate or gradients")
elif loss not decreasing:
    print("Loss plateaued - check learning rate or architecture")
```

**3. Gradient Issues:**
```python
# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 100:
            print(f"Exploding gradient in {name}: {grad_norm}")
        elif grad_norm < 1e-7:
            print(f"Vanishing gradient in {name}: {grad_norm}")
```

**4. Learning Rate:**
```python
# Try learning rate finder
lrs = []
losses = []

for lr in np.logspace(-6, -1, 100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = train_step(model, batch)
    lrs.append(lr)
    losses.append(loss.item())

plt.plot(lrs, losses)
plt.xscale('log')
# Choose LR where loss decreases fastest
```

**5. Overfit Single Batch:**
```python
# Model should overfit single batch
single_batch = next(iter(train_loader))

for epoch in range(1000):
    loss = train_step(model, single_batch)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# If can't overfit → model capacity or bug
# If overfits → add regularization
```

## Q6: Implement mixed precision training from scratch.
**Answer:**

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    """Mixed precision training implementation."""
    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
    
    def train_step(self, inputs, labels):
        """Single training step with mixed precision."""
        # Forward pass in FP16
        with autocast():
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)
        
        # Backward pass with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # Unscale gradients and clip
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step with scaling
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

# Usage
trainer = MixedPrecisionTrainer(model, optimizer)

for inputs, labels in train_loader:
    loss = trainer.train_step(inputs.cuda(), labels.cuda())
```

**Benefits:**
- 2-3× faster training
- 2× less memory
- Minimal accuracy loss

**When to use:**
- GPU with Tensor Cores (V100, A100, RTX 20xx+)
- Large models
- Large batch sizes

## Q7: Explain the trade-off between batch size and learning rate.
**Answer:**

**Relationship:**
$$ \text{LR}_{large} = \text{LR}_{small} \times \frac{\text{BS}_{large}}{\text{BS}_{small}} $$

**Linear Scaling Rule:** When increasing batch size by k, increase LR by k.

**Why:**
- Larger batch → more stable gradient → can use larger LR
- Smaller batch → noisier gradient → need smaller LR

**Example:**
- Batch size 32, LR 0.001
- Batch size 256 (8×), LR 0.008 (8×)

**Caveats:**
1. **Warm-up needed:** Don't start with large LR
2. **Generalization gap:** Very large batches (>1024) may hurt generalization
3. **Memory limit:** Can't always increase batch size

**Practical:**
```python
base_lr = 0.001
base_batch_size = 32
current_batch_size = 256

# Linear scaling
lr = base_lr * (current_batch_size / base_batch_size)

# With warm-up
def get_lr(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        return lr
```

## Q8: Design an experiment to compare two architectures.
**Answer:**

**Controlled Experiment:**

```python
import wandb

def run_experiment(architecture, config):
    """Run single experiment with logging."""
    # Initialize tracking
    wandb.init(project='architecture_comparison', config=config)
    
    # Set seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create model
    if architecture == 'resnet50':
        model = torchvision.models.resnet50(pretrained=config['pretrained'])
    elif architecture == 'efficientnet_b0':
        model = torchvision.models.efficientnet_b0(pretrained=config['pretrained'])
    
    # Train
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=config['epochs'],
        lr=config['lr']
    )
    
    # Log results
    wandb.log({
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': max(history['val_acc']),
        'training_time': training_time,
        'num_parameters': sum(p.numel() for p in model.parameters()),
    })
    
    wandb.finish()
    
    return history

# Run experiments
configs = {
    'epochs': 100,
    'lr': 0.001,
    'batch_size': 32,
    'pretrained': True,
}

# Multiple seeds for statistical significance
for seed in [42, 123, 456, 789, 101]:
    configs['seed'] = seed
    
    # ResNet-50
    run_experiment('resnet50', configs)
    
    # EfficientNet-B0
    run_experiment('efficientnet_b0', configs)

# Analyze results
# Compare mean ± std across seeds
```

**Metrics to compare:**
1. **Accuracy:** Validation accuracy
2. **Speed:** Training time, inference time
3. **Efficiency:** Parameters, FLOPs
4. **Robustness:** Performance across seeds
5. **Transfer:** Performance on downstream tasks

## Q9: What is the effect of different initialization schemes?
**Answer:**

**Xavier/Glorot (tanh/sigmoid):**
$$ W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in} + n_{out}}}) $$

**He (ReLU):**
$$ W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in}}}) $$

**Why it matters:**
- Maintains variance of activations across layers
- Prevents vanishing/exploding gradients

**Experiment:**
```python
# Bad initialization (all zeros)
for layer in model.modules():
    if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.weight, 0)

# Result: All neurons learn same thing (symmetry problem)

# Bad initialization (too large)
for layer in model.modules():
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0, std=10)

# Result: Exploding activations/gradients

# Good initialization (He)
for layer in model.modules():
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# Result: Stable training
```

## Q10: Explain the complete lifecycle of a production CV model.
**Answer:**

**1. Problem Definition:**
- Define task (classification, detection, etc.)
- Collect requirements (accuracy, latency, size)
- Gather data

**2. Data Preparation:**
- Clean and label data
- Split: train/val/test (70/15/15)
- Analyze distribution
- Create data loaders

**3. Model Development:**
- Choose baseline architecture
- Implement training pipeline
- Experiment with hyperparameters
- Track experiments (TensorBoard, W&B)

**4. Training:**
- Transfer learning from ImageNet
- Data augmentation
- Regularization
- Monitor overfitting

**5. Evaluation:**
- Test set performance
- Error analysis
- Confusion matrix
- Per-class metrics

**6. Optimization:**
- Quantization (INT8)
- Pruning (30-50%)
- Knowledge distillation
- Export to ONNX/TorchScript

**7. Deployment:**
- Containerize (Docker)
- Create API (Flask/FastAPI)
- Load balancing
- Monitoring

**8. Monitoring:**
- Inference latency
- Prediction distribution
- Data drift detection
- Model retraining triggers

**9. Maintenance:**
- Collect edge cases
- Retrain periodically
- A/B testing new versions
- Rollback if needed

**Tools:**
- Training: PyTorch, TensorFlow
- Tracking: W&B, MLflow
- Deployment: Docker, Kubernetes
- Serving: TorchServe, TensorFlow Serving
- Monitoring: Prometheus, Grafana
