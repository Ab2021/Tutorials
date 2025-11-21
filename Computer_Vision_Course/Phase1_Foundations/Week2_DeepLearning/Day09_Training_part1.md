# Day 9 Deep Dive: Advanced Training Strategies

## 1. Progressive Resizing

**Concept:** Train with small images first, then gradually increase resolution.

**Benefits:**
1. Faster initial training
2. Better generalization
3. Acts as regularization

**Implementation:**
```python
class ProgressiveTrainer:
    """Progressive resizing training."""
    
    def __init__(self, model, sizes=[128, 192, 224], epochs_per_size=[30, 20, 50]):
        self.model = model
        self.sizes = sizes
        self.epochs_per_size = epochs_per_size
    
    def train(self, dataset):
        for size, epochs in zip(self.sizes, self.epochs_per_size):
            print(f"Training with size {size}×{size} for {epochs} epochs")
            
            # Update transforms
            dataset.transform = T.Compose([
                T.Resize(size),
                T.RandomCrop(size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            # Train for specified epochs
            for epoch in range(epochs):
                self.train_epoch(dataset)
```

## 2. Knowledge Distillation

**Idea:** Train small "student" network to mimic large "teacher" network.

**Loss:**
$$ L = \alpha L_{CE}(y, \text{student}) + (1-\alpha) L_{KD}(\text{teacher}, \text{student}) $$

**KD Loss:**
$$ L_{KD} = T^2 \cdot KL(softmax(\frac{z_{teacher}}{T}), softmax(\frac{z_{student}}{T})) $$

where $T$ is temperature.

```python
class DistillationLoss(nn.Module):
    """Knowledge distillation loss."""
    
    def __init__(self, alpha=0.5, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # Hard loss (cross-entropy with true labels)
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft loss (KL divergence with teacher)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

def train_with_distillation(student, teacher, train_loader, epochs=100):
    """Train student network with teacher guidance."""
    teacher.eval()  # Teacher in eval mode
    student.train()
    
    criterion = DistillationLoss(alpha=0.3, temperature=3.0)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            
            # Student forward
            student_logits = student(inputs)
            
            # Distillation loss
            loss = criterion(student_logits, teacher_logits, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## 3. Gradient Accumulation

**Problem:** Limited GPU memory for large batch sizes.
**Solution:** Accumulate gradients over multiple mini-batches.

```python
def train_with_gradient_accumulation(model, train_loader, accumulation_steps=4):
    """Simulate large batch size with gradient accumulation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    optimizer.zero_grad()
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Normalize loss (average over accumulation steps)
        loss = loss / accumulation_steps
        
        # Backward
        loss.backward()
        
        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

**Effective batch size:** `batch_size × accumulation_steps`

## 4. Mixed Precision Training

**Idea:** Use FP16 for speed, FP32 for stability.

**Benefits:**
- 2-3× faster training
- 2× less memory
- Minimal accuracy loss

```python
from torch.cuda.amp import autocast, GradScaler

def train_mixed_precision(model, train_loader, epochs=100):
    """Mixed precision training with automatic mixed precision (AMP)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            
            # Forward in FP16
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
```

## 5. Early Stopping

**Prevent overfitting:** Stop when validation performance plateaus.

```python
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

# Usage
early_stopping = EarlyStopping(patience=10)

for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

## 6. Curriculum Learning

**Idea:** Train on easy examples first, gradually increase difficulty.

**Strategies:**
1. **Data ordering:** Sort by difficulty metric
2. **Loss-based:** Start with low-loss examples
3. **Self-paced:** Let model choose examples

```python
class CurriculumSampler:
    """Curriculum learning sampler."""
    
    def __init__(self, dataset, difficulty_scores):
        self.dataset = dataset
        self.difficulty_scores = difficulty_scores
        self.current_percentile = 0.3  # Start with easiest 30%
    
    def get_subset(self):
        """Get current curriculum subset."""
        threshold = np.percentile(self.difficulty_scores, 
                                 self.current_percentile * 100)
        indices = np.where(self.difficulty_scores <= threshold)[0]
        return Subset(self.dataset, indices)
    
    def update_curriculum(self, epoch, total_epochs):
        """Gradually increase difficulty."""
        self.current_percentile = min(1.0, 0.3 + 0.7 * (epoch / total_epochs))
```

## 7. Test-Time Augmentation (TTA)

**Idea:** Average predictions over multiple augmented versions.

```python
def test_time_augmentation(model, image, n_augmentations=10):
    """Test-time augmentation for better predictions."""
    model.eval()
    
    augmentations = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
    ])
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_augmentations):
            # Augment
            aug_image = augmentations(image)
            
            # Predict
            output = model(aug_image.unsqueeze(0).cuda())
            predictions.append(F.softmax(output, dim=1))
        
        # Average predictions
        avg_prediction = torch.stack(predictions).mean(dim=0)
    
    return avg_prediction
```

## 8. Model Ensembling

**Combine multiple models for better performance.**

```python
class Ensemble(nn.Module):
    """Ensemble of multiple models."""
    
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
    
    def forward(self, x):
        outputs = []
        for model, weight in zip(self.models, self.weights):
            output = F.softmax(model(x), dim=1)
            outputs.append(output * weight)
        
        return torch.stack(outputs).sum(dim=0)

# Usage
model1 = ResNet50()
model2 = EfficientNetB0()
model3 = VGG16()

ensemble = Ensemble([model1, model2, model3], weights=[0.4, 0.4, 0.2])
```

## 9. Hyperparameter Optimization

**Grid Search:**
```python
from itertools import product

param_grid = {
    'lr': [0.001, 0.0001, 0.00001],
    'batch_size': [32, 64, 128],
    'weight_decay': [1e-4, 1e-5, 1e-6],
}

best_acc = 0
best_params = None

for lr, bs, wd in product(*param_grid.values()):
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    acc = train_and_evaluate(model, optimizer, batch_size=bs)
    
    if acc > best_acc:
        best_acc = acc
        best_params = {'lr': lr, 'batch_size': bs, 'weight_decay': wd}
```

**Random Search:**
```python
import random

def random_search(n_trials=50):
    best_acc = 0
    best_params = None
    
    for _ in range(n_trials):
        params = {
            'lr': 10 ** random.uniform(-5, -2),
            'batch_size': random.choice([32, 64, 128, 256]),
            'weight_decay': 10 ** random.uniform(-6, -3),
        }
        
        acc = train_and_evaluate(**params)
        
        if acc > best_acc:
            best_acc = acc
            best_params = params
    
    return best_params
```

## 10. Debugging Training

**Common issues and solutions:**

**1. Loss not decreasing:**
- Check learning rate (too high/low)
- Verify data loading
- Check loss function
- Inspect gradients

**2. Overfitting:**
- Add regularization (dropout, weight decay)
- More data augmentation
- Reduce model capacity
- Early stopping

**3. Underfitting:**
- Increase model capacity
- Train longer
- Reduce regularization
- Check data quality

**Gradient monitoring:**
```python
def check_gradients(model):
    """Monitor gradient statistics."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm={grad_norm:.4f}")
            
            if grad_norm > 100:
                print(f"WARNING: Large gradient in {name}")
            elif grad_norm < 1e-7:
                print(f"WARNING: Vanishing gradient in {name}")
```

## Summary
Advanced training strategies including progressive resizing, knowledge distillation, mixed precision, and proper debugging enable efficient and effective model training.
