# Day 9: Training Techniques

## Overview
Effective training requires more than just a good architecture. This lesson covers data augmentation, transfer learning, regularization, and optimization strategies that enable models to generalize well.

## 1. Data Augmentation

### Why Augmentation?
**Problem:** Limited training data leads to overfitting.
**Solution:** Artificially expand dataset with label-preserving transformations.

### Common Transformations

**Geometric:**
- Horizontal/vertical flips
- Random rotations
- Random crops
- Affine transformations
- Perspective transforms

**Color:**
- Brightness/contrast adjustment
- Hue/saturation shifts
- Color jittering
- Grayscale conversion

**Advanced:**
- Cutout/Random erasing
- Mixup
- CutMix
- AutoAugment

### Implementation

```python
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F

class Augmentation:
    """Comprehensive data augmentation pipeline."""
    
    def __init__(self, mode='train'):
        if mode == 'train':
            self.transform = T.Compose([
                T.RandomResizedCrop(224, scale=(0.08, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.4, contrast=0.4, 
                             saturation=0.4, hue=0.1),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.5, scale=(0.02, 0.33)),
            ])
        else:  # val/test
            self.transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            ])
    
    def __call__(self, img):
        return self.transform(img)
```

### Advanced Augmentation

**Mixup:**
$$ \tilde{x} = \lambda x_i + (1-\lambda) x_j $$
$$ \tilde{y} = \lambda y_i + (1-\lambda) y_j $$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$, typically $\alpha=0.2$.

```python
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

**CutMix:**
Cut and paste patches between images.

```python
def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Random box
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y, y[index], lam
```

## 2. Transfer Learning

### Concept
**Pre-training:** Train on large dataset (ImageNet)
**Fine-tuning:** Adapt to target task with smaller dataset

**Why it works:**
- Early layers learn general features (edges, textures)
- Later layers learn task-specific features
- Pre-trained weights provide better initialization

### Strategies

**1. Feature Extraction (Frozen backbone):**
```python
# Load pre-trained model
model = torchvision.models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Only train classifier
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

**2. Fine-tuning (Unfreeze some layers):**
```python
# Load pre-trained model
model = torchvision.models.resnet50(pretrained=True)

# Replace classifier
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Fine-tune strategy
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Phase 1: Train only classifier
set_parameter_requires_grad(model, True)
model.fc.requires_grad = True
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
# Train for a few epochs...

# Phase 2: Fine-tune entire network
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower LR
```

**3. Discriminative Learning Rates:**
```python
# Different LR for different layers
params = [
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
]
optimizer = torch.optim.Adam(params)
```

## 3. Regularization Techniques

### L2 Regularization (Weight Decay)
$$ L_{total} = L_{data} + \frac{\lambda}{2} \sum_i w_i^2 $$

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

### Dropout
```python
class RegularizedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
```

### Label Smoothing
**Hard labels:** $y = [0, 0, 1, 0, 0]$
**Soft labels:** $y = [0.025, 0.025, 0.9, 0.025, 0.025]$

$$ y_i^{LS} = (1 - \epsilon) y_i + \frac{\epsilon}{K} $$

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
```

## 4. Learning Rate Schedules

### Step Decay
```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

### Cosine Annealing
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

### Warm-up + Cosine
```python
from torch.optim.lr_scheduler import LambdaLR
import math

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)
```

## 5. Complete Training Loop

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, num_epochs=100, device='cuda'):
    """Complete training loop with best practices."""
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Mixup augmentation
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += (lam * predicted.eq(labels_a).sum().float()
                            + (1 - lam) * predicted.eq(labels_b).sum().float())
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return history
```

## Summary
Effective training combines data augmentation, transfer learning, regularization, and proper learning rate scheduling to achieve optimal performance.

**Next:** Week 2 Review.
