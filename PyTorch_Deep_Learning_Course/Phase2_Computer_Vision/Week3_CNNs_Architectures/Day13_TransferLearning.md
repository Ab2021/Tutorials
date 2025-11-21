# Day 13: Transfer Learning - Theory & Implementation

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: Fine-tuning, Feature Extraction, and Domain Adaptation

## 1. Theoretical Foundation: Standing on the Shoulders of Giants

Training from scratch requires:
1.  Massive Data (ImageNet has 1.2M images).
2.  Massive Compute (Weeks on GPUs).
3.  Architecture Engineering.

**Transfer Learning**: Take a model trained on Source Domain (ImageNet) and adapt it to Target Domain (Medical X-Rays).
*   **Low-level features** (Edges, Textures) are universal.
*   **High-level features** (Eyes, Wheels) are specific.

### Strategies
1.  **Feature Extraction**: Freeze backbone, train only the classifier (Linear Head).
    *   Use when: Small dataset, similar domain.
2.  **Fine-Tuning**: Unfreeze backbone (or parts of it) and train with low LR.
    *   Use when: Large dataset, or different domain.

## 2. Implementation: The Workflow

```python
import torch
import torch.nn as nn
from torchvision import models

# 1. Load Pre-trained Model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 2. Freeze Backbone (Feature Extraction)
for param in model.parameters():
    param.requires_grad = False

# 3. Replace Head
# ResNet's final layer is named 'fc' and has 2048 inputs
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) # 10 classes for our task

# 4. Train
# Only model.fc parameters will be updated because others have requires_grad=False
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

## 3. Advanced Fine-Tuning

After training the head, we can "thaw" the backbone to adapt features.

```python
# Unfreeze all
for param in model.parameters():
    param.requires_grad = True

# Differential Learning Rates
# Low LR for backbone (preserve knowledge), High LR for head (learn fast)
optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-6},
    {'params': model.layer2.parameters(), 'lr': 1e-6},
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])
```

## 4. Domain Adaptation

What if Source (Real Photos) and Target (Cartoons) are very different?
**Discriminative Fine-tuning**:
*   Train a domain classifier to distinguish Source vs Target.
*   Train the backbone to *fool* the domain classifier (Adversarial).
*   Result: Backbone learns domain-invariant features.
