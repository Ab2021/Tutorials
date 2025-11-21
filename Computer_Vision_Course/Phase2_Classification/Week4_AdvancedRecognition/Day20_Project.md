# Day 20: Phase 2 Project - Fine-Grained Classification

## 1. Project Overview
**Goal:** Build a high-performance classifier for the **CUB-200-2011** dataset (200 bird species).
**Challenge:** Fine-grained classification requires distinguishing subtle differences (beak shape, wing pattern) between very similar classes.
**Techniques:**
*   Transfer Learning (ResNet50 / ViT-B/16).
*   Data Augmentation (RandomCrop, Rotation, ColorJitter).
*   Learning Rate Scheduling (Cosine Annealing).
*   Label Smoothing.

## 2. Dataset Setup
*   **Dataset:** Caltech-UCSD Birds-200-2011.
*   **Images:** 11,788.
*   **Classes:** 200.
*   **Split:** ~30 images per class for training, ~30 for testing.

```python
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 1. Data Augmentation
train_transforms = transforms.Compose([
    transforms.Resize((448, 448)), # Higher resolution for fine details
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Load Data (Assuming downloaded)
# train_dataset = datasets.ImageFolder('CUB_200_2011/train', transform=train_transforms)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
```

## 3. Model Selection
We will use **ResNet-50** pretrained on ImageNet.
*   **Why?** Strong baseline, good balance of speed/accuracy.
*   **Modification:** Change final FC layer to 200 outputs.

```python
import torch.nn as nn

model = models.resnet50(pretrained=True)

# Freeze early layers (optional, usually better to fine-tune all for fine-grained)
# for param in model.parameters():
#     param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 200)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

## 4. Training Loop
*   **Loss:** CrossEntropyLoss with Label Smoothing.
*   **Optimizer:** SGD with Momentum (0.9) and Weight Decay (1e-4).
*   **Scheduler:** Cosine Annealing.

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    scheduler.step()
    print(f"Epoch {epoch}: Loss {running_loss/len(train_loader):.4f}, Acc {100.*correct/total:.2f}%")
```

## 5. Evaluation
*   Top-1 Accuracy.
*   Confusion Matrix (to see which birds are confused).
*   Visualize Grad-CAM to see what parts of the bird the model focuses on.

## Summary
This project consolidates Phase 2 skills: Transfer Learning, Augmentation, and Optimization. Achieving >80% accuracy on CUB-200 is a strong milestone.
