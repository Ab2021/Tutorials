# Day 9: Transfer Learning & Fine-Tuning

## 1. Introduction to Transfer Learning
**Concept:** Take a model trained on a large dataset (Source: ImageNet) and adapt it to a smaller dataset (Target: Custom Data).
**Why?**
*   Deep networks need massive data.
*   Low-level features (edges, textures) are universal.

## 2. Strategies

### A. Feature Extraction (Fixed Feature Extractor)
1.  Load pretrained model (e.g., ResNet50).
2.  **Freeze** all convolutional layers (weights don't change).
3.  Replace the final Fully Connected (FC) layer with a new one (matching target classes).
4.  Train **only** the new FC layer.

```python
model = models.resnet50(pretrained=True)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Replace head
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes) # New layer is trainable by default
```

### B. Fine-Tuning
1.  Initialize with pretrained weights.
2.  Replace head.
3.  Train the **entire** network (or specific blocks) with a **low learning rate**.
*   Allows weights to adjust slightly to the new domain.

## 3. When to use what?

| Target Data Size | Target Data Similarity | Strategy |
| :--- | :--- | :--- |
| Small | Similar | Feature Extraction (Linear Probe) |
| Large | Similar | Fine-Tune all layers |
| Small | Different | Fine-Tune later layers / Train from scratch |
| Large | Different | Fine-Tune all / Train from scratch |

## 4. Advanced Strategies

### Progressive Unfreezing
1.  Train head only.
2.  Unfreeze last conv block, train.
3.  Unfreeze next block, train.
*   Prevents "catastrophic forgetting" / destroying pretrained weights with large gradients.

### Discriminative Learning Rates
Use lower LR for early layers (general features) and higher LR for later layers (task-specific features).
```python
optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.head.parameters(), 'lr': 1e-3}
])
```

## 5. Domain Adaptation
When Source (Synthetic) and Target (Real) distributions differ.
*   **Goal:** Align feature distributions.
*   **DANN (Domain Adversarial Neural Network):** Gradient reversal layer to make features indistinguishable between domains.

## Summary
Transfer learning is the default approach in CV. Always start with a pretrained model (ResNet, EfficientNet, ViT) unless you have millions of images.
