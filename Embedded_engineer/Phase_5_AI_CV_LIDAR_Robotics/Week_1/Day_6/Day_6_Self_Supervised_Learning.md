# Day 6: Self-Supervised Learning for Robotics
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 1: Deep Learning Foundations

---

> **📝 Content Creator Instructions:**
> Labels are expensive (human annotation). Robot data is cheap (just log sensors). 
> - **Focus:** How to learn useful representations from unlabeled data using SimCLR, MoCo, MAE, and Time-Contrastive Learning.
> - **Code:** Implementation of SimCLR Contrastive Loss and a Masked Autoencoder pipeline.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** the Self-Supervised Learning (SSL) paradigm and why it is critical for robotics.
2.  **Implement** the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss used in SimCLR.
3.  **Build** a Masked Autoencoder (MAE) to learn visual features by reconstructing missing patches.
4.  **Apply** Time-Contrastive Networks (TCN) to learn robotic skills from video demonstrations without labels.
5.  **Pre-train** a ResNet backbone on a custom unlabelled robot dataset.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU (Required for SSL training, as batch sizes need to be large for contrastive methods).

### Software Environment
```bash
pip install torch torchvision
pip install lightly  # Excellent SSL library
pip install kornia   # Differentiable augmentations
```

### Prior Knowledge
- Autoencoders.
- Data Augmentation.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Label Bottleneck

Robots generate TBs of data per day. Labeling bounding boxes or segmentation masks for all frames is impossible.
 **Self-Supervised Learning (SSL)** generates "labels" from the data structure itself (Pretext Tasks).

#### 1.1 Contrastive Learning (SimCLR / MoCo)
**Idea:** Two augmented views of the *same* image (Positive Pair) should have similar feature representations. Two views of *different* images (Negative Pair) should have distant representations.

**SimCLR Architecture:**
1.  **Augmentation:** Crop, Color Jitter, Blur.
2.  **Encoder:** $h = f(x)$ (ResNet).
3.  **Projection Head:** $z = g(h)$ (MLP). Contrastive loss is calculated on $z$, not $h$.
4.  **Loss (InfoNCE):**
    $$ \mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)} $$
    *   Numerator: Attract Positives.
    *   Denominator: Repel Negatives.

#### 1.2 Masked Autoencoders (MAE) - The ViT Revolution
**Idea:** Mask out 75% of the image patches. Ask the network to reconstruct the missing pixels.
*   **Encoder:** Processes only the *visible* patches (Efficiency!).
*   **Decoder:** Takes latent visible tokens + learnable mask tokens to reconstruct image.
*   **Result:** The model learns high-level semantics (shapes, textures) to "fill in the blanks". It forces the model to understand objects, not just low-level statistics.
*   *Relevance:* MAE pre-training is standard for modern Robot Foundation Models (RT-1, RT-2).

### 🔹 Part 2: Robotics-Specific SSL

Visual SSL learns "Texture" and "Shape". But Robots care about "Physics" and "Time".

#### 2.1 Time-Contrastive Networks (TCN)
**Idea:** Frames close in time (t, t+1) are semantic neighbors. Frames far apart (t, t+100) are negatives.
*   *Application:* A robot watching a human pour water learns the *sequence* of pouring regardless of the human's shirt color or background. We want the embedding space to organize by *action progress*.

#### 2.2 Visual-Tactile Cross-Modal SSL
*   **Idea:** If a robot touches a "rough" surface, the camera should see a "rough" texture.
*   **Method:** Contrastive learning between Tactile Sensor embeddings and Camera embeddings.
*   **Result:** Robot can "imagine" the feel of an object just by looking at it.

---

## 💻 Implementation: SimCLR from Scratch

We will implement the core logic of SimCLR.

### 🛠️ Project Structure
```text
day6_ssl/
├── data/
│   └── unlabeled_robot_logs/
├── models/
│   ├── simclr.py
│   └── resnet_encoder.py
├── train_simclr.py
└── visualize_embeddings.py
```

### 👨‍💻 Code Implementation

#### 1. SimCLR Loss Module (`models/simclr.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j):
        """
        z_i, z_j: Features from two augmented views of the batch.
        Shape: [Batch_Size, Dim]
        """
        batch_size = z_i.shape[0]
        
        # Concatenate: [2N, Dim]
        features = torch.cat([z_i, z_j], dim=0)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Cosine Similarity Matrix: [2N, 2N]
        similarity_matrix = torch.matmul(features, features.T)
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(features.device)
        # We can simulate labels: i and i+batch_size are positives
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)
        
        # Remove self-contrast cases
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        labels = labels[~mask].view(2 * batch_size, -1)
        
        # Positives
        positives = similarity_matrix[labels.bool()].view(2 * batch_size, -1)
        
        # Negatives
        negatives = similarity_matrix[~labels.bool()].view(2 * batch_size, -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / self.temperature
        
        # Cross Entropy: Ideally positives should be class 0 (index 0 in logits)
        labels = torch.zeros(2 * batch_size, dtype=torch.long).to(features.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss
```

#### 2. The SimCLR Model Wrapper

```python
import torchvision.models as models

class SimCLR(nn.Module):
    def __init__(self, base_model='resnet18', projection_dim=128):
        super().__init__()
        
        # Encoder
        self.encoder = models.resnet18(pretrained=False)
        dim_mlp = self.encoder.fc.in_features
        
        # Remove original FC
        self.encoder.fc = nn.Identity()
        
        # Projection Head (MLP)
        # Paper says: (Linear -> ReLU -> Linear)
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z
```

#### 3. Data Augmentation Pipeline
Crucial part of SimCLR.

```python
import torchvision.transforms as T

class SimCLRTransform:
    def __init__(self, size=96):
        color_jitter = T.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.transform = T.Compose([
            T.RandomResizedCrop(size=size),
            T.RandomHorizontalFlip(),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=int(0.1 * size)),
            T.ToTensor(),
            # Normalization (ImageNet stats usually)
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __call__(self, x):
        return self.transform(x), self.transform(x) # Return two views
```

---

## 🔬 Lab Exercise: Pre-training on Robot Logs

### 1. Lab Objectives
- Collect 1000 images from a simulated TurtleBot navigating a room.
- Pre-train a ResNet-18 using SimCLR for 50 epochs.
- Frozen Evaluation: Freeze the ResNet, add a Linear Classifier, and train with only 10 labeled images. Compare accuracy vs Fully Supervised.

### 2. Step-by-Step Guide

#### Phase A: Training Loop (SimCLR)

```python
# train_simclr.py
# ... Setup Dataset with SimCLRTransform ...
# ... Setup Model & Optimizer ...
loss_fn = NTXentLoss()

for epoch in range(50):
    for (x_i, x_j), _ in dataloader:
        optimizer.zero_grad()
        
        _, z_i = model(x_i.cuda())
        _, z_j = model(x_j.cuda())
        
        loss = loss_fn(z_i, z_j)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save Encoder
torch.save(model.encoder.state_dict(), "encoder_weights.pth")
```

#### Phase B: Linear Evaluation
Evaluate the quality of representations.

```python
# train_linear.py
# Load Encoder
encoder = models.resnet18()
encoder.fc = nn.Identity()
encoder.load_state_dict(torch.load("encoder_weights.pth"))

# Freeze Weights
for param in encoder.parameters():
    param.requires_grad = False

# Add Classifier
classifier = nn.Linear(512, NUM_CLASSES)
full_model = nn.Sequential(encoder, classifier).cuda()

# Train ONLY classifier on very small labeled dataset
# ...
```

### 3. Expected Results
- Fully Supervised (10 samples): ~20% Accuracy (Overfitting).
- SimCLR Pre-trained + Linear (10 samples): ~65% Accuracy.
- **Conclusion:** SSL learns robust features that generalize well with few labels.

---

## 🚀 Project: Anomaly Detection for Robot Patrol

**Goal:** Robot patrols a corridor. If it sees something "new" (Anomaly), it alerts.
**Method:** Train a lightweight Autoencoder or Use SimCLR features + One-Class SVM.

### 1. Logic
1.  **Collect Data:** Patrol normal corridors (Positive class).
2.  **Train:** Train SimCLR on this data to learn "What a corridor looks like".
3.  **Deploy:**
    *   Get feature embedding $z$.
    *   Compute Cosine Similarity to the cluster center of training data.
    *   If similarity < threshold, Alert "Anomaly!" (e.g., a box blocking the path).

### 2. Implementation Snippet

```python
def get_features(loader, model):
    features = []
    with torch.no_grad():
        for x, _ in loader:
            h, _ = model(x.cuda())
            features.append(h.cpu())
    return torch.cat(features)

# Train Phase
train_feats = get_features(train_loader, model)
center = torch.mean(train_feats, dim=0)
center = F.normalize(center, dim=0)

# Inference Phase
def detect_anomaly(image):
    feat, _ = model(image) # Normalized inside model if needed
    feat = F.normalize(feat, dim=1)
    
    similarity = torch.matmul(feat, center)
    
    if similarity < 0.8:
        return "Anomaly Detected"
    return "Normal"
```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Collapsed Mode" (SimCLR)
*   **Symptom:** Loss drops to a constant, embeddings are all identical.
*   **Cause:** Batch size too small or missing projection head.
*   **Fix:** SimCLR needs Large Batches (BS > 256). If GPU is small, use Gradient Accumulation or **MoCo** (which uses a memory queue).

#### 2. Features not helping
*   **Cause:** Augmentations were not strong enough. If the model can cheat (e.g., by matching color histograms), it won't learn shape.
*   **Fix:** Ensure Random Grayscale and Color Jitter are aggressive.

---

## ⚡ Optimization: Masked Autoencoders (MAE)

SimCLR is computationally heavy due to augmentations and negative pairs.
**MAE** is faster to train per epoch because it only processes 25% of the image.

**Task:** Use `timm` or `HuggingFace Transformers` to load a pre-trained MAE (ViT-MAE) and visualize the reconstructions of robot camera frames. Note how it reconstructs plausible backgrounds behind distinct objects.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why do we need "Negative Pairs" in Contrastive Learning?
    *   **A:** To prevent "Collapse". Without negatives, the model could just map everything to a constant zero vector and achieve perfect similarity for positive pairs.
2.  **Q:** What is a "Pretext Task"?
    *   **A:** A self-generated supervision task (e.g., "Predict the rotation of this image", "Reconstruct this masked patch") used to force the model to learn structure without human labels.
3.  **Q:** How does SSL help "Long-Tail" distributions (rare events)?
    *   **A:** Pre-training on *all* data allows the model to see rare events (unlabeled) and learn their features. When finetuning, even 1-2 labeled examples of the rare event are enough to bind the class label to the features.

### Challenge Task
> **Task:** Implement Time-Contrastive Learning.
> 1. Take a video of robot arm movement.
> 2. Anchor: Frame $t$. Positive: Frame $t+5$. Negative: Frame $t+100$.
> 3. Train embedding.
> 4. Visualize the embeddings with t-SNE. Do they form a smooth trajectory?

---

## 📚 Further Reading
- **SimCLR:** Chen et al. (ICML 2020).
- **MoCo:** He et al. (CVPR 2020).
- **MAE:** He et al. (CVPR 2022).
- **TCN (Time-Contrastive):** Sermanet et al. (ICRA 2018).

---

**Day 6 Complete**
