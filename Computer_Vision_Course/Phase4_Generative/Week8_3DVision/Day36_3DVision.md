# Day 36: 3D Vision Basics

## 1. 3D Representations
How do we represent the 3D world?
1.  **Point Clouds:** Set of $(x, y, z)$ coordinates. Simple, sparse.
2.  **Voxels:** 3D grid (like pixels). $V \times V \times V$. Computationally expensive ($O(N^3)$).
3.  **Meshes:** Vertices + Faces (Triangles). Standard in graphics. Hard for Deep Learning (graph structure).
4.  **Implicit Functions:** $f(x, y, z) = \text{Occupancy}$. Continuous.

## 2. PointNet (2017)
**Problem:** Point clouds are unordered sets. A network must be **Permutation Invariant**.
**Architecture:**
1.  **Input:** $N \times 3$ (N points).
2.  **MLP (Shared):** Apply same MLP to every point independently. $(x, y, z) \to 1024$-dim vector.
3.  **Symmetric Function:** Max Pooling over all $N$ points.
    *   Extracts a global feature vector that describes the shape.
4.  **T-Net (Transform Net):** Predicts a $3 \times 3$ rotation matrix to align the input canonical space.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

    def forward(self, x):
        # x: (Batch, 3, N)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Max Pooling (Symmetric Function)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

## 3. Loss Functions
*   **Chamfer Distance:** Measures similarity between two point clouds $S_1, S_2$.
    $$ d_{CD}(S_1, S_2) = \sum_{x \in S_1} \min_{y \in S_2} ||x-y||^2 + \sum_{y \in S_2} \min_{x \in S_1} ||x-y||^2 $$
    *   "For every point in A, find nearest in B. And vice versa."

## Summary
PointNet proved that we can apply Deep Learning directly to raw point clouds by using symmetric functions (Max Pool) to handle permutation invariance.
