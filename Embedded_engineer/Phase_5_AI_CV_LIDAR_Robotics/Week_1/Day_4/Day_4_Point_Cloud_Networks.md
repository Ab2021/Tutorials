# Day 4: Point Cloud Neural Networks
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 1: Deep Learning Foundations

---

> **📝 Content Creator Instructions:**
> LiDAR is the "Source of Truth" for autonomous driving geometry. This document dives into Deep Learning on unordered steps of 3D points.
> - **Focus:** PointNet, PointNet++, and Sparse Convolution (Minkowski Engine / spconv).
> - **Code:** Implementation of PointNet++ from scratch.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** why standard CNNs cannot directly process 3D Point Clouds.
2.  **Derive** the PointNet architecture and its properties (Permutation Invariance).
3.  **Implement** the Set Abstraction Layer of PointNet++ to capture local structures.
4.  **Differentiate** between Voxel-based (Sparse Conv) and Point-based methods.
5.  **Build** a ROS 2 node that classifies objects (Desks, Chairs, People) from a live LiDAR stream.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- LiDAR Sensor (Simulated in Gazebo or Real Velodyne/Ouster).
- GPU (Required for training).

### Software Environment
```bash
pip install torch torchvision
pip install open3d         # Visualization
pip install plyfile
# Optional for advanced sparse convs:
# pip install spconv-cu118
```

### Prior Knowledge
- Multi-Layer Perceptrons (MLP).
- Basic 3D Geometry ($x, y, z$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Nature of 3D Data

Images are dense grids of pixels. LiDAR data is **Sparse** and **Unordered**.
*   **Sparsity:** In a typical 3D scene, 99% of the voxels are empty air. Storing this in a dense 3D matrix is wasteful.
*   **Permutation Invariance:** A set of points $\{P_1, P_2, P_3\}$ represents the same shape as $\{P_3, P_1, P_2\}$. A standard MLP or CNN is detecting patterns in specific *orders*, making them unsuitable.

#### 1.1 PointNet (CVPR 2017)
The pioneer of raw point cloud processing.
**Key Idea:** Use a symmetric function (Max Pooling) to aggregate features from all points, making the output invariant to input order.

**Architecture Steps:**
1.  **Input:** $N \times 3$ (N points, xyz).
2.  **Shared MLP:** Apply an MLP $(64, 128, 1024)$ to *each point independently*. This lifts each point to a high-dimensional feature space.
3.  **T-Net (Transform Net):** predicts a $3 \times 3$ affine transformation matrix to align the input point cloud canonically (handling rotation).
4.  **Max Pooling:** Take the maximum value across all $N$ points for each feature channel. Result: Global Feature Vector.
5.  **Classification Head:** MLP on the global vector.

*Limitation:* PointNet learns a global representation but loses local context (e.g., the relationship between a wheel and a car chassis).

#### 1.2 PointNet++ (NeurIPS 2017)
Introduces hierarchical feature learning, similar to CNNs.
**Set Abstraction Layer:**
1.  **Sampling (Farthest Point Sampling - FPS):** Select $K$ centroids that are far apart to cover the point cloud.
2.  **Grouping (Ball Query):** Find all points within radius $R$ of each centroid.
3.  **PointNet:** Apply a mini-PointNet to each group to extract local features.

#### 1.3 Sparse Convolutions (Voxel-based)
Instead of processing points, we "voxelize" space (grid).
Since most voxels are empty, we use **Sparse Convolution**:
$$ O_u = \sum_{i \in \text{kernel}} W_i \cdot I_{u+i} $$
Only compute at active voxel locations.
*   *Advantage:* Extremely fast and efficient. state-of-the-art for 3D Detection (SECOND, CenterPoint).
*   *Libraries:* `spconv`, `MinkowskiEngine`.

### 🔹 Part 2: Geometric Deep Learning

Geometry is not just coordinates.
*   **Normals:** Orientation of the surface.
*   **Curvature:** How "sharp" a corner is.
*   **Density:** LiDAR points are dense near the sensor and sparse far away.

**Handling Density Shifts:**
PointNet++ introduces **Multi-Scale Grouping (MSG)**. It captures features at multiple radii ($r_1, r_2$) to be robust to varying point densities.

---

## 💻 Implementation: PointNet++ from Scratch

We will implement the Set Abstraction (SA) Layer, the core engine.

### 🛠️ Project Structure
```text
day4_pointnet/
├── layers/
│   ├── sampling.py
│   ├── grouping.py
│   └── pointnet_modules.py
├── train_sem_seg.py
└── visualize_parts.py
```

### 👨‍💻 Code Implementation

#### 1. Farthest Point Sampling (`layers/sampling.py`)

```python
import torch

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    
    # Random start point
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        
    return centroids
```

#### 2. Ball Query Grouping (`layers/grouping.py`)

```python
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points (centroids), [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    
    sqrdists = square_distance(new_xyz, xyz) # Helper function for dist matrix
    group_idx[sqrdists > radius ** 2] = N # Set invalid points to N
    
    # Sort to pick closest points (optional, or just first K)
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    # Handle case where fewer points than nsample exist (duplicate first point)
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([B, S, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx
```

#### 3. Set Abstraction Layer (`layers/pointnet_modules.py`)

```python
import torch.nn as nn
import torch.nn.functional as F

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
            
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points [B, C, N]
            points: input features [B, D, N]
        Return:
            new_xyz: sampled points [B, C, S]
            new_points: aggregated features [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1) # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1) # [B, N, D]

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            # 1. Sample
            new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
            
            # 2. Group
            # Returns [B, npoint, nsample, C+D]
            new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points) 
            
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample, npoint]
        
        # 3. PointNet (MLP + MaxPool)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
            
        # Max Pooling over nsample
        new_points = torch.max(new_points, 2)[0] # [B, OutC, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)
        
        return new_xyz, new_points
```

---

## 🔬 Lab Exercise: LiDAR Scene Classification

### 1. Lab Objectives
- Parse a raw `.pcd` file using Open3D.
- Normalize the Point Cloud (center at origin, scale to unit sphere).
- Train PointNet++ to classify shapes (ModelNet40 dataset subset).

### 2. Step-by-Step Guide

#### Phase A: Data Loading

```python
import open3d as o3d
import numpy as np

def normalize_pc(points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    farthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points /= farthest_distance
    return points

def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    
    # Downsample if too dense
    if len(points) > 2048:
        idx = np.random.choice(len(points), 2048, replace=False)
        points = points[idx]
        
    return normalize_pc(points)

# Visualization
points = load_pcd("data/chair.pcd")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])
```

#### Phase B: Training Loop
(Standard PyTorch training loop using the `PointNetSetAbstraction` defined above).

### 3. Expected Output
- **Input:** A chaotic 3D scatter plot.
- **Output:** Predicted Class: "Chair" (Confidence 92%).

---

## 🚀 Project: ROS 2 Point Cloud Processor

**Goal:** Receive `/velodyne_points`, process chunks, publish bounding boxes.

### 1. ROS 2 Node (`lidar_processor.py`)

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import torch
import numpy as np

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        self.sub = self.create_subscription(
            PointCloud2, '/velodyne_points', self.pc_callback, 10
        )
        self.model = PointNetPP().cuda().eval()
        
    def pc_callback(self, msg):
        # 1. Convert ROS msg to Numpy
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(gen))
        
        # 2. Preprocess (Voxel Grid Filter or Random Sample)
        if points.shape[0] < 100: return
        
        # 3. Simple ground removal (z < -1.5m)
        non_ground = points[points[:, 2] > -1.4]
        
        # 4. Clustering (Euclidean) - in a real app, use Scikit-Learn DBSCAN
        # For this demo, we assume the non-ground object is centered
        
        # 5. Inference
        tensor_in = torch.from_numpy(non_ground).float().unsqueeze(0).transpose(2, 1).cuda()
        with torch.no_grad():
            cls_logits = self.model(tensor_in)
            
        cls = cls_logits.argmax().item()
        self.get_logger().info(f"Detected Object Class: {cls}")

```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Input contains NaN"
*   **Cause:** LiDAR returns NaN for max-range hits or errors.
*   **Fix:** Always check for NaNs during loading (`skip_nans=True` in ROS, or `np.nan_to_num`).

#### 2. Model Divergence
*   **Cause:** Point clouds are unnormalized. Inputs like $(100.5, 52.1, 3.0)$ explode gradients compared to $(0.1, 0.5, 0.03)$.
*   **Fix:** **Always** normalize input point clouds to unit sphere or unit cube before feeding to neural net.

#### 3. Slow FPS
*   **Cause:** Python training loop for raw points is slow.
*   **Fix:** Use `spconv` (Sparse Convolution) if N > 10,000. PointNet is good for objects (N < 2048), but slow for full scenes (N=100k).

---

## ⚡ Optimization: Sparse Convolutions

For students seeking maximum performance:
**MinkowskiEngine** represents the scene as a Hash Map of active voxels.
$$ \text{Features} = \text{HashMap}[(x,y,z) \to \text{features}] $$
Convolution is a hash lookup.
*   **Task:** Install `MinkowskiEngine` and run their semantic segmentation demo.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is "Farthest Point Sampling" used instead of Random Sampling?
    *   **A:** FPS ensures better coverage of the geometry, particularly preserving extremities and shapes, whereas random sampling might cluster in dense areas.
2.  **Q:** What is the difference between PointNet and PointNet++?
    *   **A:** PointNet is global (one feature vector for whole cloud). PointNet++ uses local grouping to extract features at different scales (like a CNN).

### Challenge Task
> **Task:** Implement a "T-Net" (Transform Net) module.
> 1. Input: $N \times 3$.
> 2. MLP -> MaxPool -> MLP -> Output $3 \times 3$ matrix.
> 3. Add orthogonal regularization loss $||I - AA^T||^2$.
> 4. Multiply input points by this matrix before feeding to PointNet.

---

## 📚 Further Reading
- **PointNet:** Qi et al. (CVPR 2017).
- **PointNet++:** Qi et al. (NeurIPS 2017).
- **VoxelNet:** Zhou et al. (CVPR 2018).
- **MinkowskiEngine:** Choy et al. (CVPR 2019).

---

**Day 4 Complete**
