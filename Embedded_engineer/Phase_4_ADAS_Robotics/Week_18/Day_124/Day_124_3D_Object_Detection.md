# Day 124: 3D Object Detection (PointPillars)
## Phase 4: ADAS & Robotics Systems | Week 18: Deep Learning for Perception

---

> **📝 Day 124 Focus:**
> 2D detection (YOLO) gives you a box on an image. It doesn't tell you how far away the car is. **3D Detection** uses Lidar point clouds to draw a 3D box $(x, y, z, l, w, h, \theta)$. The challenge: Point clouds are sparse and unstructured.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the challenges of processing Point Clouds (Unordered, Sparse).
2.  **Compare** Voxel-based (VoxelNet) vs. Point-based (PointNet) methods.
3.  **Implement** the **PointPillars** encoding (converting PC to Pseudo-Image).
4.  **Visualize** 3D Bounding Boxes on Kitti dataset.
5.  **Understand** the Anchor-based 3D detection head.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 121:** CNNs.
-   **Lidar Data:** $(x, y, z, intensity)$.

### Hardware Requirements
-   **GPU:** Mandatory.

### Software Stack
-   **Python:** `open3d`, `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Representation

How do we feed a Point Cloud to a CNN?
1.  **PointNet:** Process raw points directly. Good for small objects, slow for scenes.
2.  **VoxelNet:** Voxelize space ($0.1m^3$ cubes). Use 3D Conv. Very memory intensive.
3.  **PointPillars:**
    -   Divide ground plane $(x, y)$ into a grid.
    -   Treat columns (z-axis) as "Pillars".
    -   Learn features for each pillar using a simplified PointNet.
    -   Result: A 2D "Pseudo-Image" ($H \times W \times C$).
    -   Process with standard 2D CNN (Fast!).

### 🔹 Part 2: The Architecture (PointPillars)

1.  **Pillar Feature Net (PFN):**
    -   Input: Points in a pillar $(x, y, z, r, x_c, y_c, z_c, x_p, y_p)$.
    -   Linear -> BatchNorm -> ReLU -> MaxPool.
    -   Output: Feature vector per pillar.
2.  **Backbone (2D CNN):** Downsamples the pseudo-image.
3.  **Detection Head (SSD):** Predicts 3D box parameters and class.

### 🔹 Part 3: 3D Bounding Box

Defined by 7 numbers:
-   Center: $(x, y, z)$
-   Dimensions: $(l, w, h)$
-   Heading: $\theta$ (Yaw)

---

## 💻 Implementation: Pillar Encoder

**Scenario:**
-   Input: Random Point Cloud.
-   Task: Encode into Pillars and generate Pseudo-Image.

### 🛠️ Setup
Create `week18_day124` and `point_pillars.py`.

```bash
mkdir -p ~/ros2_ws/src/week18_day124
cd ~/ros2_ws/src/week18_day124
touch point_pillars.py
```

### 👨‍💻 Code: From Points to Image

```python
import numpy as np
import matplotlib.pyplot as plt

class PointPillarsEncoder:
    def __init__(self, x_range, y_range, z_range, pillar_size, max_points_per_pillar):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.dx, self.dy = pillar_size
        self.max_points = max_points_per_pillar
        
        self.nx = int((self.x_max - self.x_min) / self.dx)
        self.ny = int((self.y_max - self.y_min) / self.dy)
        
        print(f"Grid Size: {self.nx} x {self.ny}")

    def encode(self, points):
        # points: [N, 4] (x, y, z, intensity)
        
        # 1. Filter out of range
        mask = (points[:, 0] >= self.x_min) & (points[:, 0] < self.x_max) & \
               (points[:, 1] >= self.y_min) & (points[:, 1] < self.y_max) & \
               (points[:, 2] >= self.z_min) & (points[:, 2] < self.z_max)
        points = points[mask]
        
        # 2. Assign to Pillars
        # Calculate grid indices
        x_idx = ((points[:, 0] - self.x_min) / self.dx).astype(int)
        y_idx = ((points[:, 1] - self.y_min) / self.dy).astype(int)
        
        # 3. Create Pseudo-Image (Simplified: Count points per pillar)
        # In real PointPillars, we would run a mini-net here.
        # Here we just visualize density.
        
        bev_map = np.zeros((self.ny, self.nx))
        
        # Fast histogram
        # Flatten indices: idx = y * nx + x
        flat_indices = y_idx * self.nx + x_idx
        counts = np.bincount(flat_indices, minlength=self.nx*self.ny)
        
        bev_map = counts.reshape(self.ny, self.nx)
        
        # Log scale for visualization
        bev_map = np.log1p(bev_map)
        
        return bev_map

def generate_mock_cloud():
    # Ground plane
    x = np.random.uniform(0, 50, 10000)
    y = np.random.uniform(-20, 20, 10000)
    z = np.random.normal(0, 0.05, 10000)
    ground = np.vstack((x, y, z, np.ones_like(x))).T
    
    # Object (Car) at (20, 5)
    cx, cy, cz = 20, 5, 1
    ox = np.random.uniform(cx-2, cx+2, 500)
    oy = np.random.uniform(cy-1, cy+1, 500)
    oz = np.random.uniform(cz-1, cz+1, 500)
    obj = np.vstack((ox, oy, oz, np.ones_like(ox)*2)).T
    
    return np.vstack((ground, obj))

def main():
    # Config: 0-50m X, -20-20m Y. Pillar size 0.2m
    encoder = PointPillarsEncoder(
        x_range=(0, 50), 
        y_range=(-20, 20), 
        z_range=(-3, 3), 
        pillar_size=(0.2, 0.2), 
        max_points_per_pillar=100
    )
    
    pc = generate_mock_cloud()
    print(f"Point Cloud: {pc.shape[0]} points")
    
    bev_image = encoder.encode(pc)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(bev_image, origin='lower', cmap='jet', extent=[0, 50, -20, 20])
    plt.title("PointPillars BEV Pseudo-Image")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.colorbar(label="Log Point Density")
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Car in the Grid

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   Blue background (Ground, low density).
    -   Red rectangle at $(20, 5)$ (Car, high density).
3.  **Experiment:**
    -   Change `pillar_size` to `(0.5, 0.5)`.
    -   **Result:** The image becomes blockier (lower resolution).
    -   **Trade-off:** Larger pillars = Faster CNN, but less accurate localization.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Sparse Input
**Symptom:** BEV image is mostly empty (zeros).
**Cause:** Lidar points are sparse.
**Solution:** This is normal! CNNs can handle sparse inputs (zeros are just "no feature"). But ensure your `x_range` matches the data.

#### 2. Z-Axis Clipping
**Symptom:** Objects missing.
**Cause:** `z_range` set to `(-1, 1)` but car roof is at `1.5`.
**Solution:** Check sensor mounting height. Usually `z_range=(-3, 1)` relative to sensor (if sensor is high) or `(-1, 3)` (if sensor is low).

---

## ⚡ Optimization & Best Practices

### 1. Sparse Convolution
Standard Conv2d wastes time multiplying zeros.
-   **SparseConvNet:** Only computes at non-zero locations.
-   Used in **SECOND** (Sparsely Embedded Convolutional Detection).
-   Massive speedup for 3D data.

### 2. Data Augmentation (3D)
-   **Global Rotation:** Rotate the whole scene.
-   **Global Scaling:** Scale the scene.
-   **Ground Truth Sampling:** Cut-and-paste cars from other frames into the current frame to increase training samples.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is PointPillars faster than VoxelNet?
    *   **A:** It collapses the Z-dimension early, turning a 3D problem into a 2D problem. 2D Conv is highly optimized on GPUs.
2.  **Q:** What is the output of the detection head?
    *   **A:** For each anchor: $(\Delta x, \Delta y, \Delta z, \Delta l, \Delta w, \Delta h, \Delta \theta, ClassScore)$.
3.  **Q:** How do we handle rotation ambiguity?
    *   **A:** Cars look the same rotated $0^\circ$ and $180^\circ$. We often predict $\sin(2\theta), \cos(2\theta)$ or use a direction classifier.

### Challenge Task
**Task:** Kitti Viz.
1.  Download a Kitti sample (bin file).
2.  Use `open3d` to visualize it.
3.  Draw a 3D line set box around a car manually.

---

## 📚 Further Reading & References
-   [PointPillars Paper](https://arxiv.org/abs/1812.05784)
-   [OpenPCDet (Toolbox)](https://github.com/open-mmlab/OpenPCDet)

---

**Day 124 Complete** | Phase 4: ADAS & Robotics Systems | Week 18: Deep Learning for Perception
