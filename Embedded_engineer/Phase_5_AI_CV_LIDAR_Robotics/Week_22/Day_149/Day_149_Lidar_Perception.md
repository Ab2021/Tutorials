# Day 149: Lidar Perception (PointPillars)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 22: Autonomous Driving Stack

---

> **📝 Content Creator Instructions:**
> Voxels are slow. Pillars are fast.
> - **Focus:** 3D Object Detection from Lidar, Voxelization vs PointPillars, Pseudo-Image generation, and 2D Switch (Using 2D CNNs on BEV maps).
> - **Code:** A Python script `pillar_encoder.py` that manually voxelizes a Point Cloud into "Pillars" (Vertical columns), computes feature embeddings (Max Pooling), and generates a "Pseudo-Image" ready for a CNN.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** why standard 3D Convolutions are too sparse and slow for autonomous driving.
2.  **Implement** the PointPillars encoding step: Points $\to$ Pillars $\to$ 2D Map.
3.  **Visualize** the resulting "Pseudo-Image" that represents the Lidar scene.
4.  **Understand** the output format: 7-DOF Boxes ($x, y, z, w, l, h, \theta$).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Convolutional Neural Networks (CNNs).
- Point Clouds.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Sparsity Problem

A Lidar scan has 100k points in a $100m \times 100m \times 10m$ volume.
If you make a Voxel Grid of $0.1m$, you get $1000 \times 1000 \times 100 = 100$ Million Voxels.
Only 0.1% are occupied. 3D Conv is a waste of compute.

### 🔹 Part 2: The PointPillars Solution

1.  **Discretize into Infinite Columns (Pillars):** Don't chop Z-axis. Grid is $XY$ only ($400 \times 400$).
2.  **PointNet per Pillar:** For each pillar, take all points inside. Calculate features (e.g., center offset, distance). Run a tiny MLP. Max Pool.
3.  **Pseudo-Image:** The result is a $C \times H \times W$ tensor.
4.  **Backbone:** Run a fast 2D SSD/YOLO on this "Image".

### 🔹 Part 3: Output Encoding

The Network predicts 7 numbers per anchor:
*   **Center:** $\Delta x, \Delta y, \Delta z$
*   **Size:** $\Delta w, \Delta l, \Delta h$ (Log scale)
*   **Heading:** $\sin \theta, \cos \theta$

---

## 💻 Implementation: The Pillar Feature Net

We implement the first half of PointPillars (Encoding). A full training loop requires a GPU and significant infrastructure (OpenPCDet).

### 🛠️ Project Structure
```text
day149_pointpillars/
├── src/
│   ├── pillar_encoder.py
└── output/
    ├── bev_pseudo_image.png
```

### 👨‍💻 Pillar Encoder (`src/pillar_encoder.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class PointPillarsEncoder:
    def __init__(self):
        # Config
        self.x_range = (0, 70.4)
        self.y_range = (-40, 40)
        self.voxel_size = 0.4 # meters
        
        self.grid_w = int((self.x_range[1] - self.x_range[0]) / self.voxel_size)
        self.grid_h = int((self.y_range[1] - self.y_range[0]) / self.voxel_size)
        
        self.max_points_per_pillar = 32
        self.num_features = 4 # x, y, z, intensity

    def encode(self, points):
        # points: [N, 4]
        
        # 1. Assign Points to Grid Indices
        # Shift to 0,0
        x_shifted = points[:, 0] - self.x_range[0]
        y_shifted = points[:, 1] - self.y_range[0]
        
        x_idx = (x_shifted / self.voxel_size).astype(int)
        y_idx = (y_shifted / self.voxel_size).astype(int)
        
        # Valid Mask (Inside Grid)
        mask = (x_idx >= 0) & (x_idx < self.grid_w) & (y_idx >= 0) & (y_idx < self.grid_h)
        points = points[mask]
        x_idx = x_idx[mask]
        y_idx = y_idx[mask]
        
        # 2. Group Points by Pillar
        # Using Dictionary for sparsity
        pillars = {}
        for i in range(len(points)):
            key = (x_idx[i], y_idx[i])
            if key not in pillars:
                pillars[key] = []
            if len(pillars[key]) < self.max_points_per_pillar:
                pillars[key].append(points[i])
                
        # 3. Feature Extraction (Simplified PointNet)
        # Create Pseudo-Image: [Features, H, W]
        # Let's say Feature = Mean Z height + Density
        
        pseudo_image = np.zeros((self.grid_h, self.grid_w))
        
        for key in pillars:
            pts = np.array(pillars[key])
            
            # Simple Feature: Max Height (Z)
            # In real PointPillars, this is an MLP 
            max_height = np.max(pts[:, 2])
            
            # Populate Map
            # Note: y_idx maps to Row (H), x_idx maps to Col (W)
            pseudo_image[key[1], key[0]] = max_height
            
        return pseudo_image

def main():
    encoder = PointPillarsEncoder()
    print(f"Grid Size: {encoder.grid_w} x {encoder.grid_h}")
    
    # Generate Synthetic Lidar Data (Road + Obstacles)
    points = []
    
    # Ground Plane (Z=0)
    for i in range(10000):
        x = np.random.uniform(0, 70)
        y = np.random.uniform(-40, 40)
        z = np.random.normal(0, 0.05)
        points.append([x, y, z, 1.0])
        
    # Car 1 (Box at x=20, y=0)
    for i in range(500):
        x = np.random.uniform(18, 22)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(0, 1.5)
        points.append([x, y, z, 1.0])
        
    # Wall (at x=50)
    for i in range(1000):
        x = np.random.uniform(49.5, 50.5)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(0, 3.0)
        points.append([x, y, z, 1.0])
        
    points = np.array(points)
    
    # Encode
    bev_map = encoder.encode(points)
    
    # Visualize
    plt.figure(figsize=(10, 10))
    # Flip Y to match coordinate system orientation often used
    plt.imshow(bev_map, origin='lower', cmap='jet')
    plt.colorbar(label='Max Height (m)')
    plt.title("PointPillars Pseudo-Image (BEV)")
    plt.xlabel("X Index (Longitudinal)")
    plt.ylabel("Y Index (Lateral)")
    plt.savefig("output/bev_pseudo_image.png")
    print("Encoded Map Saved.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Data Augmentation"

### 1. Lab Objectives
- **Run:** The encoder.
- **Observe:** The Car shows up as a "Hot Spot" (Red) because it is tall. Ground is Blue (Low).
- **Modify:** What if the Lidar is sparse? Reduce point count by 90%.
- **Observe:** The Pseudo-Image becomes "Swiss Cheese".
- **Concept:** This is why we need Dense feature extraction or multiple frames (Lidar accumulation) to fill gaps.

---

## 🚀 Project: "3D NMS (Non-Maximum Suppression)"

**Goal:** Clean up overlapping boxes.
1.  **Problem:** The network outputs 10 boxes for 1 car.
2.  **IoU (Intersection over Union):** Calculate 3D Volume Intersection?
    *   *Shortcut:* Use BEV 2D IoU. If boxes overlap in 2D and height is similar, they are the same object.
3.  **Task:** Implement a function `iou_2d_rotated(box1, box2)` using Polygon clipping.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Empty Pillars"
*   **Cause:** Voxel size too small (e.g., 0.05m). Most pillars have 0 points.
*   **Result:** Wasted memory and compute (if not using sparse implementation).
*   **Fix:** Tune Voxel Size (0.16m - 0.25m is standard for Cars).

#### 2. "Z-Fighting"
*   **Cause:** Ground points are $Z \approx 0$.
*   **Result:** Sometimes encoded as 0, sometimes noise masked out.
*   **Fix:** Explicitly remove ground points using RANSAC before pillarization if possible, though PointPillars can learn to ignore ground.

---

## ⚡ Optimization: Scatter Gather

Why `max_points_per_pillar = 32`?
To make the tensor static size $[P, 32, C]$.
*   **Padding:** If pillar has 5 points, pad with zeros.
*   **Sampling:** If pillar has 100 points, random sample 32.
*   This allows Batch Processing on GPU.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is BEV?
    *   **A:** Bird's Eye View. Top-down projection. No perspective distortion. Distance is linear (unlike Camera view).
2.  **Q:** Why not just project points to an image (Range View)?
    *   **A:** Range View (Spherical projection) distorts shapes. A car nearby looks huge. A car far away looks small. In BEV, a car is always $4.5m \times 1.8m$.
3.  **Q:** What is the "Anchor Box"?
    *   **A:** A pre-defined box size (e.g., Width=1.6m, Length=3.9m) placed at every grid cell. The network predicts the *offset* from this anchor.

### Challenge Task
> **Task:** Intensity Feature.
> 1. Use `points[:, 3]` (Intensity/Reflectivity).
> 2. Add `Mean Intensity` to the pseudo-image features.
> 3. Does this help detect Lane Markings (High reflectivity)?

---

## 📚 Further Reading
- **Lang et al.:** "PointPillars: Fast Encoders for Object Detection from Point Clouds".
- **OpenPCDet:** The state-of-the-art codebase for Lidar detection.

---

**Day 149 Complete**
