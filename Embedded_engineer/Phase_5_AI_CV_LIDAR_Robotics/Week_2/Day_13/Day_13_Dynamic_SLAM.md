# Day 13: Dynamic Object Handling in SLAM
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 2: Advanced SLAM & State Estimation

---

> **📝 Content Creator Instructions:**
> SLAM assumes a **Static World**. But in reality, cars move, people walk, and doors open.
> - **Focus:** Detecting dynamics, Semantic Masking (using results from Week 1), and Robust SLAM backends.
> - **Code:** Implementation of a Dynamic Point Filter using Residual Analysis.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the "Dynamic Object Problem": how moving objects pollute the map and break optimization.
2.  **Implement** a Residual-based Moving Object Detector (MOD).
3.  **Integrate** Semantic Segmentation (Mask R-CNN) to mask out dynamic classes (Person, Car) *before* SLAM.
4.  **Evaluate** SLAM robustness in highly dynamic environments (e.g., crowded cafeteria).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- LiDAR or RGB-D Camera.

### Software Environment
```bash
pip install numpy open3d torch torchvision
```

### Prior Knowledge
- Detectors (Week 1).
- ICP/Scan Matching.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Static World Assumption

Standard SLAM (e.g., GMapping, Cartographer, ORB-SLAM) assumes:
$$ P(z_t | x_t, m) $$
The map $m$ is constant.
*   **The Problem:** If a person walks in front of the robot, the LIDAR sees the person. If SLAM adds the person to the map, the robot will later try to path-plan around a "ghost" that is no longer there.
*   **Worse Problem:** If the robot is stopped and a bus moves past it, the robot might think *it* is moving backwards (Relative Motion ambiguity).

### 🔹 Part 2: Approaches to Dynamic SLAM

#### 2.1 Geometry-Based (Residuals)
*   **Idea:** If I align Scan $t$ to Scan $t-1$, static points will overlap (Low Residual). Dynamic points will have moved (High Residual).
*   **Method:**
    1.  Perform ICP.
    2.  Calculate distance between matched points.
    3.  If $dist > Threshold$, likely dynamic.
    4.  Remove these points and Re-run ICP (Iterative).

#### 2.2 Semantic-Based (Deep Learning)
*   **Idea:** We know people move. Walls (usually) don't.
*   **Method:**
    1.  Run Object Detector / Semantic Segmentor (YOLO / Mask R-CNN).
    2.  Get Masks for "Person", "Car", "Bus", "Cat".
    3.  **Blacklist:** Ignore any feature point that falls inside these masks during Tracking and Mapping.
    *   *System:* **DS-SLAM** (Dynamic Semantic SLAM).

#### 2.3 Life-Long Mapping
*   **Idea:** Maps decay.
*   **Method:** Occupancy Grid with decay. If a voxel is not observed for $T$ seconds, probability returns to 0.5 (Unknown).

---

## 💻 Implementation: Dynamic Filter

We will implement the Geometry-Based approach (Residual Filter).

### 🛠️ Project Structure
```text
day13_dynamic/
├── data/
│   └── dynamic_corridor.bag
├── src/
│   ├── icp_robust.py
│   └── dynamic_remover.py
└── run_filter.py
```

### 👨‍💻 Code Implementation (`src/dynamic_remover.py`)

```python
import numpy as np
import open3d as o3d
import copy

class DynamicRemover:
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        
    def align_and_filter(self, source, target):
        """
        source: Current Scan
        target: Previous Scan (Clean)
        Returns: 
            Transformation (4x4)
            Clean Match Percentage
        """
        # 1. Initial ICP (Noisy, includes dynamic points)
        reg = o3d.pipelines.registration.registration_icp(
            source, target, 0.5, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        t_init = reg.transformation
        
        # 2. Transform Source
        source_trans = copy.deepcopy(source)
        source_trans.transform(t_init)
        
        # 3. Compute Residuals
        dists = source_trans.compute_point_cloud_distance(target)
        dists = np.asarray(dists)
        
        # 4. Filter
        # Points with distance < threshold are STATIC
        # Points with distance > threshold are DYNAMIC (or new view)
        static_indices = np.where(dists < self.threshold)[0]
        dynamic_indices = np.where(dists >= self.threshold)[0]
        
        static_cloud = source.select_by_index(static_indices)
        
        percentage = len(static_indices) / len(source.points)
        print(f"Static Points: {percentage*100:.1f}%")
        
        # 5. Refine ICP using ONLY static points
        reg_refined = o3d.pipelines.registration.registration_icp(
            static_cloud, target, 0.05, t_init, # tighter threshold
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        return reg_refined.transformation, static_cloud, dynamic_indices
```

---

## 🔬 Lab Exercise: "Ghost Busting"

### 1. Lab Objectives
- Record a rosbag where you walk in front of a stationary robot.
- Run Standard GMapping/SLAM -> Observe the messy map (Ghost trails).
- Run the `DynamicRemover` script -> Visualize the removed points.

### 2. Step-by-Step Guide

#### Phase A: Visualization

```python
import matplotlib.pyplot as plt

def visualize_dynamic(source, dynamic_indices):
    # Paint dynamic points RED
    colors = np.asarray(source.colors)
    if len(colors) == 0:
        colors = np.zeros((len(source.points), 3))
    
    # Set static to Gray
    colors[:] = [0.7, 0.7, 0.7]
    
    # Set dynamic to Red
    colors[dynamic_indices] = [1.0, 0.0, 0.0]
    
    source.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([source])
```

#### Phase B: Semantic Masking (Optional with YOLO)
If you have a GPU:
1.  Run YOLOv8 on the camera image.
2.  Project the Bounding Boxes 2D -> 3D Frustums.
3.  Remove any LiDAR point inside the frustum of class "Person".

### 3. Expected Output
- **Geometry Filter:** Removes the person walking, but might also remove the newly seen wall revealed behind the person.
- **Semantic Filter:** Precisely removes the person, preserves the wall (if visible).

---

## 🚀 Project: "Robust Crowd Navigation"

**Goal:** Build a Mapping Node that never adds "Person" points to the Global Map.
**Architecture:**
1.  **Input:** `/scan` (LiDAR), `/image_raw` (Camera).
2.  **Node 1 (Detector):** Publishes `/dynamic_masks` (2D).
3.  **Node 2 (Filter):**
    *   Subscribes to Scan and Masks.
    *   Projects Scan to Image.
    *   If `pixel(scan_point) \in mask`: Delete scan point.
    *   Publishes `/scan_static`.
4.  **Node 3 (SLAM):** Runs GMapping on `/scan_static`.

**Result:** A clean, static map of the room walls, even if the room was full of people during mapping.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Removing Stationary Objects"
*   **Symptom:** A parked car is removed by the Semantic Filter.
*   **Cause:** YOLO detects "Car". Filter removes it. But a parked car *is* static obstacle and should be in the map!
*   **Fix:** Combine Semantic + Geometric. Only remove if (Class is Dynamic) AND (Residual is High). Or check velocity estimates.

#### 2. "Over-Filtering"
*   **Symptom:** Robot turns a corner, everything is "new", so everything is high residual. The filter removes the whole room.
*   **Fix:** Check odometry. If the whole scene matches poorly, it's likely a large motion, not a dynamic object. Trust the majority (RANSAC).

---

## ⚡ Optimization: Octree Raycasting

To verify if space is empty or dynamic:
*   Project a ray from Sensor to Wall.
*   If the ray passes through a voxel that was previously "Occupied", that voxel is now "dynamic" (it moved away). Update it to "Free".
*   This cleans up "Ghost trails" efficiently.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is "Person" a dynamic class but "Chair" is arguably static?
    *   **A:** Prior probability. People move constantly. Chairs move rarely (only when pushed). We usually treat furniture as static for short-term mapping, dynamic for long-term.
2.  **Q:** Effect of Dynamic Objects on Loop Closure?
    *   **A:** Catastrophic. If the Loop Closure matches a parked truck that wasn't there before, it generates a false constraint. Geometric Verification (RANSAC) is essential to reject these outliers.

### Challenge Task
> **Task:** Implement "Background Subtraction" for LiDAR.
> 1. Assume the robot is static.
> 2. Accumulate points in an Octree.
> 3. Any point that appears and disappears is dynamic.

---

## 📚 Further Reading
- **DS-SLAM:** Yu et al. (2018).
- **DynaSLAM:** Bescos et al. (2018).
- **People Aware Navigation:** ROS Navigation Stack.

---

**Day 13 Complete**
