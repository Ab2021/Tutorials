# Day 8: LiDAR Odometry (LOAM)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 2: Advanced SLAM & State Estimation

---

> **📝 Content Creator Instructions:**
> "Where am I?" is the most fundamental question for a robot.
> - **Focus:** Feature-based matching (Lines/Planes), Motion Distortion Correction, and the LOAM architecture.
> - **Code:** Implementation of 3D feature extraction and Scan matching.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the problem of LiDAR Motion Distortion (Skew) and how to deskew it.
2.  **Implement** Feature Extraction (Corner Points vs. Surface Points) based on smoothness.
3.  **Build** a Point-to-Line and Point-to-Plane ICP (Iterative Closest Point) registration algorithm.
4.  **Architect** a basic LOAM (Lidar Odometry and Mapping) pipeline.
5.  **Evaluate** odometry trajectory against Ground Truth (KITTI Odometry Dataset).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Velodyne/Ouster LiDAR (or Rosbag from robust dataset).

### Software Environment
```bash
pip install numpy open3d scipy
sudo apt install ros-humble-pcl-ros
```

### Prior Knowledge
- Rotation Matrices vs Quaternions.
- Least Squares Optimization.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Physics of Spinning LiDAR

A standard LiDAR spins at 10Hz (100ms per scan). If a car moves at 20m/s, it travels 2 meters *during a single scan*.
*   **Result:** The point cloud is "skewed" or distorted. A straight wall looks curved.
*   **Deskewing:** We must interpolate the robot's pose for every individual packet/point using IMU data or constant velocity assumption.

### 🔹 Part 2: LOAM (Lidar Odometry and Mapping)

Proposes splitting SLAM into two threads:
1.  **Odometry (High Freq, Low Drift):** Estimates velocity and coarse position using scan-to-scan matching.
2.  **Mapping (Low Freq, High Precision):** Aligns the current scan to the global map (scan-to-map).

#### 2.1 Feature Extraction
We don't match *all* 100k points. We select:
*   **Edge Points (High Curvature):** Corners, Tree trunks, Lamp posts.
*   **Planar Points (Low Curvature):** Ground, Walls.

**Curvature Calculation:**
For a point $p_i$, calculate roughness $c$:
$$ c = \frac{1}{|\mathcal{S}| \cdot ||p_i||} || \sum_{j \in \mathcal{S}, j \neq i} (p_j - p_i) || $$
*   High $c$ -> Edge Feature.
*   Low $c$ -> Planar Feature.

#### 2.2 Scan-to-Scan Matching
We find the transformation $T$ that minimizes distances.
*   **Edges:** Minimize distance from Point to the Line formed by nearest neighbors in previous scan.
*   **Surfaces:** Minimize distance from Point to the Plane formed by nearest neighbors.

**Optimization:** Levenberg-Marquardt (non-linear least squares).

### 🔹 Part 3: Modern Variants

*   **LeGO-LOAM:** Lightweight Ground-Optimized. Explicitly segments "Ground" points to stabilize $z, \text{roll}, \text{pitch}$ estimation.
*   **LIO-SAM:** Tightly couples Lidar with IMU using Factor Graphs (Day 11).

---

## 💻 Implementation: Feature Extractor & ICP

We will build the frontend of LOAM from scratch in Python.

### 🛠️ Project Structure
```text
day8_loam/
├── data/
│   └── kitti_sample/
├── src/
│   ├── feature_extract.py
│   ├── icp.py
│   └── deskew.py
└── run_odometry.py
```

### 👨‍💻 Code Implementation

#### 1. Feature Extraction (`src/feature_extract.py`)

```python
import numpy as np

def calculate_curvature(points, k=5):
    """
    points: [N, 3] ordered by ring/time
    k: neighborhood size
    """
    num_points = len(points)
    curvature = np.zeros(num_points)
    
    for i in range(k, num_points - k):
        diff = np.sum(points[i-k : i+k+1], axis=0) - (2*k + 1) * points[i]
        curvature[i] = np.linalg.norm(diff) / ((2*k + 1) * np.linalg.norm(points[i]))
        
    return curvature

def extract_features(points, curvature):
    # Sort by curvature
    # In practice, we split scan into 6 sectors and pick top features from each to ensure distribution
    
    indices = np.argsort(curvature)
    
    # Sharpest = Edges
    edge_indices = indices[-50:] 
    
    # Flattest = Planes
    surf_indices = indices[:1000]
    
    return points[edge_indices], points[surf_indices]
```

#### 2. Point-to-Plane ICP (`src/icp.py`)

Using Linearized Least Squares.
Minimize $E = \sum ((T \cdot p_s - p_t) \cdot n_t)^2$
Where $n_t$ is the normal vector of the target plane.

```python
import open3d as o3d

def point_to_plane_icp(source_pc, target_pc, threshold=0.2):
    # We use Open3D for speed, but understanding the math is key (Jacobian)
    
    # 1. Estimate Normals (if not present)
    target_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    
    # 2. Registration
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_pc, target_pc, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    return reg_p2l.transformation, reg_p2l.fitness
```

---

## 🔬 Lab Exercise: Running LeGO-LOAM via Docker

Writing a robust LOAM in Python is too slow for real-time. We will run the standard C++ implementation.

### 1. Lab Objectives
- Set up LeGO-LOAM with Docker.
- Play a KITTI rosbag.
- Visualize the mapping process in Rviz.

### 2. Step-by-Step Guide

#### Phase A: Docker Setup

```bash
# Dockerfile
FROM osrf/ros:humble-desktop
RUN apt-get update && apt-get install -y \
    ros-humble-pcl-ros \
    libgtsam-dev \
    libpcl-dev
    
# Clone LeGO-LOAM-BOR (ROS2 port)
WORKDIR /ros2_ws/src
RUN git clone https://github.com/facontidavide/LeGO-LOAM-BOR.git
WORKDIR /ros2_ws
RUN colcon build
```

#### Phase B: Execution

```bash
# Terminal 1
ros2 launch lego_loam_bor lego_loam.launch.py

# Terminal 2
ros2 bag play kitti_odometry_00.mcap
```

#### Phase C: Analysis
Observe the "Map Optimization" thread.
1.  **Green Points:** Ground (Constrains Z, Roll, Pitch).
2.  **Yellow Points:** Edges/Surfaces (Constrains X, Y, Yaw).
3.  **Result:** A drifted-less trajectory compared to pure Wheel Odometry.

---

## 🚀 Project: "Lidar Odometry Wrapper"

**Goal:** Create a ROS 2 node that subscribes to `/velodyne_points` and publishes `/odom`.
**Method:** Use Python Open3D ICP (Scan-to-Scan) as a "Poor Man's LOAM".

### 1. Node Structure

```python
class LidarOdom(Node):
    def __init__(self):
        super().__init__('lidar_odom')
        self.prev_cloud = None
        self.global_transform = np.eye(4)
        self.pub_odom = self.create_publisher(Odometry, '/odom_lidar', 10)
        
    def cb(self, msg):
        curr_cloud = self.ros_to_o3d(msg)
        curr_cloud = curr_cloud.voxel_down_sample(0.2) # Speed up
        
        if self.prev_cloud is not None:
            # Match
            trans, fitness = point_to_plane_icp(curr_cloud, self.prev_cloud)
            
            # Update Global Pose
            # T_global = T_global * T_incremental
            self.global_transform = self.global_transform @ trans
            
            # Publish
            self.publish_tf(self.global_transform)
            
        self.prev_cloud = curr_cloud
```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Z-axis Drift" (The robot flies away)
*   **Cause:** Not enough constraints in the Z direction (e.g., in a tunnel or long corridor w/o ceiling hits).
*   **Fix:** Use LeGO-LOAM (Ground constraints) or fuse with IMU gravity vector.

#### 2. "Degeneracy" (Odometry gets stuck)
*   **Cause:** Long hallway with no geometric features. ICP slides along the wall.
*   **Fix:** Detect degeneracy (check Eigenvalues of the Hessian matrix in optimization). If degenerate, rely on IMU/Encoders.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is Point-to-Plane better than Point-to-Point?
    *   **A:** Point-to-Point requires exact correspondence (which doesn't exist in sparse LiDAR). Point-to-Plane allows the point to slide along the planar surface, which mimics the true geometry and converges faster.
2.  **Q:** What is the difference between Odometry and Mapping?
    *   **A:** Odometry is local, incremental, and drifts. Mapping is global, searches for loop closures, and corrects drift.

### Challenge Task
> **Task:** Implement "Keyframe Selection".
> 1. Don't match every single frame.
> 2. Only update `prev_cloud` if the robot moved > 0.5 meters or rotated > 10 degrees.
> 3. Does this improve speed? Does it increase drift?

---

## 📚 Further Reading
- **LOAM:** Zhang and Singh (RSS 2014).
- **LeGO-LOAM:** Shan and Englot (IROS 2018).
- **LIO-SAM:** Shan et al. (IROS 2020).

---

**Day 8 Complete**
