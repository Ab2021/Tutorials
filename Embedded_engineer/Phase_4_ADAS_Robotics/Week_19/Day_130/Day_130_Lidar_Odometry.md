# Day 130: Lidar Odometry (ICP/NDT)
## Phase 4: ADAS & Robotics Systems | Week 19: Localization & SLAM

---

> **📝 Day 130 Focus:**
> Cameras struggle with light changes. Lidar provides precise geometry. **Lidar Odometry** estimates motion by aligning point clouds. If Cloud A matches Cloud B after moving 1 meter, then the car moved 1 meter.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the Point Cloud Registration problem.
2.  **Implement** Iterative Closest Point (ICP) algorithm.
3.  **Understand** Normal Distributions Transform (NDT) for robust matching.
4.  **Use** `open3d` to align two scans and recover the transformation matrix.
5.  **Analyze** the pros and cons of ICP vs NDT.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Linear Algebra:** Rigid Body Transformation ($T = [R|t]$).
-   **Day 124:** Point Clouds.

### Hardware Requirements
-   **None:** Kitti dataset sample recommended.

### Software Stack
-   **Python:** `open3d`, `numpy`, `copy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Registration Problem

Given Source Cloud $S$ and Target Cloud $T$.
Find $R, t$ such that $T \approx R \cdot S + t$.
Minimize Error: $E = \sum || t_i - (R s_i + t) ||^2$.

### 🔹 Part 2: ICP (Iterative Closest Point)

1.  **Associate:** For each point in $S$, find the closest point in $T$.
2.  **Estimate:** Calculate $R, t$ that minimizes distance between pairs (SVD).
3.  **Transform:** Apply $R, t$ to $S$.
4.  **Repeat:** Until convergence.
-   *Pros:* Simple, accurate if close.
-   *Cons:* Needs good initial guess, gets stuck in local minima.

### 🔹 Part 3: NDT (Normal Distributions Transform)

Instead of matching points to points, match points to **Probability Distributions**.
1.  Divide Target $T$ into grid cells (voxels).
2.  Calculate Mean $\mu$ and Covariance $\Sigma$ for points in each cell.
3.  Find $R, t$ that maximizes the likelihood of Source points fitting these distributions.
-   *Pros:* Robust to outliers, faster (grid lookup).
-   *Cons:* Grid size parameter is sensitive.

---

## 💻 Implementation: ICP with Open3D

**Scenario:**
-   Input: Two point clouds (Scan 1 and Scan 2).
-   Task: Find the relative motion.

### 🛠️ Setup
Create `week19_day130` and `lidar_odom.py`.

```bash
mkdir -p ~/ros2_ws/src/week19_day130
cd ~/ros2_ws/src/week19_day130
pip install open3d
touch lidar_odom.py
```

### 👨‍💻 Code: ICP Alignment

```python
import open3d as o3d
import numpy as np
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0]) # Yellow (Source)
    target_temp.paint_uniform_color([0, 0.651, 0.929]) # Blue (Target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4512],
                                      up=[-0.3402, -0.9189, -0.1996])

def main():
    # 1. Generate Synthetic Data
    # Create a source cloud (Cube)
    source = o3d.geometry.PointCloud()
    points = np.random.rand(1000, 3) # Random points in 1x1x1 cube
    source.points = o3d.utility.Vector3dVector(points)
    
    # Create a target cloud (Rotated and Translated)
    target = copy.deepcopy(source)
    
    # Ground Truth Transformation
    # Rotate 30 deg around Z, Translate (0.5, 0.2, 0)
    theta = np.radians(30)
    gt_trans = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0.5],
        [np.sin(theta), np.cos(theta), 0, 0.2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    target.transform(gt_trans)
    
    # Add noise
    target.points = o3d.utility.Vector3dVector(np.asarray(target.points) + np.random.normal(0, 0.01, (1000, 3)))
    
    print("Initial State (Misaligned)...")
    draw_registration_result(source, target, np.eye(4))
    
    # 2. Apply ICP
    threshold = 0.02 # Max distance for correspondence
    trans_init = np.eye(4) # Initial guess (Identity)
    
    print("Running ICP...")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    
    print("ICP Converged:")
    print(reg_p2p.transformation)
    print("\nGround Truth:")
    print(gt_trans)
    
    print("\nFitness:", reg_p2p.fitness)
    print("RMSE:", reg_p2p.inlier_rmse)
    
    draw_registration_result(source, target, reg_p2p.transformation)

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Initial Guess

### Lab Objectives
1.  Run the script.
2.  **Observation:** The Yellow cube snaps perfectly onto the Blue cube.
3.  **Experiment:**
    -   Change `theta` to 90 degrees.
    -   Run ICP with `trans_init = Identity`.
    -   **Result:** ICP fails (misaligned).
    -   **Reason:** ICP is a local optimizer. It matches the closest points. If the rotation is too large, it matches the wrong faces of the cube.
    -   **Solution:** Provide a better initial guess (e.g., from IMU or Constant Velocity Model).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Point-to-Point vs Point-to-Plane
**Symptom:** Slow convergence on flat walls.
**Cause:** Point-to-Point tries to match specific dots.
**Solution:** Use **Point-to-Plane** ICP. It minimizes the distance from a point to the *surface* of the target. Much faster for Lidar data (which scans surfaces).

#### 2. Dynamic Objects
**Symptom:** Odometry drifts when a large truck passes by.
**Cause:** ICP tries to align the static world *and* the moving truck.
**Solution:** Filter out dynamic objects (using Deep Learning or Ray Casting checks) before registration.

---

## ⚡ Optimization & Best Practices

### 1. LOAM (Lidar Odometry and Mapping)
The gold standard algorithm.
-   **Feature Extraction:** Extracts "Edge Points" (sharp corners) and "Planar Points" (flat walls).
-   **Matching:** Matches Edges to Edges, Planes to Planes.
-   **Frequency:** Odometry at 10Hz, Mapping at 1Hz.

### 2. GICP (Generalized ICP)
Combines Point-to-Point and Point-to-Plane in a probabilistic framework.
-   More robust than standard ICP.
-   Available in PCL (Point Cloud Library).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why does ICP need an initial guess?
    *   **A:** Because the cost function is non-convex (many local minima). It needs to start in the "basin of attraction" of the global minimum.
2.  **Q:** What is the advantage of NDT over ICP?
    *   **A:** NDT doesn't need explicit point correspondences. It matches the statistical shape of the cloud. It's often more robust to large displacements.
3.  **Q:** How do we get the initial guess in a real car?
    *   **A:** From the previous velocity ($P_t = P_{t-1} + V \Delta t$) or from the IMU/Wheel Odometry.

### Challenge Task
**Task:** Point-to-Plane.
1.  Estimate normals for the target cloud: `target.estimate_normals()`.
2.  Change estimation method to `TransformationEstimationPointToPlane()`.
3.  Compare convergence speed (iterations) vs Point-to-Point.

---

## 📚 Further Reading & References
-   [Open3D ICP Tutorial](http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)
-   [LOAM Paper](https://www.ri.cmu.edu/pub_files/2014/7/Ji_LidarMapping_RSS2014_v8.pdf)

---

**Day 130 Complete** | Phase 4: ADAS & Robotics Systems | Week 19: Localization & SLAM
