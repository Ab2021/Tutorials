# Day 22: Lidar Odometry & Mapping (LOAM)
## Phase 4: ADAS & Robotics Systems | Week 4: Localization & Mapping (SLAM)

---

> **📝 Day 22 Focus:**
> Cameras are great for semantics, but for precise geometry, **Lidar** is king. To build a map or localize within it, we need to align Lidar scans. Today, we master **ICP (Iterative Closest Point)** and the legendary **LOAM** algorithm, which separates high-frequency odometry from low-frequency mapping.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the structure of 3D Point Clouds (XYZ, Intensity, Ring).
2.  **Implement** the ICP algorithm to align two point clouds and estimate relative motion.
3.  **Deconstruct** the LOAM architecture: Feature Extraction (Edges/Planes) and the two-stage optimization.
4.  **Use** Open3D to visualize and register point clouds in Python.
5.  **Analyze** the difference between Scan-to-Scan and Scan-to-Map matching.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Linear Algebra:** SVD (Singular Value Decomposition) for rigid body transformation.
-   **Day 13:** SLAM Fundamentals.
-   **Python:** NumPy.

### Hardware Requirements
-   **Lidar Data:** We will use sample `.pcd` or `.ply` files (e.g., from KITTI dataset).

### Software Stack
-   **Python Libraries:** `open3d`, `numpy`, `copy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Iterative Closest Point (ICP)

Problem: Given Source Cloud $P$ and Target Cloud $Q$, find rotation $R$ and translation $t$ that minimizes distance.

$$ E(R, t) = \sum_i || q_i - (R p_i + t) ||^2 $$

**Algorithm:**
1.  **Associate:** For each point in $P$, find the nearest neighbor in $Q$.
2.  **Estimate:** Compute best $R, t$ to align the pairs (using SVD).
3.  **Transform:** Apply $R, t$ to $P$.
4.  **Repeat** until convergence.

#### 1.1 Variants
-   **Point-to-Point:** Minimizes distance between points. (Simple, but slides along flat walls).
-   **Point-to-Plane:** Minimizes distance from point to the *plane* of the target. (Faster convergence, handles sliding).

---

### 🔹 Part 2: LOAM (Lidar Odometry and Mapping)

LOAM (Zhang et al., 2014) revolutionized Lidar SLAM by splitting the problem.

#### 2.1 Feature Extraction
Raw clouds are too heavy (100k points). LOAM extracts features based on **Roughness** (Curvature).
-   **Edge Points (Sharp):** High curvature (corners, poles, tree trunks).
-   **Planar Points (Flat):** Low curvature (ground, walls).

#### 2.2 Lidar Odometry (10Hz)
-   Matches **Scan-to-Scan**.
-   Matches Edge points to Edge lines.
-   Matches Planar points to Planar patches.
-   Provides a rough, high-frequency pose estimate.

#### 2.3 Lidar Mapping (1Hz)
-   Matches **Scan-to-Map**.
-   Builds a global map using the accumulated scans.
-   Refines the pose with higher accuracy.

---

## 💻 Implementation: ICP with Open3D

We will implement a script `lidar_reg.py` that:
1.  Generates two synthetic point clouds (one rotated/translated).
2.  Aligns them using ICP.
3.  Visualizes the result.

### 🛠️ Setup
Create `week4_day22` and `lidar_reg.py`.

```bash
mkdir -p ~/ros2_ws/src/week4_day22
cd ~/ros2_ws/src/week4_day22
pip install open3d
touch lidar_reg.py
```

### 👨‍💻 Code: ICP Registration

```python
import open3d as o3d
import numpy as np
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0]) # Yellow
    target_temp.paint_uniform_color([0, 0.651, 0.929]) # Blue
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4512],
                                      up=[-0.3402, -0.9189, -0.1996])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    
    # Use built-in demo data
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    
    # Apply initial transformation to source (simulating motion)
    trans_init = np.asarray([[0.86, 0.0, -0.5, 0.5],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.5, 0.0, 0.86, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    
    draw_registration_result(source, target, np.identity(4))
    return source, target, trans_init

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This result is accurate,")
    print("   but requires a good initial guess (from RANSAC).")
    
    # Estimate normals for Point-to-Plane
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

if __name__ == "__main__":
    voxel_size = 0.05  # means 5cm for this dataset
    
    # 1. Load Data
    source, target, trans_init = prepare_dataset(voxel_size)
    
    # 2. Preprocess (Downsample + Features)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    
    # 3. Global Registration (RANSAC) - Rough alignment
    # result_ransac = execute_global_registration(source_down, target_down,
    #                                             source_fpfh, target_fpfh,
    #                                             voxel_size)
    
    # Faster alternative to RANSAC
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   voxel_size)
    
    print("Global Registration Result:")
    print(result_fast)
    draw_registration_result(source_down, target_down, result_fast.transformation)
    
    # 4. Local Refinement (ICP) - Precise alignment
    result_icp = refine_registration(source, target, result_fast, voxel_size)
    
    print("ICP Refinement Result:")
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation)
```

---

## 🔬 Lab Exercise: Odometry Drift

### Lab Objectives
1.  Run the script.
2.  **Observation:** Global registration snaps the clouds roughly together. ICP tightens the fit perfectly.
3.  **Experiment:** Add noise to the source cloud.
    ```python
    noise = np.random.normal(0, 0.02, np.asarray(source.points).shape)
    source.points = o3d.utility.Vector3dVector(np.asarray(source.points) + noise)
    ```
    -   See how ICP handles noise.
4.  **Drift:** If you run ICP sequentially ($1 \to 2$, $2 \to 3$, ...), small errors accumulate. This is why we need **Mapping** (Scan-to-Map) to correct it.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. ICP Diverges
**Symptom:** Clouds fly apart.
**Cause:** Initial guess was too far off. ICP is a local optimizer.
**Solution:** Use Global Registration (RANSAC/FPFH) first to get a good initialization.

#### 2. Slow Performance
**Cause:** Too many points.
**Solution:** Voxel Downsampling. You don't need 100k points to find a wall. 5k is enough.

#### 3. Sliding (Corridor Problem)
**Symptom:** Cloud aligns perfectly in Y and Z, but slides freely in X (hallway).
**Cause:** Lack of geometric features in the X direction.
**Solution:** Detect degeneracy (eigenvalues of covariance) and trust Odometry/IMU for the degenerate axis.

---

## ⚡ Optimization & Best Practices

### 1. GICP (Generalized ICP)
Combines Point-to-Point and Point-to-Plane into a probabilistic framework. More robust.

### 2. NDT (Normal Distributions Transform)
Instead of matching points, it divides space into cells, models each cell as a Gaussian, and matches the source cloud to these Gaussians.
-   **Pros:** Faster, smoother cost function. Used in Autoware.

### 3. Deskewing
Lidar spins while the car moves. The start of the scan is at $t=0$, the end is at $t=100ms$.
-   If car moves 20m/s, the car moved 2m *during* the scan.
-   **Solution:** Use IMU to "unwind" the distortion before ICP.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why do we downsample point clouds?
    *   **A:** To reduce computational cost and noise.
2.  **Q:** What is the difference between Point-to-Point and Point-to-Plane ICP?
    *   **A:** Point-to-Plane minimizes distance to the surface normal, allowing sliding along flat surfaces, which helps convergence.
3.  **Q:** What is the "Corridor Problem"?
    *   **A:** In a featureless hallway, the robot cannot determine its position along the hallway (longitudinal ambiguity).

### Challenge Task
**Task:** Implement Feature Extraction.
1.  Load a raw Lidar scan.
2.  Calculate curvature for each point (based on neighbors).
3.  Colorize points: Red for Edges (High curvature), Blue for Planes (Low curvature).

---

## 📚 Further Reading & References
-   [OpenCV ICP Tutorial](http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)
-   [LOAM Paper (Zhang 2014)](https://www.ri.cmu.edu/pub_files/2014/7/Ji_Zhang_RSS2014.pdf)

---

**Day 22 Complete** | Phase 4: ADAS & Robotics Systems | Week 4: Localization & Mapping (SLAM)
