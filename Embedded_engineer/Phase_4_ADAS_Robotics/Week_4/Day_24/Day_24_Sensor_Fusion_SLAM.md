# Day 24: Sensor Fusion for SLAM (VIO/LIO)
## Phase 4: ADAS & Robotics Systems | Week 4: Localization & Mapping (SLAM)

---

> **📝 Day 24 Focus:**
> Cameras and Lidars are slow (10-30Hz) and can fail in textureless or geometric-less environments. IMUs are fast (200Hz+) and robust, but drift quickly. **Visual-Inertial Odometry (VIO)** and **Lidar-Inertial Odometry (LIO)** combine the best of both worlds. Today, we master **Factor Graphs** and **IMU Pre-integration**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Contrast** Loosely-Coupled (Filter-based) and Tightly-Coupled (Optimization-based) fusion.
2.  **Explain** the concept of IMU Pre-integration to handle high-frequency IMU data in a low-frequency graph.
3.  **Construct** a Factor Graph with Prior Factors, Odometry Factors, and IMU Factors.
4.  **Implement** a basic Factor Graph optimization using `gtsam` (Georgia Tech Smoothing and Mapping).
5.  **Analyze** the architecture of LIO-SAM (Lidar Inertial Odometry via Smoothing and Mapping).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 13:** GraphSLAM.
-   **Day 22/23:** Lidar/Visual Odometry.
-   **Physics:** Kinematics (Velocity, Acceleration, Bias).

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows/Mac with Python).

### Software Stack
-   **Python Libraries:** `gtsam` (pip install gtsam), `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Coupling Strategies

#### 1.1 Loosely-Coupled
-   **Process:** VO/LO computes a pose. IMU computes a pose. Fuse them in an EKF.
-   **Pros:** Simple, modular.
-   **Cons:** Information loss. If VO fails (featureless wall), the EKF has no way to help the VO recover.

#### 1.2 Tightly-Coupled
-   **Process:** Raw feature measurements (pixels) and raw IMU measurements (accel/gyro) are optimized jointly in a single cost function.
-   **Pros:** Robust. IMU predicts where features should be, helping VO track even with motion blur.
-   **Cons:** Complex, computationally heavy.

---

### 🔹 Part 2: Factor Graphs & GTSAM

A **Factor Graph** is a bipartite graph representing the factorization of a function (probability distribution).
-   **Variable Nodes:** States (Pose, Velocity, Bias).
-   **Factor Nodes:** Constraints (Measurements).

We want to maximize the product of all factors (minimize sum of squared errors).

#### 2.1 IMU Pre-integration
IMU comes at 200Hz. Camera at 30Hz.
Between two keyframes $i$ and $j$, we have ~7 IMU measurements.
Adding 7 pose nodes to the graph is expensive.
**Pre-integration:** We integrate the 7 IMU readings into a *single* relative constraint (Delta Pose, Delta Velocity) between $i$ and $j$.
-   This "Pre-integrated Factor" is independent of the absolute pose, so we don't need to re-integrate when the linearization point changes.

---

### 🔹 Part 3: LIO-SAM Architecture

LIO-SAM is the state-of-the-art Lidar SLAM.
1.  **Image Projection:** Deskews point cloud using IMU.
2.  **Feature Extraction:** Edge/Planar points (like LOAM).
3.  **Factor Graph:**
    -   **Lidar Odometry Factor:** Scan-to-Map matching.
    -   **IMU Pre-integration Factor:** Connects consecutive poses.
    -   **GPS Factor:** Optional global constraint.
    -   **Loop Closure Factor:** Connects current pose to past pose.

---

## 💻 Implementation: Factor Graph Fusion

We will use `gtsam` to fuse noisy Odometry (e.g., from VO) with noisy GPS measurements.
*Note: Full IMU pre-integration is complex to implement from scratch in a single script, so we demonstrate the Factor Graph concept with GPS/Odom fusion, which is the "Back-end" logic of LIO.*

### 🛠️ Setup
Create `week4_day24` and `fusion_graph.py`.

```bash
mkdir -p ~/ros2_ws/src/week4_day24
cd ~/ros2_ws/src/week4_day24
pip install gtsam
touch fusion_graph.py
```

### 👨‍💻 Code: GTSAM Fusion

```python
import gtsam
import numpy as np
import matplotlib.pyplot as plt

def run_fusion():
    # 1. Create a Factor Graph container
    graph = gtsam.NonlinearFactorGraph()
    
    # 2. Add a Prior Factor (Initial Pose)
    # We are at (0,0,0) with high confidence
    prior_mean = gtsam.Pose2(0.0, 0.0, 0.0)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    graph.add(gtsam.PriorFactorPose2(1, prior_mean, prior_noise))
    
    # 3. Add Odometry Factors (Relative Motion)
    # Robot moves 2m in X direction every step
    odom_mean = gtsam.Pose2(2.0, 0.0, 0.0)
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    
    # Add factors for 10 steps: 1->2, 2->3, ...
    for i in range(1, 11):
        graph.add(gtsam.BetweenFactorPose2(i, i+1, odom_mean, odom_noise))
        
    # 4. Add GPS Factors (Absolute Position)
    # GPS is noisy but absolute (no drift)
    # Let's say GPS measures (x, y)
    gps_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0])) # High noise
    
    # Simulate GPS readings (True pos + noise)
    # True pos at step i is x = (i-1)*2
    for i in range(1, 12):
        true_x = (i-1) * 2.0
        measured_x = true_x + np.random.normal(0, 1.0) # Add noise
        measured_y = 0.0 + np.random.normal(0, 1.0)
        
        gps_point = np.array([measured_x, measured_y])
        # Unary factor on Pose i
        graph.add(gtsam.GPSFactorPose2(i, gtsam.Point2(gps_point), gps_noise))
        
    # 5. Initialize Estimates (Initial Guess)
    initial_estimate = gtsam.Values()
    # Initialize with Dead Reckoning (Odometry only) - Drift accumulates
    current_pose = gtsam.Pose2(0.0, 0.0, 0.0)
    initial_estimate.insert(1, current_pose)
    
    dr_path = [(0,0)]
    
    for i in range(1, 11):
        # Apply noisy odometry for initialization
        noisy_odom = gtsam.Pose2(2.0 + np.random.normal(0, 0.1), 
                                 0.0 + np.random.normal(0, 0.1), 
                                 0.0)
        current_pose = current_pose.compose(noisy_odom)
        initial_estimate.insert(i+1, current_pose)
        dr_path.append((current_pose.x(), current_pose.y()))
        
    # 6. Optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    
    # 7. Visualize
    print("Final Result:")
    result.print()
    
    # Extract path
    opt_path = []
    for i in range(1, 12):
        pose = result.atPose2(i)
        opt_path.append((pose.x(), pose.y()))
        
    dr_path = np.array(dr_path)
    opt_path = np.array(opt_path)
    
    plt.figure(figsize=(10, 5))
    plt.plot(dr_path[:,0], dr_path[:,1], 'r--o', label='Dead Reckoning (Drift)')
    plt.plot(opt_path[:,0], opt_path[:,1], 'g-o', label='Fused Estimate (Graph)')
    plt.plot(np.arange(0, 22, 2), np.zeros(11), 'k*', label='Ground Truth')
    plt.title("Factor Graph Fusion (Odometry + GPS)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    run_fusion()
```

---

## 🔬 Lab Exercise: Loop Closure

### Lab Objectives
1.  Run the script.
2.  **Modify:** Change the trajectory to a loop (Square: Move X, Move Y, Move -X, Move -Y).
3.  **Drift:** Observe that Dead Reckoning does not return to (0,0).
4.  **Loop Factor:** Add a `BetweenFactorPose2` between the last node and the first node (Pose 1).
    ```python
    # Constraint: Pose N is same as Pose 1
    graph.add(gtsam.BetweenFactorPose2(N, 1, gtsam.Pose2(0,0,0), small_noise))
    ```
5.  **Result:** The entire graph snaps into a perfect square.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Optimization Divergence
**Symptom:** Result contains NaNs or huge numbers.
**Cause:**
-   Initial guess is too far from solution (Local Minima).
-   Constraints are contradictory (e.g., Odom says move 10m, GPS says move 0m, both with tiny noise).
**Solution:** Check noise models. GPS noise should be large ($>1m$), Odom noise small ($<0.1m$).

#### 2. Graph Sparsity
**Symptom:** Slow optimization.
**Cause:** Adding too many factors without marginalization.
**Solution:** Use **iSAM2** (Incremental Smoothing and Mapping). It only updates the parts of the graph affected by new measurements (Bayes Tree), allowing real-time performance.

---

## ⚡ Optimization & Best Practices

### 1. Bias Estimation
IMUs have bias (accelerometer offset).
-   In the Factor Graph, add **Bias** as a state variable to be estimated.
-   The optimizer will find the bias value that explains the drift, effectively calibrating the IMU online.

### 2. Extrinsics Estimation
We often don't know the exact transform between Camera and IMU.
-   Add $T_{bc}$ (Body-Camera transform) as a variable in the graph.
-   The system will self-calibrate the extrinsics during motion.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the main benefit of Tightly-Coupled VIO?
    *   **A:** Robustness. The IMU drives the system when visual features are lost (e.g., lights off, white wall).
2.  **Q:** Why do we "Pre-integrate" IMU measurements?
    *   **A:** To avoid adding hundreds of nodes to the graph for high-frequency IMU data. It summarizes motion between keyframes into a single constraint.
3.  **Q:** What is a "Prior Factor"?
    *   **A:** A factor that anchors a variable to a specific value (absolute constraint), preventing the entire map from floating away.

### Challenge Task
**Task:** 1D VIO Simulation.
1.  Simulate a robot moving in 1D with constant acceleration.
2.  Generate noisy Accel readings and noisy Position readings.
3.  Build a graph to estimate Position, Velocity, and Accel Bias.

---

## 📚 Further Reading & References
-   [GTSAM Tutorials](https://gtsam.org/tutorials/intro.html)
-   [LIO-SAM Paper](https://github.com/TixiaoShan/LIO-SAM)

---

**Day 24 Complete** | Phase 4: ADAS & Robotics Systems | Week 4: Localization & Mapping (SLAM)
