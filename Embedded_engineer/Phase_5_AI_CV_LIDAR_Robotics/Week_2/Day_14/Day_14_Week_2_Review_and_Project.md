# Day 14: Week 2 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 2: Advanced SLAM & State Estimation

---

> **📝 Content Creator Instructions:**
> We have covered Odometry, Fusion, and Optimization. Now we build the "Holy Grail" of navigation: Multi-Modal SLAM.
> - **Goal:** Implement Lidar-Visual-Inertial Odometry (LVIO) logic.
> - **Code:** A Factor Graph integrating IMU Preintegration, LiDAR registration, and Visual Features.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Synthesize** LOAM, VIO, and Factor Graphs into a single architecture (LVI-SAM style).
2.  **Implement** IMU Preintegration to handle high-frequency update constraints.
3.  **Perform** Extrinsic Calibration between LiDAR and Camera using a Checkerboard.
4.  **Benchmark** the unified system on the KITTI or HILTI dataset.

---

## 📚 Week 2 Review: The State Estimator

| Day | Component | Key Concept | Pros | Cons |
|-----|-----------|-------------|------|------|
| **8** | **LiDAR** | ICP / LOAM | High Precision, Night Vision | Skew, Geometry Degeneracy |
| **9** | **Vision** | Epipolar / ORB | Texture Rich, Loop Closure | Scale Drift, Motion Blur |
| **10** | **Fusion** | EKF | Real-time, Probabilistic | Linearization Error |
| **11** | **Graph** | GTSAM | Global Consistency | Computationally Heavy |
| **12** | **Loops** | BoW | Corrects Drift | Perceptual Aliasing |

### The "Golden Rule" of SLAM
*   **Local Accuracy:** Comes from IMU + LiDAR (Odometry).
*   **Global Accuracy:** Comes from Loop Closure (Vision/Place Recognition).

---

## 🚀 Weekly Capstone: "Titan-SLAM" (LVIO System)

**Scenario:** An autonomous delivery robot must navigate a variety of environments: Indoors (Texture rich, narrow), Outdoors (Texture poor, open), and Tunnels (GPS denied, dark).
**Solution:** Tight coupling of LiDAR, Vision, and IMU.

### 🛠️ Project Structure
```text
week2_capstone/
├── config/
│   └── params_titan.yaml
├── src/
│   ├── lvio_system.py
│   ├── imu_preintegrator.py
│   ├── visual_frontend.py
│   └── lidar_frontend.py
└── launch/
    └── titan_slam.launch.py
```

### 👨‍💻 Code Implementation: The Factor Graph

We will use GTSAM to fuse preintegrated IMU factors with LiDAR Odometry factors.

#### 1. IMU Preintegration (`src/imu_preintegrator.py`)
Raw IMU comes at 200Hz. Adding 200 nodes/sec to the graph is impossible.
**Preintegration** condenses 20 IMU messages into **one** relative motion constraint (Factor) between Pose $i$ and Pose $j$.

```python
import gtsam
import numpy as np

class IMUPreintegrator:
    def __init__(self, bias_acc, bias_gyr):
        params = gtsam.PreintegrationParams.MakeSharedU()
        params.setAccelerometerCovariance(1e-3 * np.eye(3))
        params.setGyroscopeCovariance(1e-4 * np.eye(3))
        params.setIntegrationCovariance(1e-8 * np.eye(3))
        
        self.pim = gtsam.PreintegratedImuMeasurements(params, gtsam.imuBias.ConstantBias(bias_acc, bias_gyr))
        
    def add_measurement(self, acc, gyr, dt):
        self.pim.integrateMeasurement(acc, gyr, dt)
        
    def get_factor(self, pose_i_key, vel_i_key, pose_j_key, vel_j_key, bias_key):
        return gtsam.ImuFactor(pose_i_key, vel_i_key, pose_j_key, vel_j_key, bias_key, self.pim)
        
    def reset(self):
        self.pim.resetIntegration()
```

#### 2. The LVI Graph (`src/lvio_system.py`)

```python
class TitanSLAM:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.imu_pre = IMUPreintegrator(bias_acc, bias_gyr)
        
        self.key_idx = 0
        
        # 1. Add Prior Factor (Start at 0)
        self.add_prior(0)
        
    def callback(self, lidar_odom, visual_odom, imu_buffer):
        key_i = self.key_idx
        key_j = self.key_idx + 1
        
        # 2. Add IMU Factor (High Rate Constraint)
        for imu_msg in imu_buffer:
             self.imu_pre.add_measurement(imu_msg.acc, imu_msg.gyr, imu_msg.dt)
        
        imu_factor = self.imu_pre.get_factor(X(key_i), V(key_i), X(key_j), V(key_j), B(key_i))
        self.graph.add(imu_factor)
        
        # 3. Add LiDAR Factor (Geometric Constraint)
        # Scan-to-Map matching provides absolute pose constraint
        lidar_noise = gtsam.noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
        lidar_factor = gtsam.BetweenFactorPose3(X(key_i), X(key_j), lidar_odom.relative_pose, lidar_noise)
        self.graph.add(lidar_factor)
        
        # 4. Add Visual Factor (Optional Loop Closure)
        if visual_odom.has_loop:
            loop_factor = gtsam.BetweenFactorPose3(X(key_j), X(loop_idx), loop_trans, loop_noise)
            self.graph.add(loop_factor)
            
        # 5. Optimize (iSAM2)
        self.isam.update(self.graph, self.values)
        result = self.isam.calculateEstimate()
        
        # 6. Reset for next step
        self.imu_pre.reset()
        self.key_idx += 1
```

---

## ⚡ Extrinsic Calibration

LVI-SLAM fails if the transformation between LiDAR and Camera is wrong.
**Task:**
1.  Print a Checkerboard.
2.  Hold it in front of the robot.
3.  **Camera** detects corners ($u, v$).
4.  **LiDAR** detects planar board surface ($n, d$).
5.  **PnP:** Solve for $T_{cam}^{lidar}$.
*   *Tool:* `calibration_toolkit` or `lidar_camera_calibration` package.

---

## 📝 Self-Assessment Quiz

1.  **Architecture:**
    *   Why do we perform IMU Preintegration instead of just adding every IMU reading to the graph?
        *   **A:** To reduce graph size. Adding 200 nodes/sec is computationally intractable. Preintegration creates 1 node/keyframe but preserves the motion information.

2.  **Degeneracy:**
    *   In a long, featureless tunnel, which sensor do we trust?
        *   **A:** IMU (for short term) and Wheel Encoders. Vision fails (no features). LiDAR fails (geometrically degenerate in longitudinal direction).

3.  **Optimization:**
    *   What is the role of the "Bias" variable in the Factor Graph?
        *   **A:** IMU bias drifts over time due to temperature. By adding Bias as a variable to be estimated, the graph can "learn" the current bias and correct the measurements dynamically.

---

## ⏭️ Look Ahead: Week 3
Now that we know **Where we are** and **What is around us**...
**Week 3: Advanced Planning & Navigation.**
*   MPPI (Model Predictive Path Integral) Control.
*   Hybrid A* Search.
*   Traversability Analysis.

---

**Week 2 Complete**
