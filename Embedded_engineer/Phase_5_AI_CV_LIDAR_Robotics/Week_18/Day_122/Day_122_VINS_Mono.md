# Day 122: Visual Inertial Odometry (VINS-Mono)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 18: Aerial & Underwater Robotics

---

> **📝 Content Creator Instructions:**
> Flying blind is impossible. Flying with just a camera is hard. Flying with Camera + IMU is robust.
> - **Focus:** VINS (Visual-Inertial Navigation System), IMU Pre-integration, Scale Ambiguity in Monocular vision, and testing with the EuRoC Dataset.
> - **Code:** Setup `VINS-Fusion` on ROS 2, and a Python node `path_plotter.py` to compare Ground Truth vs VINS Estimate.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** why Monocular SLAM has no scale (The world could be tiny or huge) and how IMU solves this (Accelerometers measure *true* meters/$s^2$).
2.  **Configure** VINS-Fusion for a Realsense D435i (Stereo + IMU).
3.  **Visualize** the feature tracking and point cloud in Rviz.
4.  **Analyze** drift over time using EVO tools.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Camera with IMU (Realsense D435i / Oak-D). Or Dataset.

### Software Environment
```bash
# VINS-Fusion port for ROS 2 (or native ROS 1 bridge)
sudo apt install ros-humble-vins-fusion
# Or compile from source
```

### Prior Knowledge
- Optimization (Bundle Adjustment).
- IMU Noise (Bias, Random Walk).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why VIO?

*   **GPS:** Fails indoors. 5m accuracy.
*   **Lidar:** Heavy. Power hungry. (Bad for small drones).
*   **Camera:** Lightweight.
    *   **Mono:** Scale Ambiguity.
    *   **Stereo:** Limited range.
*   **IMU:** High speed (200Hz). Drifts fast (Double integration of noise).
*   **VIO:** Fuses Camera (Drifts slowly, No scale) + IMU (No drift *locally*, Absolute Scale). Best of both.

### 🔹 Part 2: IMU Pre-Integration

Optimization takes time (e.g., 100ms). IMU data comes at 5ms.
We cannot optimize every IMU reading.
*   **Technique:** "Pre-integrate" 20 IMU readings into a single "Relative Motion Constraint" between Image Frame $k$ and Frame $k+1$.
*   This greatly reduces the size of the Factor Graph.

### 🔹 Part 3: Initialization

The hardest part.
1.  Drone must move (Excitation).
2.  Algorithm aligns "Visual Structure" with "Accelerometer Gravity Vector".
3.  Estimates local scale and gravity direction.
4.  **Static Initialization:** New methods allow starting while stationary (using Stereo).

---

## 💻 Implementation: Running VINS on Dataset

We will use the EuRoC MAV Dataset (Standard benchmark).

### 🛠️ Project Structure
```text
day122_vio/
├── config/
│   ├── euroc_mono_imu.yaml
├── launch/
│   ├── run_euroc.launch.py
└── src/
    ├── comparison_node.py
```

### 👨‍💻 Config (`config/euroc_mono_imu.yaml`)

Key parameters for VINS.
*(Simplified snippet)*

```yaml
# Sensor
imu: 1
num_of_cam: 1

# IMU Params (Critical!)
acc_n: 0.1          # Accelerometer noise
gyr_n: 0.01         # Gyroscope noise
acc_w: 0.0002       # Accelerometer bias random walk
gyr_w: 2.0e-5       # Gyro bias walk
g_norm: 9.81007     # Gravity magnitude

# External Calibration (Camera to IMU)
# Rotation and Translation must be exact (mm precision)
body_T_cam0: !!opencv-matrix
   rows: 4
   cols: 4
   dt: d
   data: [0.014, -0.99, 0.007, -0.02,
          0.99, 0.0149, -0.01, -0.06,
          0.01, 0.007, 0.99, 0.007,
          0, 0, 0, 1]

# Feature Tracking
max_cnt: 150        # Max features
min_dist: 30        # Min pixel distance between features
freq: 10            # Image freq
```

### 👨‍💻 Launch (`launch/run_euroc.launch.py`)

Using `rosbag2` to play data.

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    # 1. Play Bag
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '/path/to/MH_01_easy', '--clock'],
        output='screen'
    )
    
    # 2. VINS Node
    vins_node = Node(
        package='vins_fusion',
        executable='vins_node',
        parameters=['config/euroc_mono_imu.yaml'],
        remappings=[
            ('/imu0', '/imu/data_raw'),
            ('/cam0/image_raw', '/cam0/image_raw')
        ]
    )
    
    # 3. Rviz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        # arguments=['-d', 'vins.rviz']
    )
    
    return LaunchDescription([bag_play, vins_node, rviz])
```

### 👨‍💻 Comparison Node (`src/comparison_node.py`)

Estimates Error between `ground_truth` and `vins_estimator/odometry`.

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np

class ErrorMonitor(Node):
    def __init__(self):
        super().__init__('error_monitor')
        
        self.gt_pos = None
        self.est_pos = None
        
        # Subs
        self.create_subscription(Odometry, '/gt/odom', self.gt_cb, 10)
        self.create_subscription(Odometry, '/vins_estimator/odometry', self.est_cb, 10)
        
        self.create_timer(1.0, self.print_metrics)

    def gt_cb(self, msg):
        self.gt_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def est_cb(self, msg):
        self.est_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def print_metrics(self):
        if self.gt_pos is not None and self.est_pos is not None:
            # Note: GT and Est need to be aligned (Transformation)
            # This is simplified: Assume aligned at start or just raw diff
            error = np.linalg.norm(self.gt_pos - self.est_pos)
            self.get_logger().info(f"Euclidean Error: {error:.3f} m")

def main():
    rclpy.init()
    rclpy.spin(ErrorMonitor())
```

---

## 🔬 Lab Exercise: "The Shake Test"

### 1. Lab Objectives
- **Setup:** Real Realsense D435i connected to Laptop.
- **Run:** `vins_fusion`.
- **Action:** Move the camera smoothly.
- **Observe:** Rviz path follows correctly.
- **Action:** Cover the lens (Blind).
- **Observe:** The path continues for ~1-2 seconds (IMU Integration) then drifts away rapidly (No visual correction).
- **Action:** Uncover.
- **Observe:** If drift was small, it recovers. If large, tracking fails (Optimization blows up).

---

## 🚀 Project: "The Loop Closure"

**Goal:** Correct drift after a long flight.
1.  **Fly:** A large circle (start at A, fly 50m, return to A).
2.  **Drift:** When returning to A, VINS might think you are at A' (offset by 1m).
3.  **DBoW (Bag of Words):** VINS Loop Fusion detects the image at A matches the start image.
4.  **Snap:** It calculates the error $(A - A')$ and distributes the correction back across the entire path (Pose Graph Optimization).
5.  **Result:** The map "Snaps" shut.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Initialization Fails"
*   **Cause:** Not moving enough / Moving too much.
*   **Fix:** Slide the camera sideways repeatedly to excite the accelerometers.

#### 2. "Scale is wrong (Tiny World)"
*   **Cause:** IMU noise parameters in Config are too low. Estimator trusts IMU too much or too little.
*   **Fix:** Tune `acc_n` and `gyr_n` based on Allen Variance datasheets.

---

## ⚡ Optimization: Optical Flow vs Feature Matching

*   **VINS-Mono:** Uses KLT Optical Flow (Tracking points frame-to-frame). Fast. Good for smooth video.
*   **ORB-SLAM3:** Uses ORB Descriptors. Slower. Good for large loop closures and "Kidnapped Robot" problem (Recovery after total loss).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can VIO work in the dark?
    *   **A:** No. (Unless it is Thermal VIO). IMU integration drifts cubically with time without visual corrections.
2.  **Q:** What is the Extrinsic Calibration?
    *   **A:** The precise Rotation/Translation between the Camera Center and the IMU Center. If off by 1cm or 1 degree, the math fails.
3.  **Q:** Output rate?
    *   **A:** IMU rate (200Hz) or Camera rate (30Hz)? VINS outputs pose at IMU rate (High freq) using propagation.

### Challenge Task
> **Task:** CPU Load.
> 1. Run VINS on Raspberry Pi 4.
> 2. Check `htop`.
> 3. Does it keep up? If not, reduce `max_cnt` (features) from 150 to 80.

---

## 📚 Further Reading
- **HKUST Aerial Robotics:** "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator".
- **OpenVINS:** Another excellent open-source solver.

---

**Day 122 Complete**
