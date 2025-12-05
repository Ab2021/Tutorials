# Day 59: Sensor Simulation (Plugins)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 9: Simulation & Sim-to-Real

---

> **📝 Content Creator Instructions:**
> A blind robot is boring. Let's give it eyes.
> - **Focus:** Gazebo Sensor Plugins (Lidar, Camera, IMU, Depth), Noise Models (Gaussian), and GPU Acceleration.
> - **Code:** Adding a `libgazebo_ros_velodyne_laser.so` plugin to our URDF.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Attach** Gazebo plugins to URDF links to simulate Camera and Lidar.
2.  **Configure** sensor parameters (FOV, Resolution, Update Rate, Range).
3.  **Inject** Sensor Noise models (Gaussian Drift) to simulate reality.
4.  **Visualize** sensor data in Rviz2 (Image, PointCloud2, LaserScan).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
sudo apt install ros-humble-gazebo-plugins
```

### Prior Knowledge
- Sensor Drivers (Week 1).
- URDF/Xacro (Day 57).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: How Plugins Work

Gazebo logic is modular. Plugins are C++ shared libraries (`.so`) loaded at runtime.
*   **System Plugin:** Controls the whole world (e.g., Pedestrian Actor).
*   **Model Plugin:** Controls a robot (e.g., Diff Drive).
*   **Sensor Plugin:** Generates data (Topic Publisher).

### 🔹 Part 2: Ray vs GPU Sensors

1.  **CPU Ray Sensor:** Ray-tracing on CPU. Accurate. Slow for High-Res Lidar (Velodyne-64).
2.  **GPU Ray Sensor:** Uses OpenGL shaders to render depth buffer. Super fast.
    *   *Requirement:* Headless mode usually disables GPU rendering unless configured correctly (`rendering` plugin enabled).

### 🔹 Part 3: The Noise Model

Perfect data kills algorithms.
*   If we train a SLAM on perfect Data, it fails on Real Robot.
*   **Gaussian Noise:** Add `stddev` to range measurements.
*   **Bias:** Add constant offset (Calibration error).
*   **Dropouts:** Randomly return `inf` (Glass reflection).

---

## 💻 Implementation: Adding Sensors

We attach sensors to `lidar_link` and `camera_link`.

### 🛠️ Project Structure
```text
day59_sensors/
├── urdf/
│   ├── sensors.xacro
│   └── robot_with_sensors.urdf.xacro
└── launch/
    └── sim_sensors.launch.py
```

### 👨‍💻 Sensor Macros (`urdf/sensors.xacro`)

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

    <!-- LIDAR (Ray Sensor) -->
    <xacro:macro name="lidar_sensor" params="link_name">
        <gazebo reference="${link_name}">
            <sensor name="lidar" type="ray">
                <pose>0 0 0 0 0 0</pose>
                <visualize>true</visualize>
                <update_rate>10</update_rate>
                <ray>
                    <scan>
                        <horizontal>
                            <samples>360</samples>
                            <min_angle>-3.14</min_angle>
                            <max_angle>3.14</max_angle>
                        </horizontal>
                    </scan>
                    <range>
                        <min>0.10</min>
                        <max>30.0</max>
                    </range>
                    <noise>
                        <type>gaussian</type>
                        <mean>0.0</mean>
                        <stddev>0.01</stddev> <!-- 1cm noise -->
                    </noise>
                </ray>
                <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
                    <ros>
                        <remapping>~/out:=scan</remapping>
                    </ros>
                    <output_type>sensor_msgs/LaserScan</output_type>
                </plugin>
            </sensor>
        </gazebo>
    </xacro:macro>

    <!-- CAMERA -->
    <xacro:macro name="camera_sensor" params="link_name">
        <gazebo reference="${link_name}">
            <sensor name="camera" type="camera">
                <update_rate>30.0</update_rate>
                <camera name="head">
                    <horizontal_fov>1.3962634</horizontal_fov>
                    <image>
                        <width>800</width>
                        <height>800</height>
                        <format>R8G8B8</format>
                    </image>
                    <clip>
                        <near>0.02</near>
                        <far>300</far>
                    </clip>
                    <noise>
                        <type>gaussian</type>
                        <mean>0.0</mean>
                        <stddev>0.007</stddev>
                    </noise>
                </camera>
                <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
                    <alwaysOn>true</alwaysOn>
                    <cameraName>robot/camera1</cameraName>
                    <imageTopicName>image_raw</imageTopicName>
                    <cameraInfoTopicName>camera_info</cameraInfoTopicName>
                    <frameName>${link_name}_optical_frame</frameName>
                </plugin>
            </sensor>
        </gazebo>
    </xacro:macro>

</robot>
```

### 👨‍💻 Viewing Output

1.  Launch Simulation.
2.  Open Rviz2.
3.  Add `LaserScan` -> Topic `/scan`.
    *   You should see red dots flickering (Noise) hitting the walls.
4.  Add `Image` -> Topic `/robot/camera1/image_raw`.
    *   You should see the Gazebo world.

---

## 🔬 Lab Exercise: The "Phantom" Obstacle

### 1. Lab Objectives
- Lidar glass reflection simulation.
- **Task:** Create a box with `<visual>` transparency but `<collision>` solid.
- **Physics:** Robot crashes.
- **Lidar:** Passes through (if glass) or Reflects?
- **Gazebo limitation:** Standard rays obey collision mesh. To simulate glass passing, you need **Collision Bitmasks** (Advanced).
- **Goal:** Understand simulator limitations.

---

## 🚀 Project: "Depth Camera Simulation"

**Goal:** Simulate an Intel RealSense D435.
1.  **Plugin:** `libgazebo_ros_camera.so` type `depth`.
2.  **Output:** `sensor_msgs/PointCloud2`.
3.  **Visualization:** Since Rviz handles PC2 heavy rendering, check FPS.
4.  **Shadows:** Real depth cams have "Shadows" where IR projector is blocked. Sim cameras are perfect.
    *   **Sim-to-Real Gap:** Sim algorithms fail on Real shadows.
    *   **Fix:** Add "Post-Processing Dropout" plugin to randomly delete pixels based on neighboring gradient.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "No Data on Topic"
*   **Symptom:** `ros2 topic list` shows `/scan`, but `ros2 topic echo /scan` is empty.
*   **Cause:** Gazebo Simulation is paused. Or Plugin failed to load (Check console `[Err]`).
*   **Check:** Is `<is_static>true</is_static>`? Sensors on static (world-fixed) objects sometimes behave differently.

#### 2. "Wrong Frame ID"
*   **Symptom:** Rviz error "Transform [lidar_link] to [map] does not exist".
*   **Cause:** The `<frameName>` param in plugin must match the URDF link name.
*   **Fix:** Check `view_frames`.

---

## ⚡ Optimization: GPU Lidar

For 3D Lidar (Velodyne VLP-16):
*   CPU Ray: 16 rings * 1800 samples = 28,000 rays/frame @ 10Hz = 280k rays/sec. (CPU bottleneck).
*   GPU Ray: `type="gpu_ray"`. Plugin `libgazebo_ros_velodyne_gpu.so`.
*   Renders a $360 \times 16$ pixel depth image in 1ms using GPU. Converts to PointCloud.
*   Speedup: 10x.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why add Noise?
    *   **A:** To bridge the Sim-to-Real gap. A robust algorithm works with measurement uncertainty ($x \pm \sigma$).
2.  **Q:** Scan vs PointCloud?
    *   **A:** Scan is 2D (Array of ranges). PointCloud is 3D (Array of XYZ).
3.  **Q:** Does Camera Plugin simulate Motion Blur?
    *   **A:** No. Standard Gazebo renders perfect sharp frames. Need custom shaders for Motion Blur / Rolling Shutter.

### Challenge Task
> **Task:** IMU Drift.
> 1. Add `libgazebo_ros_imu_sensor.so`.
> 2. Set Bias to `0.05` rad/s.
> 3. Watch the Yaw angle drift over 10 minutes even when robot is still.
> 4. This forces you to use EKF (Day 10) to fuse Wheel Odometry.

---

## 📚 Further Reading
- **Gazebo Plugin Tutorial:** gazebosim.org.
- **Velodyne Simulator:** DataspeedInc GitHub.

---

**Day 59 Complete**
