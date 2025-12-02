# Day 46: Sensor Simulation (Lidar, Camera, IMU, GPS)
## Phase 4: ADAS & Robotics Systems | Week 7: Simulation & Modeling

---

> **📝 Day 46 Focus:**
> A simulation with perfect sensors is a video game, not a robotics tool. Real sensors are noisy, drift over time, and get blinded by the sun. Today, we make our Gazebo simulation **Realistic** by adding Cameras, IMUs, and GPS, and injecting **Gaussian Noise** to test the robustness of our algorithms.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Configure** Gazebo plugins for Camera, IMU, and GPS sensors.
2.  **Model** sensor noise using Gaussian distributions (Mean, StdDev).
3.  **Simulate** lens distortion (Pinhole model) for cameras.
4.  **Visualize** sensor data in Rviz2 (Point Clouds, Image Feeds, Odometry).
5.  **Analyze** the impact of noise on localization algorithms (e.g., drift).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 44:** Gazebo Plugins.
-   **Statistics:** Normal Distribution (Bell Curve).

### Hardware Requirements
-   **GPU:** Required for Camera simulation.

### Software Stack
-   **ROS 2:** `gazebo_ros_pkgs`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Noise Model

Perfect data ($x_{true}$) is useless for testing. We need measured data ($z$).
$$ z = x_{true} + \epsilon, \quad \epsilon \sim N(\mu, \sigma^2) $$

-   **Lidar Noise:** Distance error (~1-2cm).
-   **IMU Noise:**
    -   **White Noise:** Random jitter.
    -   **Bias:** Constant offset (drifts with temperature).
    -   **Random Walk:** Bias changes over time.
-   **GPS Noise:**
    -   **Position:** ~2-5m error.
    -   **Velocity:** ~0.1m/s error.

### 🔹 Part 2: Sensor Plugins

#### 2.1 Camera (`libgazebo_ros_camera.so`)
-   **Parameters:** Resolution ($W \times H$), Field of View (FOV), Frame Rate.
-   **Distortion:** $k_1, k_2, k_3$ (Radial), $p_1, p_2$ (Tangential).

#### 2.2 IMU (`libgazebo_ros_imu_sensor.so`)
-   **Outputs:** Orientation (Quaternion), Angular Velocity, Linear Acceleration.
-   **Critical:** Gravity vector ($9.81 m/s^2$) is included in acceleration.

#### 2.3 GPS (`libgazebo_ros_gps_sensor.so`)
-   **Outputs:** Latitude, Longitude, Altitude.
-   **Reference:** Needs a spherical coordinate reference (WGS84) defined in the world file.

---

## 💻 Implementation: The "Noisy" RoboCar

We will update `robocar.xacro` (from Day 44) to include these sensors.

### 🛠️ Setup
Use `week7_day46` (copy from `week7_day44`).

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake week7_day46
# Copy urdf/launch folders from previous day
```

### 👨‍💻 Code: robocar.xacro (Sensor Additions)

Add these sections to your Xacro file.

#### 1. Camera Link & Plugin

```xml
    <!-- Camera Link -->
    <link name="camera_link">
        <visual>
            <geometry>
                <box size="0.05 0.05 0.05"/>
            </geometry>
            <material name="black"/>
        </visual>
    </link>

    <joint name="camera_joint" type="fixed">
        <parent link="chassis"/>
        <child link="camera_link"/>
        <origin xyz="0.25 0 0.08" rpy="0 0 0"/>
    </joint>

    <!-- Optical Frame (Standard ROS coordinate system: Z forward) -->
    <link name="camera_link_optical"/>
    <joint name="camera_optical_joint" type="fixed">
        <parent link="camera_link"/>
        <child link="camera_link_optical"/>
        <origin xyz="0 0 0" rpy="${-pi/2} 0 ${-pi/2}"/>
    </joint>

    <!-- Camera Plugin -->
    <gazebo reference="camera_link">
        <sensor name="camera" type="camera">
            <pose>0 0 0 0 0 0</pose>
            <visualize>true</visualize>
            <update_rate>30</update_rate>
            <camera>
                <horizontal_fov>1.089</horizontal_fov>
                <image>
                    <format>R8G8B8</format>
                    <width>640</width>
                    <height>480</height>
                </image>
                <clip>
                    <near>0.05</near>
                    <far>8.0</far>
                </clip>
                <noise>
                    <type>gaussian</type>
                    <mean>0.0</mean>
                    <stddev>0.007</stddev> <!-- Image Grain -->
                </noise>
            </camera>
            <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
                <frame_name>camera_link_optical</frame_name>
            </plugin>
        </sensor>
    </gazebo>
```

#### 2. IMU Link & Plugin

```xml
    <!-- IMU Link -->
    <link name="imu_link"/>
    <joint name="imu_joint" type="fixed">
        <parent link="chassis"/>
        <child link="imu_link"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
    </joint>

    <!-- IMU Plugin -->
    <gazebo reference="imu_link">
        <gravity>true</gravity>
        <sensor name="imu_sensor" type="imu">
            <always_on>true</always_on>
            <update_rate>100</update_rate>
            <visualize>true</visualize>
            <topic>__default_topic__</topic>
            <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_plugin">
                <topicName>imu</topicName>
                <bodyName>imu_link</bodyName>
                <updateRateHZ>10.0</updateRateHZ>
                <gaussianNoise>0.0</gaussianNoise>
                <xyzOffset>0 0 0</xyzOffset>
                <rpyOffset>0 0 0</rpyOffset>
                <frameName>imu_link</frameName>
                <initialOrientationAsReference>false</initialOrientationAsReference>
            </plugin>
            <pose>0 0 0 0 0 0</pose>
        </sensor>
    </gazebo>
```

#### 3. GPS Plugin

```xml
    <!-- GPS Plugin (Attached to Chassis) -->
    <gazebo reference="chassis">
        <sensor name="gps_sensor" type="gps">
            <always_on>true</always_on>
            <update_rate>1.0</update_rate>
            <plugin name="gps_controller" filename="libgazebo_ros_gps_sensor.so">
                <ros>
                    <remapping>~/out:=/gps/fix</remapping>
                </ros>
                <frame_name>chassis</frame_name>
            </plugin>
        </sensor>
    </gazebo>
```

### 👨‍💻 Code: launch_sensors.launch.py

Standard launch file (same as Day 44).

### 🛠️ Build & Run

```bash
cd ~/ros2_ws
colcon build --packages-select week7_day46
source install/setup.bash
ros2 launch week7_day46 launch_sensors.launch.py
```

---

## 🔬 Lab Exercise: Sensor Verification

### Lab Objectives
1.  **Launch:** Open Gazebo and Rviz2.
2.  **Camera:**
    -   Rviz2: Add "Image" display. Topic: `/camera/image_raw`.
    -   *Observation:* You see the Gazebo world. Notice the graininess (Noise).
3.  **IMU:**
    -   Terminal: `ros2 topic echo /imu/data`.
    -   *Observation:* `linear_acceleration.z` should be approx `9.8`. Orientation should be stable.
4.  **GPS:**
    -   Terminal: `ros2 topic echo /gps/fix`.
    -   *Observation:* Lat/Lon values. Move the robot (Teleop) and watch them change.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Camera Image is Black
**Symptom:** `/camera/image_raw` is publishing, but all pixels are 0.
**Cause:**
-   Lighting is too dim.
-   Clipping plane (`<near>`) is too far.
-   Camera is inside the chassis (blocked).
**Solution:** Move camera joint `xyz` forward. Check `<near>0.05</near>`.

#### 2. IMU Drifts Wildly
**Symptom:** Robot is stationary, but IMU reports rotation.
**Cause:** Bias in the plugin settings (if configured) or simulation instability.
**Solution:** Check `<initialOrientationAsReference>`. Usually set to `false` for absolute orientation.

#### 3. GPS is (0,0)
**Symptom:** Lat/Lon is always 0.
**Cause:** World file missing `<spherical_coordinates>`.
**Solution:** Add this to your `.world` file:
    ```xml
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <latitude_deg>37.4</latitude_deg>
      <longitude_deg>-122.1</longitude_deg>
      <elevation>0.0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>
    ```

---

## ⚡ Optimization & Best Practices

### 1. Optical Frames
Cameras in ROS use: X=Right, Y=Down, Z=Forward.
Cameras in Gazebo/URDF use: X=Forward, Y=Left, Z=Up.
**Always** add an `_optical` link rotated by `rpy="-pi/2 0 -pi/2"` to align them. Otherwise, your point clouds will look sideways.

### 2. Depth Cameras
To simulate a Kinect/RealSense:
-   Use `type="depth"`.
-   Plugin: `libgazebo_ros_camera.so` (it handles both RGB and Depth).
-   Output: `/camera/depth/image_raw` (Float32 image, value = distance in meters).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is Gaussian Noise?
    *   **A:** Random noise added to sensor readings, following a Normal Distribution (Bell Curve). It simulates real-world sensor imperfections.
2.  **Q:** Why does the IMU measure gravity?
    *   **A:** An accelerometer measures *proper acceleration*. Sitting on a table, it feels 1g upwards (resisting gravity). In freefall, it measures 0g.
3.  **Q:** What is the difference between `sensor_msgs/Image` and `sensor_msgs/CompressedImage`?
    *   **A:** `Image` is raw (huge bandwidth). `CompressedImage` is JPEG/PNG (small bandwidth). Gazebo publishes Raw.

### Challenge Task
**Task:** Noisy Odometry.
1.  The `diff_drive` plugin publishes perfect Odometry.
2.  Write a Python node `noisy_odom.py`.
3.  Subscribe to `/odom`.
4.  Add Gaussian noise to `pose.position.x` and `y`.
5.  Publish to `/noisy_odom`.
6.  Visualize both in Rviz (Set decay time to 10s to see trails).

---

## 📚 Further Reading & References
-   [Gazebo Sensor Noise Model](http://gazebosim.org/tutorials?tut=sensor_noise)
-   [ROS 2 Camera Plugin](https://github.com/ros-simulation/gazebo_ros_pkgs/wiki/ROS-2-Migration:-Camera)

---

**Day 46 Complete** | Phase 4: ADAS & Robotics Systems | Week 7: Simulation & Modeling
