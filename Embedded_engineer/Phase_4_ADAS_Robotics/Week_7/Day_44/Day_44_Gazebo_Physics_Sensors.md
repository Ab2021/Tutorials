# Day 44: Gazebo Simulation (Physics & Sensors)
## Phase 4: ADAS & Robotics Systems | Week 7: Simulation & Modeling

---

> **📝 Day 44 Focus:**
> Rviz shows you what the robot *thinks* is happening. Gazebo shows you what *actually* happens (Physics). Today, we take our static URDF model and bring it to life with gravity, friction, motors, and laser scanners using **Gazebo Plugins**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Distinguish** between Visual (Rviz) and Collision/Inertial (Gazebo) geometries.
2.  **Add** Gazebo-specific tags (`<gazebo>`) to URDF for colors and friction.
3.  **Implement** a Differential Drive Plugin to control the robot with `cmd_vel`.
4.  **Simulate** a Lidar sensor using the `ray` sensor plugin.
5.  **Spawn** the robot into a custom Gazebo world with obstacles.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 43:** URDF/Xacro.
-   **Physics:** Mass, Inertia, Friction coefficients ($\mu$).

### Hardware Requirements
-   **GPU:** Recommended for 3D rendering.

### Software Stack
-   **ROS 2:** `gazebo_ros_pkgs`.
-   **Simulator:** Gazebo Classic (v11).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Physics Engine

Gazebo uses **ODE (Open Dynamics Engine)** by default.
It calculates:
-   **Rigid Body Dynamics:** $F = ma$.
-   **Collision Detection:** Do two shapes overlap?
-   **Constraints:** Joints (Hinges, Sliders).

**Critical URDF Tags for Gazebo:**
1.  **`<inertial>`:** MUST be present and valid. If Mass = 0, the link is ignored or static.
2.  **`<collision>`:** Defines the physical shape.
3.  **`<gazebo reference="link_name">`:**
    -   `<material>`: Gazebo color (e.g., `Gazebo/Blue`).
    -   `<mu1>, <mu2>`: Friction coefficients.

### 🔹 Part 2: Plugins

Gazebo is a standalone program. ROS 2 talks to it via **Plugins**.
Plugins are C++ libraries loaded by Gazebo at runtime.

#### 2.1 Model Plugins
Control the robot's joints.
-   **Differential Drive:** Subscribes to `/cmd_vel`, calculates wheel speeds, applies torque.
-   **Joint State Publisher:** Publishes ground truth joint angles.

#### 2.2 Sensor Plugins
Simulate sensors.
-   **Ray (Lidar):** Casts rays, detects intersections. Publishes `sensor_msgs/LaserScan`.
-   **Camera:** Renders the scene. Publishes `sensor_msgs/Image`.
-   **IMU:** Simulates accelerometer/gyroscope.

---

## 💻 Implementation: Gazebo-fying the RoboCar

We will modify `robocar.xacro` to add physics and plugins.

### 🛠️ Setup
Use the `week7_day43` package (or copy to `week7_day44`). We will assume we are editing the existing files.

```bash
cd ~/ros2_ws/src/week7_day43
mkdir worlds
touch worlds/obstacle.world
```

### 👨‍💻 Code: robocar.xacro (Updated)

Add these sections to your existing Xacro file.

```xml
    <!-- ... Existing URDF ... -->

    <!-- 1. Gazebo Colors & Friction -->
    <gazebo reference="chassis">
        <material>Gazebo/Blue</material>
    </gazebo>

    <gazebo reference="front_left_wheel">
        <material>Gazebo/Black</material>
        <mu1>1.0</mu1>
        <mu2>1.0</mu2>
    </gazebo>
    <!-- Repeat for other wheels... -->

    <!-- 2. Differential Drive Plugin -->
    <gazebo>
        <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
            <!-- Wheel Info -->
            <left_joint>front_left_wheel_joint</left_joint>
            <right_joint>front_right_wheel_joint</right_joint>
            <wheel_separation>0.35</wheel_separation>
            <wheel_diameter>0.2</wheel_diameter>

            <!-- Limits -->
            <max_wheel_torque>200</max_wheel_torque>
            <max_wheel_acceleration>10.0</max_wheel_acceleration>

            <!-- Output -->
            <odometry_frame>odom</odometry_frame>
            <robot_base_frame>base_link</robot_base_frame>
            <publish_odom>true</publish_odom>
            <publish_odom_tf>true</publish_odom_tf>
            <publish_wheel_tf>true</publish_wheel_tf>
        </plugin>
    </gazebo>

    <!-- 3. Lidar Sensor Plugin -->
    <gazebo reference="lidar_link">
        <sensor name="lidar" type="ray">
            <pose>0 0 0 0 0 0</pose>
            <always_on>true</always_on>
            <visualize>true</visualize>
            <update_rate>10</update_rate>
            <ray>
                <scan>
                    <horizontal>
                        <samples>360</samples>
                        <resolution>1</resolution>
                        <min_angle>-3.14</min_angle>
                        <max_angle>3.14</max_angle>
                    </horizontal>
                </scan>
                <range>
                    <min>0.3</min>
                    <max>12</max>
                </range>
            </ray>
            <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
                <ros>
                    <argument>~/out:=scan</argument>
                </ros>
                <output_type>sensor_msgs/LaserScan</output_type>
                <frame_name>lidar_link</frame_name>
            </plugin>
        </sensor>
    </gazebo>
```

*Note: For a 4-wheeled car, standard diff drive plugins usually control 2 wheels. We can set the rear wheels to be "casters" (frictionless) or link them to the front wheels. For simplicity, let's assume we control the front wheels and the rear wheels just roll (continuous joints).*

### 👨‍💻 Code: obstacle.world

```xml
<?xml version="1.0" ?>
<sdf version="1.5">
  <world name="default">
    <!-- Sun -->
    <include>
      <uri>model://sun</uri>
    </include>
    <!-- Ground Plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Obstacles -->
    <model name="box">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### 👨‍💻 Code: gazebo.launch.py

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_path = get_package_share_directory('week7_day43') # Reusing package
    
    # 1. Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        launch_arguments={'world': os.path.join(pkg_path, 'worlds', 'obstacle.world')}.items()
    )

    # 2. Spawn Robot
    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'robocar'],
                        output='screen')

    # 3. Robot State Publisher (Same as before)
    # ... (Include display.launch.py logic or copy here)

    return LaunchDescription([
        gazebo,
        spawn_entity,
        # ... RSP node
    ])
```

### 🛠️ Build & Run

```bash
cd ~/ros2_ws
colcon build --packages-select week7_day43
source install/setup.bash
ros2 launch week7_day43 gazebo.launch.py
```

---

## 🔬 Lab Exercise: Driving & Scanning

### Lab Objectives
1.  **Launch:** You should see Gazebo open with a car and a box.
2.  **Drive:**
    -   Open a new terminal.
    -   `ros2 run teleop_twist_keyboard teleop_twist_keyboard`
    -   Drive the car. Watch it move in Gazebo.
3.  **Visualize:**
    -   Open Rviz2.
    -   Add "LaserScan" display. Topic: `/scan`.
    -   *Observation:* You should see red dots appearing on the box obstacle.
4.  **Physics Check:**
    -   Drive into the box.
    -   *Result:* The car should bounce off (Collision). If it passes through, your `<collision>` tags are wrong.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Robot Falls Through Ground
**Symptom:** Robot spawns and immediately falls into the abyss.
**Cause:**
-   No `<collision>` tags on wheels/chassis.
-   Spawn height (z) is too low (inside ground).
**Solution:** Spawn at z=0.5. Check collision geometry.

#### 2. Wheels Don't Turn
**Symptom:** `cmd_vel` is sent, but robot doesn't move.
**Cause:**
-   Inertia too high (Robot too heavy).
-   Torque too low in plugin.
-   Friction too high.
**Solution:** Check `<inertial>` values. Increase `<max_wheel_torque>`.

#### 3. Lidar Rays Not Visible
**Symptom:** No blue rays in Gazebo.
**Cause:** `<visualize>false</visualize>` in sensor plugin.
**Solution:** Set to true. (Note: This is just for debugging, doesn't affect `/scan` topic).

---

## ⚡ Optimization & Best Practices

### 1. Simplified Collision
Visual mesh can be high-poly (10k triangles).
Collision mesh MUST be simple (Box/Cylinder/Sphere).
-   **Why?** Physics calculation is $O(N^2)$ with triangles.
-   **Tip:** Use `<geometry><cylinder .../></geometry>` for wheels, not the STL.

### 2. Update Rate
Don't simulate Lidar at 100Hz if real hardware is 10Hz.
-   Simulating rays is CPU intensive.
-   Set `<update_rate>` to match real hardware.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between `<visual>` and `<collision>`?
    *   **A:** Visual is for the GPU (Rendering). Collision is for the CPU (Physics).
2.  **Q:** Why do we need the `robot_state_publisher` if Gazebo is running?
    *   **A:** Gazebo simulates the physics, but ROS needs the TF tree. The RSP reads the joint states from Gazebo and publishes the TFs for Rviz/SLAM.
3.  **Q:** What does the `libgazebo_ros_diff_drive.so` plugin do?
    *   **A:** It listens to `/cmd_vel` (Twist), calculates the required angular velocity for left/right wheels, and applies torque to the joints in the physics engine.

### Challenge Task
**Task:** Add a Camera.
1.  Add a camera link and joint.
2.  Add the `libgazebo_ros_camera.so` plugin.
3.  Launch and view the feed in Rviz2 (`/camera/image_raw`).
4.  Drive around and see the world from the robot's perspective.

---

## 📚 Further Reading & References
-   [Gazebo ROS 2 Plugins](https://github.com/ros-simulation/gazebo_ros_pkgs/wiki)
-   [SDF Specification](http://sdformat.org/spec)

---

**Day 44 Complete** | Phase 4: ADAS & Robotics Systems | Week 7: Simulation & Modeling
