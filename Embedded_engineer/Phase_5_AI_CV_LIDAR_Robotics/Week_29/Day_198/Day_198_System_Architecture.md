# Day 198: System Architecture Design
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 29: Capstone Project Part 1

---

> **📝 Content Creator Instructions:**
> If it works in Sim, it *might* work in Real Life.
> - **Focus:** Creating the Gazebo Simulation Environment. Integrating the Mobile Manipulator (Husky + UR5).
> - **Code:** `farm.world`, `agribot_urdf.xacro`.
> - **Concept:** Simulation-based Verification.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Compose** a complex XACRO file combining a mobile base and a manipulator.
2.  **Design** a Gazebo world with crop visuals (Repeated models).
3.  **Config** the `ros2_control` parameters for the combined system.
4.  **Launch** the complete simulation with `ros2 launch`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU recommended for Gazebo.

### Software Environment
```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-ur-description ros-humble-husky-description
```

### Prior Knowledge
- URDF/XACRO (Day 11).
- Gazebo (Day 15).

---

## 📖 Theoretical Design Pattern

### 🔹 The Mobile Manipulator
Combining a Base and an Arm creates a redundant system (7+ DoF).
*   **TF Tree:** `odom` $\to$ `base_link` $\to$ `ur5_base_link` $\to$ `ee_link`.
*   **Control:** Usually split.
    *   `diff_drive_controller` handles wheels.
    *   `joint_trajectory_controller` handles arm.

### 🔹 The Farm Environment
We need "Unstructured" structure.
*   **Rows:** Parallel lines 1m apart.
*   **Terrain:** Uneven ground (Simulated with heightmap).
*   **Lighting:** Sunlight (Directional Light) varies by time of day.

---

## 💻 Implementation: The Simulation

### 🛠️ Project Structure
```text
agribot_description/
├── urdf/
│   ├── agribot.urdf.xacro
│   └── materials.xacro
├── worlds/
│   └── farm.world
└── launch/
    └── gazebo.launch.py
```

### 👨‍💻 Robot Description (`urdf/agribot.urdf.xacro`)

We include the standard Husky and UR5 descriptions and weld them together.

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="agribot">

  <!-- Include Husky -->
  <xacro:include filename="$(find husky_description)/urdf/husky.urdf.xacro" />

  <!-- Include UR5 -->
  <xacro:include filename="$(find ur_description)/urdf/ur5.urdf.xacro" />

  <!-- Add UR5 to Husky Top Plate -->
  <!-- The Husky has a 'top_plate_link' -->
  <joint name="arm_mount_joint" type="fixed">
    <parent link="top_plate_link" />
    <child link="base_link" /> <!-- UR5 Base -->
    <origin xyz="0.2 0 0.0" rpy="0 0 0" />
  </joint>

  <!-- Instantiate UR5 macro -->
  <xacro:ur5_robot prefix="" joint_limited="true"/>
  
  <!-- Add Sensors -->
  <!-- Realsense Camera on Wrist -->
  <link name="camera_link">
    <visual>
      <geometry><box size="0.05 0.05 0.05"/></geometry>
      <material name="black"/>
    </visual>
  </link>
  
  <joint name="camera_joint" type="fixed">
    <parent link="ee_link"/> <!-- End Effector -->
    <child link="camera_link"/>
    <origin xyz="0 0 0.05" rpy="0 -1.57 0"/> <!-- Pointing forward -->
  </joint>

  <!-- Gazebo Plugins -->
  <gazebo>
    <plugin name="gazebo_ros2_control" filename="libgazebo_ros2_control.so">
      <parameters>$(find agribot_control)/config/controllers.yaml</parameters>
    </plugin>
  </gazebo>

</robot>
```

### 👨‍💻 The Farm World (`worlds/farm.world`)

We create a population of "Bush" models.

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">
    <!-- Sun & Ground -->
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>

    <!-- Crop Rows -->
    <population name="row_1">
      <model name="bush_1">
        <include><uri>model://bush</uri></include> <!-- Assume model exists -->
      </model>
      <pose>0 1 0 0 0 0</pose>
      <box>
        <size>20 0.5 1</size> <!-- 20m long row -->
      </box>
      <model_count>20</model_count>
      <distribution>
        <type>linear-x</type>
      </distribution>
    </population>

    <population name="row_2">
      <model name="bush_2">
        <include><uri>model://bush</uri></include>
      </model>
      <pose>0 -1 0 0 0 0</pose>
      <box><size>20 0.5 1</size></box>
      <model_count>20</model_count>
      <distribution><type>linear-x</type></distribution>
    </population>
    
    <!-- Physics -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>
  </world>
</sdf>
```

---

## 🔬 Lab Exercise: "Spawn"

### 1. Lab Objectives
- **Launch:** Run `gazebo.launch.py`.
- **Verify:** Robot appears. Arm is attached to base.
- **Drive:** Use `teleop_twist_keyboard` to drive the Husky.
    *   Does the arm wobble? (Physics check).
    *   Does the arm collide with the wheels? (Self-collision check).
- **Control:** Use `ros2 topic pub` to send joint commands to the arm.

---

## 🚀 Project Steps

1.  **Create Package:** `ros2 pkg create --build-type ament_cmake agribot_description`.
2.  **Copy Meshes:** Ensure you have the `dae`/`stl` files for Husky and UR5 (from standard packages).
3.  **Controllers:** Create `config/controllers.yaml` defining `joint_state_broadcaster`, `diff_drive_controller`, and `joint_trajectory_controller`.
4.  **Launch:** Create the launch file that spawns the robot at `x=0, y=0`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Robot Explodes"
*   **Scene:** Robot flies into space upon spawning.
*   **Cause:** Collision meshes overlapping. The UR5 base is inside the Husky plate.
*   **Fix:** Adjust the `<origin z=...>` in the joint to raise the arm slightly.

#### 2. "Wheels Don't Turn"
*   **Cause:** Missing transmission tags in URDF or `gazebo_ros2_control` plugin missing.
*   **Fix:** Verify `hardware_interface` tags in the xacro.

---

## ⚡ Optimization: Simplified Collisions

Detailed meshes (STL) are slow for collision checking.
*   **Visual:** High res STL.
*   **Collision:** Simple Primitives (Cylinder, Box).
*   **Result:** Sim runs at 60 FPS instead of 5 FPS.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `<visual>` and `<collision>`?
    *   **A:** Visual is for the GPU (Rendering). Collision is for the Physics Engine (Contacts).
2.  **Q:** Why use `xacro`?
    *   **A:** To avoid copy-pasting XML. Allows reusing the UR5 macro inside the AgriBot file.

### Challenge Task
> **Task:** "Add Fruit".
> 1. Create a simple RED SPHERE model.
> 2. Attach it to the bushes in the SDF.
> 3. Give it a collision box so the gripper can touch it.

---

## 📚 Further Reading
- **Gazebo Tutorials:** "Building a World".
- **ROS 2 Control Demos:** "Mobile Manipulator".

---

**Day 198 Complete**
