# Day 85: Humanoid Robot Platforms
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 13: Humanoid Robotics

---

> **📝 Content Creator Instructions:**
> Humanoids are the F1 cars of robotics. Maximum complexity.
> - **Focus:** Overview of Platforms (Reachy, Poppy, Unitree H1, Tesla Optimus clones), URDF structures for Humanoids, and setting up a Humanoid Simulation in Gazebo/Isaac Sim.
> - **Code:** A unified `humanoid_description` package with `xacro` macros for Legs, Arms, and Head, spawnable in Gazebo.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Survey** the landscape of modern humanoid platforms (Open source vs Commercial).
2.  **Structure** a Humanoid URDF using standard conventions (`base_link` at pelvis vs foot).
3.  **Simulate** a floating-base robot in Gazebo (Handling the "Falling" problem before control).
4.  **Integrate** High-Level interfaces (Hugging Face Reachy SDK).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- High-end GPU (RTX 3070+) for Physics Simulation (Isaac Sim preferred, Gazebo works).

### Software Environment
```bash
sudo apt install ros-humble-gazebo-ros-pkgs
pip install reachy-sdk # Example for Reachy
```

### Prior Knowledge
- URDF (Week 2).
- ROS 2 Control (Week 4).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Humanoid Resurgence

Why now?
*   **AI:** Foundation models allow general-purpose learning.
*   **Hardware:** Harmonic drives and high-torque density actuators (Quasi-Direct Drive).
*   **Platforms:**
    *   **Unitree H1/G1:** Affordable ($16k), RL-ready.
    *   **Tesla Optimus:** Custom actuators, Vision-only.
    *   **Figure 01:** End-to-end Neural Networks (OpenAI).
    *   **Open Source:** Poppy (3D printed), Reachy (Upper body), OpenHA.

### 🔹 Part 2: Kinematic Structure

*   **Floating Base:** Unlike an Arm (fixed base), a Humanoid root (`pelvis`) is unactuated and floats in space. It has 6 DOF (XYZ+RPY) relative to the world.
*   **DoF Count:** Typical 20-30 DoF.
    *   Legs: 6 DoF x 2 (Hip Yaw/Roll/Pitch, Knee, Ankle Pitch/Roll).
    *   Arms: 7 DoF x 2.
    *   Torso/Head: 2-3 DoF.
*   **The Problem:** The robot falls under gravity. The "Control" must actively maintain the "Floating Base" pose via ground reaction forces.

### 🔹 Part 3: Simulation Challenges

*   **Contact Dynamics:** Feet sliding vs sticking. Hard contacts cause "jitter".
*   **Actuator Models:** Simulating a "Perfect Motor" is bad. Humanoids limited by Torque Limits and Joint Friction.
*   **Isaac Sim vs Gazebo:**
    *   *Gazebo (ODE):* Good for high-level logic. Hard for contact-rich walking.
    *   *Isaac Sim (PhysX):* GPU-accelerated. Standard for RL training (Isaac Gym).

---

## 💻 Implementation: Humanoid URDF

We will build a generic 12-DOF Lower Body + Torso URDF.

### 🛠️ Project Structure
```text
day85_humanoid/
├── urdf/
│   ├── humanoid.xacro
│   ├── leg.xacro
│   └── transmission.xacro
├── launch/
│   └── view_robot.launch.py
├── config/
│   └── joint_names.yaml
└── meshes/
    └── (Simple cylinders for now)
```

### 👨‍💻 Leg Macro (`urdf/leg.xacro`)

Standard 6-DOF Leg design.

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

  <xacro:macro name="humanoid_leg" params="prefix side reflect">
    <!-- Hip Yaw -->
    <link name="${prefix}_hip_yaw_link">
      <visual>
        <geometry><cylinder radius="0.05" length="0.1"/></geometry>
        <material name="grey"/>
      </visual>
      <collision><geometry><cylinder radius="0.05" length="0.1"/></geometry></collision>
      <inertial><mass value="1.0"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    </link>

    <joint name="${prefix}_hip_yaw" type="revolute">
      <parent link="pelvis"/>
      <child link="${prefix}_hip_yaw_link"/>
      <origin xyz="0 ${reflect * 0.1} -0.05" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="-1.5" upper="1.5" effort="100" velocity="10"/>
    </joint>

    <!-- Hip Roll, Pitch, Knee, Ankle Pitch, Ankle Roll ... -->
    <!-- Simplified for brevity: Usually Hip R/P/Y intersect at one point -->

    <!-- Knee -->
    <link name="${prefix}_shin">
       <visual>
         <origin xyz="0 0 -0.2"/>
         <geometry><box size="0.05 0.05 0.4"/></geometry>
       </visual>
       <collision>
         <origin xyz="0 0 -0.2"/>
         <geometry><box size="0.05 0.05 0.4"/></geometry>
       </collision>
       <inertial><mass value="2.0"/><inertia ixx="0.1" iyy="0.1" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    </link>
    
    <!-- Foot -->
    <link name="${prefix}_foot">
       <visual><geometry><box size="0.2 0.1 0.05"/></geometry></visual>
       <collision>
          <!-- Friction parameters crucial for walking! -->
          <surface>
            <friction>
              <ode><mu>1.0</mu><mu2>1.0</mu2></ode>
            </friction>
          </surface>
          <geometry><box size="0.2 0.1 0.05"/></geometry>
       </collision>
       <inertial><mass value="0.5"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    </link>

  </xacro:macro>
</robot>
```

### 👨‍💻 Main Xacro (`urdf/humanoid.xacro`)

```xml
<?xml version="1.0"?>
<robot name="atlas_lite" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <xacro:include filename="$(find day85_humanoid)/urdf/leg.xacro"/>

  <!-- Floating Base Root -->
  <link name="base_link"></link>

  <link name="pelvis">
    <visual><geometry><box size="0.2 0.3 0.15"/></geometry></visual>
    <inertial><mass value="10.0"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>

  <joint name="base_to_pelvis" type="fixed">
    <parent link="base_link"/>
    <child link="pelvis"/>
    <origin xyz="0 0 0.9" rpy="0 0 0"/> <!-- Start 90cm up -->
  </joint>

  <!-- Left Leg -->
  <xacro:humanoid_leg prefix="l" side="left" reflect="1"/>
  <!-- Right Leg -->
  <xacro:humanoid_leg prefix="r" side="right" reflect="-1"/>
  
  <!-- Gazebo Control Plugin -->
  <gazebo>
    <plugin name="gazebo_ros2_control" filename="libgazebo_ros2_control.so">
      <parameters>$(find day85_humanoid)/config/controllers.yaml</parameters>
    </plugin>
  </gazebo>

</robot>
```

### 👨‍💻 Pin the Robot (`launch/view_humanoid.launch.py`)

For Day 85, we don't have a walking controller yet. If we spawn it, it collapses.
*   **Technique:** Use a `fixed` joint to the world momentarily, or "Hang" it from a gantry in Sim.
*   **Or:** Spawn in `paused` state.

```python
# Launch file spawning robot in Gazebo
# ...
    Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'humanoid', '-z', '1.0'],
        output='screen'
    )
```

---

## 🔬 Lab Exercise: "The Ragdoll"

### 1. Lab Objectives
- Spawn the Humanoid in Gazebo.
- **Action:** Unpause physics.
- **Observation:** The robot falls (Ragdoll physics).
- **Task:** 
    1.  Add `p_gain` (Stiffness) to the joints via `ros2_control` (set `position` command to 0).
    2.  Even with stiff joints, the *whole robot* falls over like a statue.
    3.  This proves that "Joint Control" $\neq$ "Balance Control".

---

## 🚀 Project: "Reachy Twin"

**Goal:** Simulate `Reachy` (Upper Body Humanoid).
1.  **Clone:** `github.com/pollen-robotics/reachy_ros2`.
2.  **Launch:** Running the simulation with mock hardware.
3.  **Interact:**
    *   Move head (Antennas).
    *   Move Arm.
    *   Reachy is fixed-base (Torso mounted), so it doesn't fall.
    *   Good starting point for manipulation tasks on humanoids.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Exploding Simulation"
*   **Symptom:** Robot flies into space.
*   **Cause:** Interpenetrating colliders at the hip joints.
*   **Fix:** Check URDF `origin` tags. Visual can overlap, Collision cannot. Use `collision` tags with slightly smaller geometry.

#### 2. "Floating Feet"
*   **Symptom:** Feet hover 1cm above ground.
*   **Cause:** Collision box is larger than visual? Or Gazebo contact padding.
*   **Fix:** Visualize Collision shapes in Gazebo.

---

## ⚡ Optimization: MJCF (MuJoCo Format)

URDF is great for ROS, but MuJoCo (MJCF) is better for RL.
*   **Why?** Stable contacts (Soft contacts), faster solver.
*   **Tool:** `urdf2mjcf` converter.
*   **Workflow:** Model in URDF $\to$ Convert to MJCF for Training $\to$ Deploy policy via ROS 2.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the "Floating Base" frame usually called?
    *   **A:** `pelvis` or `torso`. `base_link` often maps to this. World-to-Base transform is provided by Odometry (Perfect in Sim, VIO/LIO in Real).
2.  **Q:** Why are humanoid legs usually 6 DOF?
    *   **A:** To control 6-DOF of the foot relative to pelvis (XYZ + RPY). Allows placing the foot flat on uneven terrain.
3.  **Q:** What is "Underactuated"?
    *   **A:** The interaction between feet and ground is not a "joint". We cannot directly command "Move Torso Forward". We must exert force on the ground to push the torso.

### Challenge Task
> **Task:** Dual Arm Setup.
> 1. Add two 4-DOF arms to the URDF.
> 2. Check Mass distribution.
> 3. If arms are heavy and extended forward, the CoM moves forward. The robot (if not pinned) will tip over face-first.

---

## 📚 Further Reading
- **IEEE Spectrum:** "History of Humanoid Robots".
- **Drake (MIT):** Simulation for Humanoids.

---

**Day 85 Complete**
