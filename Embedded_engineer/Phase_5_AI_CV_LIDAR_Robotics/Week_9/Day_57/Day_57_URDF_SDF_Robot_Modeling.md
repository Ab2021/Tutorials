# Day 57: URDF & SDF (Robot Modeling)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 9: Simulation & Sim-to-Real

---

> **📝 Content Creator Instructions:**
> Before we simulate, we must describe.
> - **Focus:** URDF (XML), Xacro (Macros), SDF (Simulation Description Format), and Mesh Optimization (Collision vs Visual).
> - **Code:** Creating a mobile robot model (`my_robot.urdf`) from scratch.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Construct** a valid URDF for a differential drive robot with Lidar.
2.  **Use** Xacro to modularize repetitive parts (e.g., wheels).
3.  **Differentiate** between URDF (ROS Kinematics) and SDF (Gazebo Physics).
4.  **Visualize** the model in Rviz using `robot_state_publisher`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
sudo apt install ros-humble-xacro ros-humble-joint-state-publisher-gui
```

### Prior Knowledge
- TF2 Frames (Day 3).
- XML Syntax.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: URDF (Unified Robot Description Format)

XML file describing a robot's structure.
*   **Links:** Rigid bodies (Mass, Inertia, Visual, Collision).
*   **Joints:** Connections (Continuous, Revolute, Prismatic, Fixed).
*   **Transmission:** Actuators (Motor interface).

### 🔹 Part 2: Visual vs Collision

*   **Visual Geometry:** High-poly Mesh (`robot.dae`). Looks pretty. GPU calculated.
*   **Collision Geometry:** Low-poly Primitive (Cylinder/Box). Simple physics. CPU calculated.
*   *Trap:* Using a 10MB STL for collision will kill your physics engine (1 FPS).

### 🔹 Part 3: SDF (Simulation Description Format)

URDF is for ROS (Kinematics). SDF is for Gazebo (Physics).
*   Gazebo converts URDF $\to$ SDF automatically (mostly).
*   SDF allows: Closed loops, deformable objects, multiple robots in one file (World).

---

## 💻 Implementation: Building `my_bot`

We will build a simple 2-wheeled robot with a Lidar.

### 🛠️ Project Structure
```text
day57_urdf/
├── urdf/
│   ├── common_properties.xacro
│   ├── mobile_base.xacro
│   └── robot.urdf.xacro
├── meshes/
│   └── wheel.stl (optional)
├── launch/
│   └── display.launch.py
└── CMakeLists.txt
```

### 👨‍💻 Common Macros (`urdf/common_properties.xacro`)

Defining Inertia matrices.

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

    <xacro:macro name="box_inertia" params="m l w h">
        <inertial>
            <mass value="${m}"/>
            <inertia ixx="${(m/12) * (h*h + w*w)}" ixy="0" ixz="0"
                     iyy="${(m/12) * (l*l + h*h)}" iyz="0"
                     izz="${(m/12) * (l*l + w*w)}" />
        </inertial>
    </xacro:macro>

    <xacro:macro name="cylinder_inertia" params="m r h">
        <inertial>
            <mass value="${m}"/>
            <inertia ixx="${(m/12) * (3*r*r + h*h)}" ixy="0" ixz="0"
                     iyy="${(m/12) * (3*r*r + h*h)}" iyz="0"
                     izz="${(m/2) * (r*r)}" />
        </inertial>
    </xacro:macro>

</robot>
```

### 👨‍💻 Robot Description (`urdf/robot.urdf.xacro`)

```xml
<?xml version="1.0"?>
<robot name="my_bot" xmlns:xacro="http://www.ros.org/wiki/xacro">

    <xacro:include filename="common_properties.xacro"/>

    <!-- Base Link -->
    <link name="base_link">
        <visual>
            <geometry>
                <box size="0.5 0.3 0.1"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <geometry>
                <box size="0.5 0.3 0.1"/>
            </geometry>
        </collision>
        <xacro:box_inertia m="5.0" l="0.5" w="0.3" h="0.1"/>
    </link>

    <!-- Wheels Macro -->
    <xacro:macro name="wheel" params="prefix y_offset">
        <link name="${prefix}_wheel">
            <visual>
                <geometry>
                    <cylinder radius="0.05" length="0.04"/>
                </geometry>
                <material name="black">
                    <color rgba="0 0 0 1"/>
                </material>
            </visual>
            <collision>
                <geometry>
                    <cylinder radius="0.05" length="0.04"/>
                </geometry>
            </collision>
            <xacro:cylinder_inertia m="1.0" r="0.05" h="0.04"/>
        </link>

        <joint name="${prefix}_wheel_joint" type="continuous">
            <parent link="base_link"/>
            <child link="${prefix}_wheel"/>
            <origin xyz="0 ${y_offset} 0" rpy="-1.57 0 0"/>
            <axis xyz="0 0 1"/>
        </joint>
    </xacro:macro>

    <xacro:wheel prefix="left" y_offset="0.175"/>
    <xacro:wheel prefix="right" y_offset="-0.175"/>

    <!-- Lidar Link -->
    <link name="lidar_link">
        <visual>
             <geometry> <cylinder radius="0.05" length="0.05"/> </geometry>
             <material name="red"> <color rgba="1 0 0 1"/> </material>
        </visual>
    </link>

    <joint name="lidar_joint" type="fixed">
        <parent link="base_link"/>
        <child link="lidar_link"/>
        <origin xyz="0.2 0 0.075" rpy="0 0 0"/>
    </joint>

</robot>
```

### 👨‍💻 Launch File (`launch/display.launch.py`)

Using `xacro` command to process the file at runtime.

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_path = get_package_share_directory('day57_urdf')
    xacro_file = os.path.join(pkg_path, 'urdf', 'robot.urdf.xacro')
    
    # Process Xacro
    robot_desc = xacro.process_file(xacro_file).toxml()
    
    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}]
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', os.path.join(pkg_path, 'config.rviz')]
        )
    ])
```

---

## 🔬 Lab Exercise: The "Exploding" Robot

### 1. Lab Objectives
- Purposefully set Inertia to 0 or extremely small values (e.g., $10^{-9}$).
- **Observe:** In Rviz it looks fine. In Gazebo (Day 58), the robot will fly into space or vibrate uncontrollably.
- **Lesson:** Physics engines hate Mass=0. Even "Fixed" links need dummy inertia or proper `fixed` joint type handling.

---

## 🚀 Project: "CAD to URDF"

**Goal:** Export a SolidWorks/Fusion360 model to URDF.
1.  **Plugin:** Use `sw2urdf` or Fusion script.
2.  **Meshes:** Exports STLs.
3.  **Origin:** Verify the Coordinate Systems (CS) align. Standard: X-Forward, Z-Up. CAD often uses Y-Up or Z-Forward.
4.  **Fix:** Create a `base_footprint` link to rotate the CAD model into ROS standards.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Wheels rotate wrong axis"
*   **Symptom:** Robot moves sideways when wheels spin.
*   **Cause:** `<axis xyz="..."/>` is wrong. Or `<origin rpy="..."/>` is wrong.
*   **Check:** In Rviz, enable TF. The Z-axis (Blue) of the wheel joint should point along the axle.

#### 2. "Floating Mesh"
*   **Symptom:** Visual mesh is 2 meters above the collision box.
*   **Cause:** The STL origin is not centered.
*   **Fix:** Open Blender. Import STL. "Set Origin to Geometry". Export.

---

## ⚡ Optimization: Convex Hulls

For complex collision meshes (e.g., a hand):
*   Don't use the usage Visual mesh (thousands of triangles).
*   Don't use a simple Box (too imprecise).
*   **Convex Hull:** A mesh wrapping tightly around the object with minimal vertices.
*   Physics engines (Bullet/PhysX) are optimized for Convex-Convex collisions.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is Xacro?
    *   **A:** XML Macro language. Allows variables, logic, and include files to make URDFs maintainable preventing code duplication.
2.  **Q:** `continuous` vs `revolute` joint?
    *   **A:** Continuous = Infinite rotation (Wheels). Revolute = Limited range (e.g., Elbow: -1.0 to +1.0 rad).
3.  **Q:** Why calculate Moment of Inertia?
    *   **A:** $F=ma$ ($\tau = I \alpha$). If $I$ is wrong, physics simulation will be unrealistic (Robot accelerates too fast or slow).

### Challenge Task
> **Task:** 4-Wheel Skid Steer.
> 1. Add 2 more wheels.
> 2. No steering mechanism.
> 3. Define skid steering plugin (Day 59) later.

---

## 📚 Further Reading
- **ROS 2 URDF Tutorials:** Official docs.
- **SDF Format Specification:** sdformat.org.

---

**Day 57 Complete**
