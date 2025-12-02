# Day 43: URDF & Xacro (Robot Modeling)
## Phase 4: ADAS & Robotics Systems | Week 7: Simulation & Modeling

---

> **📝 Day 43 Focus:**
> Before we build a real robot, we build a virtual one. **URDF (Unified Robot Description Format)** is the standard way to tell ROS "this is what my robot looks like, how its joints move, and how heavy it is." Today, we master URDF and its powerful cousin, **Xacro**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the structure of a URDF file: Links, Joints, Visuals, Collisions, Inertials.
2.  **Visualize** a robot model in Rviz2 using `robot_state_publisher`.
3.  **Optimize** URDFs using **Xacro** macros to reduce code duplication.
4.  **Model** a 4-wheeled mobile robot with a Lidar sensor.
5.  **Debug** common URDF issues like "No Transform" or "Collapsing Joints".

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **XML:** Basic syntax (Tags, Attributes).
-   **TF2:** Coordinate transforms (Day 11).

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS.

### Software Stack
-   **ROS 2:** `urdf`, `xacro`, `robot_state_publisher`, `joint_state_publisher_gui`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Anatomy of a URDF

A robot is a tree of **Links** connected by **Joints**.

#### 1.1 Links
Represent rigid bodies (Chassis, Wheel, Lidar).
-   **Visual:** What it looks like (Mesh/Box/Cylinder).
-   **Collision:** Simplified shape for physics engine (Box/Sphere).
-   **Inertial:** Mass and Moment of Inertia ($I_{xx}, I_{yy}, I_{zz}$).

#### 1.2 Joints
Connect two links (Parent -> Child).
-   **Fixed:** No movement (e.g., Lidar on Chassis).
-   **Continuous:** Rotates forever (e.g., Wheel).
-   **Revolute:** Rotates with limits (e.g., Robot Arm).
-   **Prismatic:** Slides (e.g., Elevator).

### 🔹 Part 2: Xacro (XML Macros)

Writing raw URDF is painful (Copy-pasting 4 wheels).
**Xacro** adds programming features to XML:
-   **Properties:** Variables (`<xacro:property name="wheel_radius" value="0.1"/>`).
-   **Macros:** Functions (`<xacro:macro name="create_wheel" params="side">`).
-   **Math:** `${2 * pi}`.

### 🔹 Part 3: The Publishing Pipeline

1.  **URDF/Xacro** file describes the robot.
2.  **`robot_state_publisher`** reads URDF + Joint Angles -> Publishes TF tree.
3.  **`joint_state_publisher`** publishes Joint Angles (Fake/Real).
4.  **Rviz2** reads TF tree + URDF -> Draws the robot.

---

## 💻 Implementation: Modeling "RoboCar"

We will create a package `my_robot_description` and model a car with 4 wheels and a Lidar.

### 🛠️ Setup
Create `week7_day43` package.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake week7_day43 --dependencies urdf xacro
cd week7_day43
mkdir urdf launch rviz
touch urdf/robocar.xacro launch/display.launch.py
```

### 👨‍💻 Code: robocar.xacro

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="robocar">

    <!-- Properties -->
    <xacro:property name="chassis_len" value="0.5"/>
    <xacro:property name="chassis_width" value="0.3"/>
    <xacro:property name="chassis_height" value="0.1"/>
    <xacro:property name="wheel_radius" value="0.1"/>
    <xacro:property name="wheel_width" value="0.05"/>
    
    <!-- Colors -->
    <material name="blue">
        <color rgba="0 0 0.8 1"/>
    </material>
    <material name="black">
        <color rgba="0 0 0 1"/>
    </material>

    <!-- Base Link (Dummy) -->
    <link name="base_link"/>

    <!-- Chassis -->
    <joint name="chassis_joint" type="fixed">
        <parent link="base_link"/>
        <child link="chassis"/>
        <origin xyz="-0.1 0 0"/>
    </joint>

    <link name="chassis">
        <visual>
            <origin xyz="${chassis_len/2} 0 ${chassis_height/2}"/>
            <geometry>
                <box size="${chassis_len} ${chassis_width} ${chassis_height}"/>
            </geometry>
            <material name="blue"/>
        </visual>
        <collision>
            <origin xyz="${chassis_len/2} 0 ${chassis_height/2}"/>
            <geometry>
                <box size="${chassis_len} ${chassis_width} ${chassis_height}"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
        </inertial>
    </link>

    <!-- Wheel Macro -->
    <xacro:macro name="wheel" params="prefix x_reflect y_reflect">
        <link name="${prefix}_wheel">
            <visual>
                <geometry>
                    <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
                </geometry>
                <material name="black"/>
            </visual>
            <collision>
                <geometry>
                    <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
                </geometry>
            </collision>
            <inertial>
                <mass value="0.1"/>
                <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
            </inertial>
        </link>

        <joint name="${prefix}_wheel_joint" type="continuous">
            <parent link="base_link"/>
            <child link="${prefix}_wheel"/>
            <origin xyz="${x_reflect*0.15} ${y_reflect*0.175} 0" rpy="${-pi/2} 0 0"/>
            <axis xyz="0 0 1"/>
        </joint>
    </xacro:macro>

    <!-- Create 4 Wheels -->
    <xacro:wheel prefix="front_left" x_reflect="1" y_reflect="1"/>
    <xacro:wheel prefix="front_right" x_reflect="1" y_reflect="-1"/>
    <xacro:wheel prefix="rear_left" x_reflect="-1" y_reflect="1"/>
    <xacro:wheel prefix="rear_right" x_reflect="-1" y_reflect="-1"/>

    <!-- Lidar -->
    <link name="lidar_link">
        <visual>
            <geometry>
                <cylinder radius="0.05" length="0.04"/>
            </geometry>
            <material name="black"/>
        </visual>
    </link>

    <joint name="lidar_joint" type="fixed">
        <parent link="chassis"/>
        <child link="lidar_link"/>
        <origin xyz="0.2 0 0.12" rpy="0 0 0"/>
    </joint>

</robot>
```

### 👨‍💻 Code: display.launch.py

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node

def generate_launch_description():
    pkg_path = get_package_share_directory('week7_day43')
    xacro_file = os.path.join(pkg_path, 'urdf', 'robocar.xacro')

    # Process Xacro
    robot_description_config = Command(['xacro ', xacro_file])
    params = {'robot_description': robot_description_config}

    # Robot State Publisher
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    # Joint State Publisher GUI
    node_joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen'
    )

    # Rviz2
    node_rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        # arguments=['-d', os.path.join(pkg_path, 'rviz', 'config.rviz')]
    )

    return LaunchDescription([
        node_robot_state_publisher,
        node_joint_state_publisher_gui,
        node_rviz
    ])
```

### 👨‍💻 Code: CMakeLists.txt

```cmake
install(DIRECTORY urdf launch rviz
  DESTINATION share/${PROJECT_NAME}
)
```

### 🛠️ Build & Run

```bash
cd ~/ros2_ws
colcon build --packages-select week7_day43
source install/setup.bash
ros2 launch week7_day43 display.launch.py
```

---

## 🔬 Lab Exercise: Rviz Visualization

### Lab Objectives
1.  Launch the file. Rviz2 opens.
2.  **Setup:**
    -   Set "Fixed Frame" to `base_link`.
    -   Add "RobotModel" display.
    -   Add "TF" display.
3.  **Interact:**
    -   Use the **Joint State Publisher GUI** slider.
    -   Move the wheel joints.
    -   *Observation:* The wheels rotate in Rviz. The TF frames rotate with them.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "No Transform from [X] to [Y]"
**Symptom:** Robot model is white/broken in Rviz. "Global Status: Error".
**Cause:** `robot_state_publisher` is not running, or the TF tree is broken (disconnected joints).
**Solution:** Check `ros2 run tf2_tools view_frames`. Ensure all links connect back to `base_link`.

#### 2. Model Collapsed at Origin
**Symptom:** All parts are jumbled at (0,0,0).
**Cause:** Missing `<origin>` tags in joints.
**Solution:** Define offsets (xyz) for every joint.

#### 3. Xacro Parse Error
**Symptom:** Launch fails.
**Cause:** XML syntax error (unclosed tag, typo in `${}`).
**Solution:** Run `xacro robocar.xacro` manually in terminal to see the error message.

---

## ⚡ Optimization & Best Practices

### 1. Dummy Base Link
Always start with a dummy `base_link` (no geometry).
Connect the chassis to it.
-   **Why?** Allows attaching the robot to the world (Gazebo) or changing the chassis without breaking the root.

### 2. Meshes
Don't use `<box>` for everything.
Use CAD models (STL/DAE).
-   `<mesh filename="package://my_robot/meshes/chassis.stl" scale="1 1 1"/>`
-   **Tip:** Use simplified STL (low poly) for `<collision>` to speed up physics.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between `robot_state_publisher` and `joint_state_publisher`?
    *   **A:** `joint_state_publisher` says "The wheel is at 45 degrees". `robot_state_publisher` calculates the 3D transform matrix (TF) for that angle and publishes it.
2.  **Q:** Why use Xacro instead of URDF?
    *   **A:** To avoid code duplication (Macros) and allow parameterization (Properties).
3.  **Q:** What happens if I forget the `<inertial>` tag?
    *   **A:** Rviz works fine (Visual only). Gazebo crashes or the robot flies away (Physics engine needs mass).

### Challenge Task
**Task:** Add a Camera.
1.  Create a camera link (box).
2.  Joint it to the front of the chassis.
3.  Add an "Optical Frame" (rotated -90 deg) for correct image orientation.
4.  Visualize the TF tree.

---

## 📚 Further Reading & References
-   [ROS 2 URDF Tutorials](https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html)
-   [Xacro Wiki](http://wiki.ros.org/xacro)

---

**Day 43 Complete** | Phase 4: ADAS & Robotics Systems | Week 7: Simulation & Modeling
