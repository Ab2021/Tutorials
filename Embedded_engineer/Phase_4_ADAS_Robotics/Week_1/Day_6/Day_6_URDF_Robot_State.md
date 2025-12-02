# Day 6: URDF, Xacro, and Robot State Publisher
## Phase 4: ADAS & Robotics Systems | Week 1: ROS 2 Fundamentals

---

> **📝 Day 6 Focus:**
> To simulate an autonomous vehicle or visualize it in RViz, we need a digital twin. **URDF (Unified Robot Description Format)** is the standard XML format for describing robot geometry and kinematics. We will use **Xacro** to make these descriptions modular and scalable.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Construct** a complete vehicle model using URDF (Links, Joints, Visuals, Collisions).
2.  **Optimize** URDF development using Xacro macros and properties.
3.  **Deploy** the `robot_state_publisher` to broadcast the static TF tree based on the model.
4.  **Integrate** sensor models (LiDAR, Camera) onto the vehicle chassis.
5.  **Visualize** and debug the kinematic chain in RViz.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **XML:** Basic syntax.
-   **TF2:** Understanding of parent/child frames (Day 5).
-   **Kinematics:** Basic understanding of joints (revolute, continuous, fixed).

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS with ROS 2 Humble.

### Software Stack
-   **ROS 2 Humble**
-   **Packages:** `urdf`, `xacro`, `robot_state_publisher`, `joint_state_publisher_gui`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: URDF Fundamentals

#### 1.1 The Structure of a Robot

A robot in URDF is a tree of **Links** connected by **Joints**.

1.  **Link:** A rigid body (e.g., Chassis, Wheel, Sensor).
    -   **Visual:** What it looks like (Mesh, Cylinder, Box).
    -   **Collision:** Simplified geometry for physics engine (Box, Cylinder).
    -   **Inertial:** Mass and Moment of Inertia (for Gazebo simulation).

2.  **Joint:** The connection between two links.
    -   **Type:**
        -   `fixed`: No movement (e.g., Sensor on chassis).
        -   `continuous`: Rotates forever (e.g., Wheel).
        -   `revolute`: Rotates with limits (e.g., Steering knuckle).
        -   `prismatic`: Slides (e.g., Suspension).
    -   **Parent/Child:** Defines the TF hierarchy.
    -   **Origin:** The transform from Parent to Child.
    -   **Axis:** The axis of rotation/translation.

#### 1.2 XML Syntax Example

```xml
<robot name="my_car">
  <!-- Chassis Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="4.5 1.8 1.5"/>
      </geometry>
      <material name="blue"/>
    </visual>
  </link>

  <!-- Wheel Link -->
  <link name="wheel_left">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.3"/>
      </geometry>
    </visual>
  </link>

  <!-- Joint -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left"/>
    <origin xyz="1.5 0.9 0.3" rpy="-1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
```

---

### 🔹 Part 2: Xacro (XML Macros)

Writing raw URDF is repetitive (4 wheels = 4 nearly identical blocks).
**Xacro** is a macro language that generates URDF.

#### 2.1 Properties (Variables)
```xml
<xacro:property name="wheel_radius" value="0.3" />
<xacro:property name="wheel_width" value="0.2" />

<cylinder radius="${wheel_radius}" length="${wheel_width}"/>
```

#### 2.2 Macros (Functions)
```xml
<xacro:macro name="wheel" params="prefix x_reflect y_reflect">
  <link name="${prefix}_wheel">
    ...
  </link>
  <joint name="${prefix}_wheel_joint" type="continuous">
    <origin xyz="${x_reflect*wheelbase/2} ${y_reflect*track/2} 0" ... />
    ...
  </joint>
</xacro:macro>

<!-- Usage -->
<xacro:wheel prefix="front_left" x_reflect="1" y_reflect="1" />
<xacro:wheel prefix="rear_right" x_reflect="-1" y_reflect="-1" />
```

#### 2.3 Math
Xacro supports python math expressions inside `${...}`.
```xml
<origin xyz="0 0 ${wheel_radius + 0.05}" />
```

---

### 🔹 Part 3: Robot State Publisher

How does the URDF become TF frames?

1.  **`robot_state_publisher` Node:**
    -   Reads the URDF (passed via `robot_description` parameter).
    -   Subscribes to `/joint_states` (positions of moving joints).
    -   Calculates the forward kinematics.
    -   Publishes the full TF tree (`/tf` and `/tf_static`).

2.  **`joint_state_publisher` Node:**
    -   Publishes `/joint_states`.
    -   For simulation/hardware, this comes from the driver.
    -   For testing, we use `joint_state_publisher_gui` to move joints with sliders.

---

## 💻 Implementation: ADAS Vehicle Model

We will build a package `adas_description` containing the Xacro model of a simplified autonomous car with sensors.

### 🛠️ Package Setup

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake \
  --dependencies urdf xacro \
  --node-name dummy \
  adas_description

cd adas_description
mkdir urdf launch rviz meshes
```

### 📦 Xacro Implementation

#### 1. `urdf/vehicle_properties.xacro`
Define constants here.

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Vehicle Dimensions (Approx. Sedan) -->
  <xacro:property name="chassis_length" value="4.5" />
  <xacro:property name="chassis_width" value="1.8" />
  <xacro:property name="chassis_height" value="1.4" />
  <xacro:property name="wheelbase" value="2.8" />
  <xacro:property name="track_width" value="1.6" />
  <xacro:property name="ground_clearance" value="0.2" />
  
  <!-- Wheel Properties -->
  <xacro:property name="wheel_radius" value="0.35" />
  <xacro:property name="wheel_width" value="0.25" />
  
  <!-- Sensor Positions -->
  <xacro:property name="lidar_x" value="1.0" /> <!-- Forward from center -->
  <xacro:property name="lidar_z" value="1.5" /> <!-- Height -->
</robot>
```

#### 2. `urdf/macros.xacro`
Reusable components.

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  
  <!-- Wheel Macro -->
  <xacro:macro name="vehicle_wheel" params="prefix x_pos y_pos">
    <link name="${prefix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="${pi/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black">
          <color rgba="0.1 0.1 0.1 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="${pi/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="chassis_link"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${x_pos} ${y_pos} ${wheel_radius}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <!-- LiDAR Macro -->
  <xacro:macro name="lidar_sensor" params="parent x y z">
    <link name="lidar_link">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.1"/>
        </geometry>
        <material name="red">
          <color rgba="1 0 0 1"/>
        </material>
      </visual>
    </link>

    <joint name="lidar_joint" type="fixed">
      <parent link="${parent}"/>
      <child link="lidar_link"/>
      <origin xyz="${x} ${y} ${z}" rpy="0 0 0"/>
    </joint>
  </xacro:macro>

</robot>
```

#### 3. `urdf/adas_car.urdf.xacro`
Main file.

```xml
<?xml version="1.0"?>
<robot name="adas_car" xmlns:xacro="http://www.ros.org/wiki/xacro">
  
  <!-- Includes -->
  <xacro:include filename="vehicle_properties.xacro"/>
  <xacro:include filename="macros.xacro"/>

  <!-- Base Footprint (Projection on ground) -->
  <link name="base_footprint"/>

  <!-- Chassis Link -->
  <link name="chassis_link">
    <visual>
      <origin xyz="0 0 ${chassis_height/2 + ground_clearance}" rpy="0 0 0"/>
      <geometry>
        <box size="${chassis_length} ${chassis_width} ${chassis_height}"/>
      </geometry>
      <material name="silver">
        <color rgba="0.8 0.8 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 ${chassis_height/2 + ground_clearance}" rpy="0 0 0"/>
      <geometry>
        <box size="${chassis_length} ${chassis_width} ${chassis_height}"/>
      </geometry>
    </collision>
  </link>

  <joint name="base_joint" type="fixed">
    <parent link="base_footprint"/>
    <child link="chassis_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <!-- Wheels -->
  <xacro:vehicle_wheel prefix="front_left" x_pos="${wheelbase/2}" y_pos="${track_width/2}"/>
  <xacro:vehicle_wheel prefix="front_right" x_pos="${wheelbase/2}" y_pos="-${track_width/2}"/>
  <xacro:vehicle_wheel prefix="rear_left" x_pos="-${wheelbase/2}" y_pos="${track_width/2}"/>
  <xacro:vehicle_wheel prefix="rear_right" x_pos="-${wheelbase/2}" y_pos="-${track_width/2}"/>

  <!-- Sensors -->
  <xacro:lidar_sensor parent="chassis_link" x="${lidar_x}" y="0" z="${lidar_z}"/>

</robot>
```

### 🚀 Launch File

We need a launch file that:
1.  Processes the Xacro file into URDF.
2.  Publishes the description to the parameter server.
3.  Starts `robot_state_publisher`.
4.  Starts `joint_state_publisher_gui`.
5.  Starts RViz.

Create `launch/display.launch.py`:

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('adas_description')
    
    # Paths
    xacro_file = os.path.join(pkg_share, 'urdf', 'adas_car.urdf.xacro')
    rviz_config = os.path.join(pkg_share, 'rviz', 'view_car.rviz')

    # Process Xacro
    robot_description_content = Command(['xacro ', xacro_file])
    
    # Create robot_description parameter
    robot_description = {'robot_description': robot_description_content}

    # Nodes
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    node_joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )

    node_rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    return LaunchDescription([
        node_robot_state_publisher,
        node_joint_state_publisher_gui,
        node_rviz
    ])
```

### 🛠️ CMakeLists.txt Updates

We need to install the directories so the launch file can find them.

```cmake
install(DIRECTORY
  launch
  urdf
  rviz
  DESTINATION share/${PROJECT_NAME}
)
```

---

## 🔬 Lab Exercise: Visualizing the Car

### Lab Objectives
1.  Build the package.
2.  Launch the display.
3.  Configure RViz.
4.  Verify the TF tree.

### Part 1: Build & Launch
```bash
colcon build --packages-select adas_description
source install/setup.bash
ros2 launch adas_description display.launch.py
```

### Part 2: RViz Configuration
1.  RViz opens (empty).
2.  **Fixed Frame:** Set to `base_footprint`.
3.  **Add Display:** RobotModel.
    -   Description Topic: `/robot_description`.
    -   You should see the silver box and black wheels.
4.  **Add Display:** TF.
    -   You should see the frames.
5.  **Save Config:** File -> Save Config As -> `src/adas_description/rviz/view_car.rviz`.

### Part 3: Joint Interaction
1.  Find the "Joint State Publisher" GUI window.
2.  Move the sliders for the wheels.
3.  Observe the wheels rotating in RViz.
    -   *Note:* Since they are `continuous` joints, they spin.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Robot Description not found"
**Symptom:** RViz RobotModel shows error.
**Cause:** `robot_state_publisher` didn't receive the URDF, or the topic is wrong.
**Solution:**
-   Check `ros2 param get /robot_state_publisher robot_description`.
-   Ensure `Command(['xacro ...'])` worked.

#### 2. "No Transform from [X] to [Y]"
**Symptom:** RViz TF error, model parts white/transparent.
**Cause:** Broken TF tree.
**Solution:**
-   Run `ros2 run tf2_tools view_frames`.
-   Ensure `joint_state_publisher` is running (it publishes the transforms for non-fixed joints).

#### 3. Xacro Syntax Error
**Symptom:** Launch fails with XML parsing error.
**Solution:**
-   Run `xacro src/adas_description/urdf/adas_car.urdf.xacro` manually in terminal to see the error message.

---

## ⚡ Optimization & Best Practices

### 1. Simplify Collision Geometry
-   Visuals can be high-poly meshes (DAE/STL).
-   Collisions **MUST** be simple primitives (Box, Cylinder, Sphere) for physics performance.
-   Don't use the visual mesh for collision unless it's a convex hull and very simple.

### 2. Use `base_footprint`
-   Standard practice: `base_footprint` is at ground level (z=0).
-   `base_link` is the chassis center (z > 0).
-   This makes navigation planning easier (2D map projection).

### 3. Modular Xacro
-   Keep sensors in separate files (`lidar.xacro`, `camera.xacro`).
-   Pass arguments to macros (`x`, `y`, `z`, `r`, `p`, `y`) to mount them easily.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between `robot_state_publisher` and `joint_state_publisher`?
    *   **A:** `joint_state_publisher` publishes the *angles* of joints. `robot_state_publisher` takes those angles + URDF and publishes the *3D Transforms* (TF).

2.  **Q:** Why use Xacro instead of plain URDF?
    *   **A:** Variables, math, and macros reduce code duplication and make the model configurable.

3.  **Q:** What joint type would you use for a steering mechanism?
    *   **A:** `revolute` (because it has limits, unlike `continuous`).

### Challenge Task
**Task:** Add a Steering Mechanism.
1.  Modify the front wheels to be mounted on "Steering Knuckles".
2.  Add a `revolute` joint between Chassis and Knuckle (Yaw axis).
3.  Add a `continuous` joint between Knuckle and Wheel (Pitch axis).
4.  Visualize the steering action in GUI.

---

## 📚 Further Reading & References
-   [URDF XML Reference](http://wiki.ros.org/urdf/XML)
-   [Xacro Tutorials](http://wiki.ros.org/xacro)
-   [ROS 2 URDF Tutorial](https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html)

---

**Day 6 Complete** | Phase 4: ADAS & Robotics Systems | Week 1: ROS 2 Fundamentals
