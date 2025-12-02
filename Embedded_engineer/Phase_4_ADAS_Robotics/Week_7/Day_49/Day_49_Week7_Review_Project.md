# Day 49: Week 7 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 7: Simulation & Modeling

---

> **📝 Day 49 Focus:**
> We have built a virtual universe. We have a robot (URDF), a world (SDF), physics (Gazebo), sensors (Plugins), and inhabitants (Actors). Today, we combine all these elements into the **Autonomous City Explorer** project—a comprehensive simulation environment for testing ADAS stacks.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Integrate** URDF, SDF, and Plugins into a unified simulation package.
2.  **Launch** a complex Gazebo world with static and dynamic elements.
3.  **Deploy** the "RoboCar" with a full sensor suite (Lidar, Camera, GPS, IMU).
4.  **Execute** a navigation mission in the simulated city.
5.  **Evaluate** your mastery of Week 7 concepts through a comprehensive assessment.

---

## 📚 Week 7 Review

### 1. Robot Modeling (URDF/Xacro)
-   **Links:** Visual (Mesh), Collision (Primitive), Inertial (Mass).
-   **Joints:** Fixed, Continuous, Revolute.
-   **Xacro:** Macros to reduce code duplication.

### 2. Gazebo Physics
-   **ODE:** Rigid body dynamics.
-   **Properties:** Friction (`mu`), Colors (`Gazebo/Blue`), Collision bitmasks.

### 3. Plugins
-   **Model:** Diff Drive (Control).
-   **Sensor:** Ray (Lidar), Camera, IMU, GPS.
-   **Noise:** Gaussian noise makes simulation useful.

### 4. World Building
-   **Building Editor:** Floor plans.
-   **SDF:** The native language of Gazebo.
-   **Actors:** Scripted pedestrians and traffic.

### 5. HIL (Hardware-in-the-Loop)
-   Connecting real ECUs to the simulation bridge.

---

## 🛠️ Capstone Project: Autonomous City Explorer

**Goal:** Create a "Digital Twin" of a small city block and deploy the RoboCar to navigate it.

**Requirements:**
1.  **World:** A loop road with 2 buildings, a stop sign, and a tree.
2.  **Traffic:** One car driving in a loop.
3.  **Pedestrian:** One walker crossing the street.
4.  **Robot:** RoboCar with Lidar, Camera, GPS, IMU.
5.  **Mission:** Drive one lap around the block (Teleop or Auto).

### Package Structure
Create `week7_project` folder.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake week7_project
mkdir urdf launch worlds models rviz
```

### 👨‍💻 Code: The World (city_mission.world)

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="city_mission">
    <!-- Environment -->
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>

    <!-- Physics -->
    <physics type="ode">
      <real_time_update_rate>1000.0</real_time_update_rate>
      <max_step_size>0.001</max_step_size>
    </physics>

    <!-- 1. Road Network (Simple Loop) -->
    <road name="loop_road">
      <width>8.0</width>
      <point>0 0 0</point>
      <point>40 0 0</point>
      <point>40 40 0</point>
      <point>0 40 0</point>
      <point>0 0 0</point>
    </road>

    <!-- 2. Buildings (Standard Models) -->
    <include>
      <uri>model://grocery_store</uri>
      <pose>20 -15 0 0 0 0</pose>
    </include>
    <include>
      <uri>model://post_office</uri>
      <pose>20 55 0 0 0 0</pose>
    </include>

    <!-- 3. Stop Sign -->
    <include>
      <uri>model://stop_sign</uri>
      <pose>35 -2 0 0 0 1.57</pose>
    </include>

    <!-- 4. Dynamic Traffic -->
    <actor name="traffic_car">
      <skin>
        <filename>model://pickup/meshes/pickup.dae</filename>
      </skin>
      <script>
        <loop>true</loop>
        <auto_start>true</auto_start>
        <trajectory id="0" type="driving">
          <waypoint><time>0</time><pose>0 2 0 0 0 0</pose></waypoint>
          <waypoint><time>10</time><pose>40 2 0 0 0 0</pose></waypoint>
          <waypoint><time>15</time><pose>40 42 0 0 0 1.57</pose></waypoint>
          <waypoint><time>25</time><pose>0 42 0 0 0 3.14</pose></waypoint>
          <waypoint><time>30</time><pose>0 2 0 0 0 4.71</pose></waypoint>
          <waypoint><time>35</time><pose>0 2 0 0 0 6.28</pose></waypoint>
        </trajectory>
      </script>
    </actor>

    <!-- 5. Pedestrian -->
    <actor name="walker">
      <skin><filename>walk.dae</filename></skin>
      <animation name="walking"><filename>walk.dae</filename></animation>
      <script>
        <loop>true</loop>
        <auto_start>true</auto_start>
        <trajectory id="0" type="walking">
          <waypoint><time>0</time><pose>20 -5 0 0 0 1.57</pose></waypoint>
          <waypoint><time>5</time><pose>20 5 0 0 0 1.57</pose></waypoint>
          <waypoint><time>10</time><pose>20 -5 0 0 0 -1.57</pose></waypoint>
        </trajectory>
      </script>
    </actor>
    
    <!-- GPS Reference -->
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <latitude_deg>37.4</latitude_deg>
      <longitude_deg>-122.1</longitude_deg>
      <elevation>0.0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>

  </world>
</sdf>
```

### 👨‍💻 Code: main.launch.py

This launch file brings up everything: Gazebo, Robot, RSP, Rviz.

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node

def generate_launch_description():
    pkg_path = get_package_share_directory('week7_project')
    
    # 1. World
    world_file = os.path.join(pkg_path, 'worlds', 'city_mission.world')
    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        launch_arguments={'world': world_file}.items()
    )

    # 2. Robot Description (Xacro)
    # Reusing the robust Xacro from Day 46 (with sensors)
    # Assuming it's copied to week7_project/urdf/robocar.xacro
    xacro_file = os.path.join(pkg_path, 'urdf', 'robocar.xacro')
    robot_desc = Command(['xacro ', xacro_file])
    
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}]
    )

    # 3. Spawn Robot
    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'robocar', '-x', '5', '-y', '-2', '-z', '0.5'],
        output='screen'
    )

    # 4. Rviz2
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        # arguments=['-d', os.path.join(pkg_path, 'rviz', 'config.rviz')]
    )

    return LaunchDescription([
        gazebo,
        rsp,
        spawn,
        rviz
    ])
```

### 🛠️ Build & Run

```bash
cd ~/ros2_ws
colcon build --packages-select week7_project
source install/setup.bash
ros2 launch week7_project main.launch.py
```

---

## 🧪 Verification & Testing

### 1. Sensor Check
-   **Lidar:** `/scan` should show the buildings and the passing car.
-   **Camera:** `/camera/image_raw` should show the stop sign.
-   **GPS:** `/gps/fix` should show valid coordinates.

### 2. Interaction
-   Drive the robot (`teleop_twist_keyboard`).
-   Wait for the pedestrian to cross.
-   Follow the traffic car.

### 3. Physics
-   Crash into the grocery store.
-   *Result:* The robot should stop abruptly (Collision).

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Modeling
1.  **Q:** Why do we define `<inertial>` properties?
    *   **A:** For the physics engine (ODE) to calculate forces ($F=ma$). Without mass, the object is static or behaves erratically.
2.  **Q:** What is the purpose of the `<transmission>` tag in URDF?
    *   **A:** It links a Joint to an Actuator (Motor), defining the reduction ratio. Used by `ros_control` (though Gazebo plugins often abstract this).

### Section 2: Simulation
3.  **Q:** How do we simulate a Lidar?
    *   **A:** Using a Ray Sensor plugin. It casts rays from a center point and returns the distance to the first collision.
4.  **Q:** What is the limitation of Gazebo Actors?
    *   **A:** They are "Ghosts". They have visual meshes and follow paths, but usually lack collision geometry and do not react to physics (ragdoll) or the robot (unless a custom plugin is written).

### Section 3: Integration
5.  **Q:** Why does the robot drift in Rviz even if I don't move it?
    *   **A:** Sensor Noise! If you are visualizing Odometry or IMU with noise enabled, the estimated position will jitter.

---

## 🏆 Conclusion

Congratulations on completing Week 7!
-   You have mastered the art of **Simulation**.
-   You can build robots, worlds, and scenarios.
-   You have a safe playground to test your code.

**Next Week:** We move to **State Estimation & Fusion**. We have noisy sensors; now we need to clean them up. We will implement **Kalman Filters** and **Particle Filters** to track the robot's position accurately.

---

**Day 49 Complete** | Phase 4: ADAS & Robotics Systems | Week 7: Simulation & Modeling
