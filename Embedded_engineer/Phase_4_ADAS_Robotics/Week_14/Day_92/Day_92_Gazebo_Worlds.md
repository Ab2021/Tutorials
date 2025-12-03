# Day 92: Gazebo Worlds and Models
## Phase 4: ADAS & Robotics Systems | Week 14: Simulation (CARLA & Gazebo)

---

> **📝 Day 92 Focus:**
> Testing on real roads is dangerous and expensive. **Simulation** is the answer. We start with **Gazebo**, the standard simulator for ROS 2. Today, we learn how to build the world: Physics, Lighting, and Models (SDF).

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Analyze** the structure of an SDF (Simulation Description Format) file.
2.  **Create** a custom Gazebo World with physics properties (Gravity, Friction).
3.  **Import** 3D meshes (Collada .dae / STL) into Gazebo.
4.  **Configure** Sensor Plugins (Camera, LiDAR) in the model file.
5.  **Launch** a simulation with a custom robot and world.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **XML:** SDF is XML-based.
-   **ROS 2:** Launch files.
-   **3D Modeling:** Basic concept of meshes.

### Hardware Requirements
-   **GPU:** Recommended for rendering.

### Software Stack
-   **Gazebo:** `gazebo_ros_pkgs`.
-   **ROS 2:** `ros2 launch`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: SDF vs URDF

-   **URDF (Unified Robot Description Format):** Describes a *single robot* (Links, Joints). Used by `robot_state_publisher`.
-   **SDF (Simulation Description Format):** Describes the *entire world* (Robots, Lights, Physics, Terrain). Used by Gazebo.
-   **Conversion:** Gazebo can convert URDF to SDF internally, but native SDF is more powerful (supports closed loops, friction params).

### 🔹 Part 2: The World File (`.world`)

An XML file that defines:
1.  **Physics Engine:** ODE, Bullet, Dart. (Time step, Gravity).
2.  **Light Source:** Sun, Point lights.
3.  **Models:** Ground plane, Buildings, Trees.
4.  **Plugins:** World control logic (e.g., Traffic Manager).

### 🔹 Part 3: Sensor Plugins

Gazebo doesn't simulate sensors by default. We attach **Plugins** (C++ libraries) to links.
-   `libgazebo_ros_camera.so`: Publishes `sensor_msgs/Image`.
-   `libgazebo_ros_velodyne_laser.so`: Publishes `sensor_msgs/PointCloud2`.
-   **Noise Models:** We can inject Gaussian noise to simulate real-world imperfections.

---

## 💻 Implementation: Custom World & Sensor Rig

**Scenario:**
-   **World:** A flat ground with a "City Block" (Walls) and a "Sun".
-   **Robot:** A simple box with a Camera and a LiDAR.
-   **Task:** Launch it and visualize sensor data in Rviz.

### 🛠️ Setup
Create `week14_day92` package.

```bash
mkdir -p ~/ros2_ws/src/week14_day92/worlds
mkdir -p ~/ros2_ws/src/week14_day92/models/box_bot
cd ~/ros2_ws/src/week14_day92
```

### 👨‍💻 Code: The World File (`worlds/city.world`)

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">
    <!-- 1. Sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- 2. Ground Plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- 3. Physics Settings -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- 4. Custom Obstacles (Walls) -->
    <model name="wall_1">
      <pose>5 0 1 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 10 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 10 2</size>
            </box>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Bricks</name>
            </script>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### 👨‍💻 Code: The Robot SDF (`models/box_bot/model.sdf`)

```xml
<?xml version='1.0'?>
<sdf version='1.6'>
  <model name="box_bot">
    <static>false</static>
    <pose>0 0 0.5 0 0 0</pose>

    <!-- Chassis -->
    <link name='chassis'>
      <pose>0 0 0 0 0 0</pose>
      <collision name='collision'>
        <geometry>
          <box>
            <size>1 0.6 0.3</size>
          </box>
        </geometry>
      </collision>
      <visual name='visual'>
        <geometry>
          <box>
            <size>1 0.6 0.3</size>
          </box>
        </geometry>
        <material>
          <script>
            <name>Gazebo/Blue</name>
          </script>
        </material>
      </visual>
    </link>

    <!-- LiDAR Sensor -->
    <link name="lidar_link">
      <pose>0.4 0 0.2 0 0 0</pose>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.1</length>
          </cylinder>
        </geometry>
      </visual>
      
      <sensor name="lidar" type="ray">
        <pose>0 0 0 0 0 0</pose>
        <visualize>true</visualize>
        <update_rate>10</update_rate>
        <ray>
          <scan>
            <horizontal>
              <samples>360</samples>
              <resolution>1</resolution>
              <min_angle>-3.14159</min_angle>
              <max_angle>3.14159</max_angle>
            </horizontal>
          </scan>
          <range>
            <min>0.10</min>
            <max>10.0</max>
            <resolution>0.01</resolution>
          </range>
        </ray>
        <plugin name="laser_controller" filename="libgazebo_ros_ray_sensor.so">
          <ros>
            <remapping>~/out:=scan</remapping>
          </ros>
          <output_type>sensor_msgs/LaserScan</output_type>
        </plugin>
      </sensor>
    </link>
    
    <!-- Joint -->
    <joint name="lidar_joint" type="fixed">
      <parent>chassis</parent>
      <child>lidar_link</child>
    </joint>

  </model>
</sdf>
```

### 👨‍💻 Code: Launch File (`launch/sim.launch.py`)

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    # Paths
    # Note: In a real package, you'd use get_package_share_directory
    # Here we assume local paths for simplicity or hardcode for the lab
    world_path = os.path.expanduser('~/ros2_ws/src/week14_day92/worlds/city.world')
    model_path = os.path.expanduser('~/ros2_ws/src/week14_day92/models/box_bot/model.sdf')

    return LaunchDescription([
        # 1. Start Gazebo Server
        ExecuteProcess(
            cmd=['gzserver', '--verbose', '-s', 'libgazebo_ros_init.so', world_path],
            output='screen'
        ),
        
        # 2. Start Gazebo Client
        ExecuteProcess(
            cmd=['gzclient'],
            output='screen'
        ),
        
        # 3. Spawn Robot
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'box_bot', '-file', model_path, '-x', '0', '-y', '0', '-z', '1'],
            output='screen'
        )
    ])
```

---

## 🔬 Lab Exercise: The Virtual Lidar

### Lab Objectives
1.  Run `ros2 launch week14_day92 sim.launch.py`.
2.  **Observation:**
    -   Gazebo opens. You see a Blue Box (Robot) and a Brick Wall.
    -   Blue rays (LiDAR) are scanning the wall.
3.  **Rviz:**
    -   Run `rviz2`.
    -   Add `LaserScan`. Topic: `/scan`.
    -   Fixed Frame: `lidar_link` (You might need a static transform publisher since we didn't start `robot_state_publisher`).
    -   **Result:** You see red dots outlining the wall 5 meters away.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Model not found"
**Symptom:** Gazebo shows a black screen or errors.
**Cause:** `GAZEBO_MODEL_PATH` env variable is missing.
**Solution:** `export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/ros2_ws/src/week14_day92/models`

#### 2. Real Time Factor < 1.0
**Symptom:** Simulation runs in slow motion.
**Cause:** CPU overload. Physics is too complex or GPU is weak.
**Solution:** Increase `max_step_size` in `.world` file (e.g., 0.001 -> 0.004). This reduces accuracy but increases speed.

---

## ⚡ Optimization & Best Practices

### 1. Mesh Simplification
Don't use a 100MB CAD model for collision.
-   **Visual:** High-poly mesh.
-   **Collision:** Simple primitive (Box, Cylinder) or Low-poly Convex Hull.
-   Physics engines hate high-poly concave meshes.

### 2. Headless Mode
For automated testing (CI/CD), don't run the GUI (`gzclient`).
-   Run only `gzserver`.
-   Saves massive GPU/CPU resources.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between `<visual>` and `<collision>`?
    *   **A:** **Visual** is what you see (Textures, Colors). **Collision** is what the physics engine calculates (Bouncing, Friction).
2.  **Q:** Why do we need a Plugin for the LiDAR?
    *   **A:** The physics engine calculates collisions, but it doesn't know how to format that data into a ROS `LaserScan` message. The plugin bridges Gazebo and ROS.
3.  **Q:** Can I use Python to write Gazebo Plugins?
    *   **A:** Generally No. Gazebo Classic plugins are C++. (Gazebo Ignition supports Python bindings, but C++ is standard).

### Challenge Task
**Task:** Add a Camera.
1.  Add a new link `camera_link` to the SDF.
2.  Add a `<sensor type="camera">`.
3.  Use plugin `libgazebo_ros_camera.so`.
4.  Visualize the image in Rviz.

---

## 📚 Further Reading & References
-   [Gazebo SDF Specification](http://sdformat.org/spec)
-   [Gazebo ROS 2 Plugins](https://github.com/ros-simulation/gazebo_ros_pkgs/wiki)

---

**Day 92 Complete** | Phase 4: ADAS & Robotics Systems | Week 14: Simulation (CARLA & Gazebo)
