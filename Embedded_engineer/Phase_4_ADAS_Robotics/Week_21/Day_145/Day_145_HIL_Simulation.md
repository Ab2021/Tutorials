# Day 145: Hardware-in-the-Loop (HIL) & Simulation
## Phase 4: ADAS & Robotics Systems | Week 21: System Integration & Capstone

---

> **📝 Day 145 Focus:**
> Testing on a real car is expensive and dangerous. **Simulation** is the answer. **CARLA** is an open-source simulator for autonomous driving research. We will bridge our ROS 2 stack to CARLA to test our algorithms in a photorealistic world.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Install** and **Run** the CARLA Simulator.
2.  **Configure** the `carla-ros-bridge`.
3.  **Spawn** a vehicle with sensors (Lidar, Camera, GNSS).
4.  **Visualize** sensor data in Rviz.
5.  **Control** the simulated car from ROS 2.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **ROS 2:** Topics, Transforms (TF).
-   **Docker:** (Optional, if running CARLA in Docker).

### Hardware Requirements
-   **GPU:** Mandatory (NVIDIA GTX 1060+). CARLA is heavy.
-   **RAM:** 16GB+.

### Software Stack
-   **CARLA:** 0.9.13 or newer.
-   **carla-ros-bridge:** Humble branch.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Simulator (CARLA)

-   **Unreal Engine 4:** Provides physics and rendering.
-   **Server-Client:** CARLA runs as a server (World). Python scripts connect as clients to spawn actors.
-   **Assets:** Towns, Cars, Pedestrians, Weather.

### 🔹 Part 2: The Bridge

How does ROS talk to Unreal Engine?
-   **carla-ros-bridge:** A ROS node that connects to the CARLA Python API.
-   **Sensors:** Converts CARLA data -> ROS Messages (`Image`, `PointCloud2`, `Odometry`).
-   **Control:** Converts ROS Messages (`AckermannDrive`) -> CARLA Vehicle Control.
-   **Clock:** Synchronizes ROS time with Sim time.

---

## 💻 Implementation: Connecting to CARLA

**Scenario:**
-   Spawn an Ego Vehicle in "Town01".
-   Equip it with a Lidar and Camera.
-   Drive it manually via ROS.

### 🛠️ Setup
1.  **Install CARLA:** Download pre-compiled binary from GitHub or use Docker.
    ```bash
    # Docker method (Easiest)
    docker run --priviliged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.13 /bin/bash ./CarlaUE4.sh -RenderOffScreen
    ```
2.  **Install Bridge:**
    ```bash
    sudo apt install ros-humble-carla-ros-bridge
    ```

### 📄 Launch File (`launch/carla_bridge.launch.py`)

Create `week21_day145/launch/carla_bridge.launch.py`.

```python
import os
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Arguments
    host = LaunchConfiguration('host', default='localhost')
    port = LaunchConfiguration('port', default='2000')
    town = LaunchConfiguration('town', default='Town01')
    
    # Bridge Launch
    bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('carla_ros_bridge'), 'launch', 'carla_ros_bridge.launch.py')
        ),
        launch_arguments={
            'host': host,
            'port': port,
            'town': town,
            'timeout': '10.0'
        }.items()
    )
    
    # Spawn Ego Vehicle
    # This spawns a car defined in 'objects.json' (standard config)
    spawn_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('carla_spawn_objects'), 'launch', 'carla_spawn_objects.launch.py')
        ),
        launch_arguments={
            'objects_definition_file': os.path.join(get_package_share_directory('week21_day145'), 'config', 'objects.json')
        }.items()
    )
    
    # Manual Control (Optional)
    manual_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('carla_manual_control'), 'launch', 'carla_manual_control.launch.py')
        )
    )

    return LaunchDescription([
        bridge_launch,
        spawn_launch,
        manual_control
    ])
```

### 📄 Objects Config (`config/objects.json`)

```json
{
    "objects": [
        {
            "type": "vehicle.tesla.model3",
            "id": "ego_vehicle",
            "sensors": [
                {
                    "type": "sensor.camera.rgb",
                    "id": "rgb_front",
                    "spawn_point": {"x": 2.0, "y": 0.0, "z": 1.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                    "image_size_x": 800,
                    "image_size_y": 600,
                    "fov": 90.0
                },
                {
                    "type": "sensor.lidar.ray_cast",
                    "id": "lidar_top",
                    "spawn_point": {"x": 0.0, "y": 0.0, "z": 2.4},
                    "range": 50,
                    "channels": 32,
                    "points_per_second": 320000,
                    "rotation_frequency": 20
                }
            ]
        }
    ]
}
```

---

## 🔬 Lab Exercise: Virtual Drive

### Lab Objectives
1.  **Start CARLA Server:** (In separate terminal/docker).
2.  **Launch Bridge:**
    ```bash
    ros2 launch week21_day145 carla_bridge.launch.py
    ```
3.  **Open Rviz:**
    ```bash
    rviz2
    ```
    -   Set Fixed Frame to `ego_vehicle` or `map`.
    -   Add `Image` topic `/carla/ego_vehicle/rgb_front/image`.
    -   Add `PointCloud2` topic `/carla/ego_vehicle/lidar_top`.
4.  **Drive:**
    -   A PyGame window should appear (Manual Control).
    -   Use WASD to drive.
    -   **Observation:** The Lidar and Camera in Rviz update in real-time as you drive.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Connection Refused
**Symptom:** Bridge keeps saying "Connecting to CARLA...".
**Cause:** CARLA server is not running or port 2000 is blocked.
**Solution:** Check `docker ps`. Ensure firewall allows port 2000-2002.

#### 2. Lag / Low FPS
**Symptom:** Simulation is choppy.
**Cause:** GPU overload. Rviz + CARLA + Bridge is heavy.
**Solution:** Run CARLA in "No Rendering" mode (`-RenderOffScreen`) if you only need sensors. Reduce Lidar points per second.

---

## ⚡ Optimization & Best Practices

### 1. Synchronous Mode
By default, CARLA runs as fast as possible (Async).
-   **Sync Mode:** CARLA waits for a "Tick" from ROS before advancing physics.
-   Essential for Sensor Fusion. Ensures Camera and Lidar are perfectly timestamped.
-   Enable in `carla_ros_bridge` config: `synchronous_mode: true`.

### 2. Scenario Runner
Don't just drive randomly.
-   **OpenSCENARIO:** Define scenarios like "Cut-in", "Pedestrian Crossing".
-   **Scenario Runner:** Executes these scenarios automatically to benchmark your ADAS stack.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the role of `carla-ros-bridge`?
    *   **A:** It acts as a translator between CARLA's internal data format and standard ROS 2 messages.
2.  **Q:** How do we define sensors on the vehicle?
    *   **A:** Using a JSON configuration file passed to the spawn node.
3.  **Q:** Why is simulation important for RL?
    *   **A:** RL requires millions of episodes. You can't crash a real car millions of times. Sim allows fast, safe training.

### Challenge Task
**Task:** Auto-Drive.
1.  Write a simple ROS 2 node that subscribes to `/carla/ego_vehicle/odometry`.
2.  Publish `CarlaEgoVehicleControl` to `/carla/ego_vehicle/vehicle_control_cmd`.
3.  Implement a simple PID to keep speed at 20 km/h.

---

## 📚 Further Reading & References
-   [CARLA Documentation](https://carla.readthedocs.io/)
-   [CARLA ROS Bridge](https://github.com/carla-simulator/ros-bridge)

---

**Day 145 Complete** | Phase 4: ADAS & Robotics Systems | Week 21: System Integration & Capstone
