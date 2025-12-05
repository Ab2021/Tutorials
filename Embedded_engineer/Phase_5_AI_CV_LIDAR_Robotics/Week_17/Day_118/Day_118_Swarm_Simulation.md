# Day 118: Swarm Simulation (Gazebo/Ignition)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 17: Swarm Robotics

---

> **📝 Content Creator Instructions:**
> Don't melt your CPU.
> - **Focus:** Ignition Gazebo (Gazebo Fortress/Ionic), Entity-Component-System (ECS) architecture, Efficient Multi-Robot Spawning, and maintaining Real-Time Factor (RTF).
> - **Code:** A scalable launch script (`spawn_swarm.py`) that generates $N$ robots in a grid formation within Ignition Gazebo, handling namespaces and TF prefixes correctly.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Migrate** mental models from Gazebo Classic (Monolithic) to Ignition (Client-Server modular).
2.  **Script** the spawning of 20+ robots using `ros_gz_sim`.
3.  **Optimize** physics settings (Step size, Solver iterations) to run swarms on consumer hardware.
4.  **Debug** Namespace collisions in TF trees (The #1 issue in swarm sim).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Multicore CPU (More cores = More robots). RAM > 16GB.

### Software Environment
```bash
sudo apt install ros-humble-ros-gz
# Environment variable for resource paths
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$(pwd)/models
```

### Prior Knowledge
- Launch Files.
- URDF/SDF.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Ignition vs Classic

*   **Classic:** `gazebo_ros_pkgs`. Sim logic and Rendering in one process. Hard to scale.
*   **Ignition (Gazebo Sim):**
    *   **Server:** Runs physics (Headless).
    *   **Gui:** Optional 3D viewer.
    *   **Transport:** Msg interface (Ign Transport).
    *   **Architecture:** ECS (Entity Component System). Faster, more cache-friendly.

### 🔹 Part 2: The Namespace Problem

When you spawn 10 robots:
*   Default: Everyone publishes to `/scan` and `base_link`.
*   Result: Chaos. Rviz shows 10 robots flickering at the center.
*   **Fix:**
    *   Robot 1: Topic `/robot_1/scan`, Frame `robot_1/base_link`.
    *   **Warning:** The URDF itself doesn't know about namespaces. You must use `frame_prefix` in `robot_state_publisher`.

### 🔹 Part 3: Level of Detail (LOD)

To simulate 50 robots:
1.  **Physics:** Use simplified colliders (Box instead of Mesh).
2.  **Sensors:** Lower update rate (10Hz vs 60Hz). Reduce Lidar resolution.
3.  **Rendering:** Don't render cameras if you aren't looking at them (`headless` mode).

---

## 💻 Implementation: Swarm Spawner

We will create a launch file that spawns a grid of Diff Drive robots.

### 🛠️ Project Structure
```text
day118_swarm_sim/
├── launch/
│   └── swarm.launch.py
├── models/
│   └── simple_robot/
│       ├── model.sdf
│       └── model.config
└── urdf/
    └── diff_drive.xacro
```

### 👨‍💻 Launch Script (`launch/swarm.launch.py`)

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    
    # 1. Start Gazebo Server (Empty World)
    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    # 2. Start Gazebo GUI (Optional)
    gz_client = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-g'] 
    )

    # 3. Spawn Robots Loop
    robots = []
    
    # Grid Config
    rows = 3
    cols = 3
    spacing = 2.0
    
    for i in range(rows):
        for j in range(cols):
            bot_id = i * cols + j
            name = f"bot_{bot_id}"
            x_pos = i * spacing
            y_pos = j * spacing
            
            # Spawn Group
            spawn_action = GroupAction([
                PushRosNamespace(name),
                
                # A. Robot State Publisher (Publishes TF with prefix)
                # Note: 'frame_prefix' argument adds prefix to all links
                Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    name='robot_state_publisher',
                    parameters=[{
                        'robot_description': '<robot>...</robot>', # Load Xacro here
                        'frame_prefix': f"{name}/" 
                    }]
                ),
                
                # B. Spawn Entity in Ignition
                Node(
                    package='ros_gz_sim',
                    executable='create',
                    arguments=[
                        '-name', name,
                        '-topic', 'robot_description', # Read from local NSP param
                        '-x', str(x_pos),
                        '-y', str(y_pos),
                        '-z', '0.1'
                    ],
                    output='screen'
                ),
                
                # C. Bridge (Gz <-> ROS)
                # Bridge /cmd_vel, /odom, /scan
                Node(
                    package='ros_gz_bridge',
                    executable='parameter_bridge',
                    arguments=[
                        f"/model/{name}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
                        f"/model/{name}/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
                        # Remap Gz topic to ROS topic
                    ],
                    remappings=[
                        (f"/model/{name}/cmd_vel", f"/{name}/cmd_vel"),
                        (f"/model/{name}/odometry", f"/{name}/odom")
                    ]
                )
            ])
            robots.append(spawn_action)
            
    return LaunchDescription([
        gz_server,
        # gz_client,
        *robots
    ])
```

---

## 🔬 Lab Exercise: "Stress Test"

### 1. Lab Objectives
- **Run:** `ros2 launch day118_swarm_sim swarm.launch.py`.
- **Observe:** 9 Robots appear in a grid.
- **Check TF:** `ros2 run tf2_tools view_frames`.
    *   Verify `bot_0/base_link`, `bot_1/base_link` exist.
- **Perf:** Check `RTF` (Real Time Factor) in Gazebo bottom bar.
    *   If RTF < 0.8, reduce robot count or physics steps.
- **Command:** `ros2 topic pub /bot_4/cmd_vel ...`. Only the middle robot should move.

---

## 🚀 Project: "The Beehive"

**Goal:** Centralized Visualization, Decentralized Physics.
1.  **Sim:** 20 Robots. Headless Gazebo.
2.  **Bridge:** Only bridge `Pose` (cheap), not `Lidar` (expensive) for all.
    *   Bridge Lidar only for `bot_0` (The Scout).
3.  **Viz:** Single Rviz instance showing positions of all 20 robots (Markers).
4.  **Behavior:** Boids algorithm (Day 115) running on all 20.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Robots explode on spawn"
*   **Cause:** Spawning inside the ground or inside each other.
*   **Fix:** Adjust Initial Z ($z=0.2$). Check spacing.

#### 2. "Services Warning"
*   **Log:** "Service /spawn_entity already exists".
*   **Cause:** Race condition in launch system.
*   **Fix:** Add `TimerAction` to stagger spawns. Spawn Robot 1 at $t=1s$, Robot 2 at $t=2s$.

---

## ⚡ Optimization: Hardware Acceleration

Ignition supports running physics on GPU (experimental) or specific sensors on GPU.
*   **Lidar:** `gpu_ray` sensor. Much faster than CPU raycasting.
*   **Camera:** Always uses GPU.
*   **Tip:** If headless, ensure you have a virtual display (`xvfb`) or EGL setup for rendering sensors.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why use `frame_prefix`?
    *   **A:** If we don't, every robot publishes a transform `base_link` -> `wheel`. TF tree cannot handle multiple transforms with same name. We need `bot1/base_link` and `bot2/base_link`.
2.  **Q:** What is the bridge?
    *   **A:** `ros_gz_bridge`. It translates Protobuf (Ignition) messages to ROS 2 (DDS) messages.
3.  **Q:** Max robots on a laptop?
    *   **A:** Typically 10-20 simple diff drives. 1-2 Humanoids.

### Challenge Task
> **Task:** Follow the Leader.
> 1. Spawn 10 robots.
> 2. Teleop `bot_0`.
> 3. Bots 1-9 subscribe to `bot_0/odom` and follow.

---

## 📚 Further Reading
- **Gazebo Sim Docs:** "Migration from Classic".
- **ROS 2 Integration:** `ros_gz` package documentation.

---

**Day 118 Complete**
