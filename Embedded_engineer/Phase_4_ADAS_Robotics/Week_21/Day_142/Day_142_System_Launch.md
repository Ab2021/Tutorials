# Day 142: System Launch & Configuration
## Phase 4: ADAS & Robotics Systems | Week 21: System Integration & Capstone

---

> **📝 Day 142 Focus:**
> A robot is a symphony of nodes. You need a conductor. **ROS 2 Launch** is that conductor. It starts nodes, loads parameters, and manages dependencies. We will build a **Master Launch File** to bring up the entire ADAS stack.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Write** complex Python Launch files (Includes, Groups, Conditions).
2.  **Manage** parameters using YAML files.
3.  **Implement** a `LifecycleManager` to sequence node startup.
4.  **Use** `ComponentContainer` for efficient intra-process communication.
5.  **Debug** launch issues using `rqt_graph` and `ros2 param`.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 141:** Lifecycle Nodes.
-   **Python:** `launch` API.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **ROS 2:** `launch`, `nav2_lifecycle_manager`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Launch System

-   **Python-based:** Dynamic logic (e.g., "If simulation, load sim_params.yaml").
-   **Include:** Reuse launch files from other packages (`IncludeLaunchDescription`).
-   **Namespaces:** Group nodes to avoid name collisions (`PushRosNamespace`).

### 🔹 Part 2: Parameter Management

-   **YAML:** Standard format for config.
    ```yaml
    /planner_node:
      ros__parameters:
        max_speed: 10.0
        planner_type: "AStar"
    ```
-   **Wildcards:** `/**` to apply to all nodes (careful!).

### 🔹 Part 3: The Lifecycle Manager

The **Nav2 Lifecycle Manager** is a standard tool to manage a list of nodes.
1.  **Startup:** Configure A -> Configure B -> Activate A -> Activate B.
2.  **Shutdown:** Deactivate B -> Deactivate A -> Cleanup B -> Cleanup A.
3.  **Health Check:** Monitors nodes and triggers recovery if one dies.

---

## 💻 Implementation: The Master Launch

**Scenario:**
-   Stack: `MapServer` -> `Localization` -> `Planner` -> `Controller`.
-   Requirement: Start in order.

### 🛠️ Setup
Create `week21_day142` package.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python week21_day142
mkdir -p week21_day142/launch
mkdir -p week21_day142/config
```

### 📄 Config (`config/adas_params.yaml`)

```yaml
lifecycle_manager:
  ros__parameters:
    autostart: true
    node_names: ['map_server', 'planner', 'controller']

map_server:
  ros__parameters:
    map_file: "/tmp/my_map.yaml"

planner:
  ros__parameters:
    algorithm: "AStar"
    max_vel: 5.0

controller:
  ros__parameters:
    kp: 1.0
    ki: 0.1
    kd: 0.05
```

### 👨‍💻 Code: Master Launch File (`launch/system.launch.py`)

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    params_file = os.path.join(
        get_package_share_directory('week21_day142'),
        'config',
        'adas_params.yaml'
    )

    # 2. Nodes (Mocking the stack with simple Lifecycle Nodes)
    # In reality, these would be your actual package executables
    
    # We use 'lifecycle_camera' from Day 141 as a placeholder for all 3 nodes
    # just to demonstrate the manager.
    # You must have built week21_day141 for this to work.
    
    map_server = Node(
        package='week21_day141', 
        executable='lifecycle_camera',
        name='map_server',
        output='screen',
        parameters=[params_file]
    )
    
    planner = Node(
        package='week21_day141',
        executable='lifecycle_camera',
        name='planner',
        output='screen',
        parameters=[params_file]
    )
    
    controller = Node(
        package='week21_day141',
        executable='lifecycle_camera',
        name='controller',
        output='screen',
        parameters=[params_file]
    )

    # 3. Lifecycle Manager
    # This node orchestrates the others
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[params_file]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),
            
        map_server,
        planner,
        controller,
        lifecycle_manager
    ])
```

---

## 🔬 Lab Exercise: The Orchestration

### Lab Objectives
1.  Install Nav2 Lifecycle Manager:
    ```bash
    sudo apt install ros-humble-nav2-lifecycle-manager
    ```
2.  Build and Run:
    ```bash
    colcon build
    ros2 launch week21_day142 system.launch.py
    ```
3.  **Observation:**
    -   The `lifecycle_manager` starts.
    -   It prints "Configuring map_server...".
    -   It prints "Configuring planner...".
    -   It prints "Activating map_server...".
    -   All nodes transition to Active automatically.
4.  **Experiment:**
    -   Kill the `map_server` (Ctrl+C isn't easy here, try `ros2 lifecycle set /map_server deactivate`).
    -   The Manager might detect it (if Bond is used) or just report state mismatch.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Parameter Not Loading
**Symptom:** Node starts but uses default values.
**Cause:** YAML file path is wrong or node name in YAML doesn't match node name in Launch.
**Solution:** Check `ros2 param list` and `ros2 param get /node_name param_name`.

#### 2. Manager Stuck
**Symptom:** Manager hangs at "Configuring...".
**Cause:** One of the managed nodes failed to transition (returned FAILURE).
**Solution:** Check the logs of the individual nodes.

---

## ⚡ Optimization & Best Practices

### 1. Composition (Component Container)
Launching 10 nodes = 10 processes = High CPU/Memory overhead (Context Switching).
-   **Composition:** Load multiple nodes (Components) into a **Single Process**.
-   **Zero-Copy:** Messages between components in the same process are passed by pointer (no serialization).
-   **Launch:**
    ```python
    container = ComposableNodeContainer(
        name='my_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(package='pkg', plugin='MyNode', name='node1'),
            ComposableNode(package='pkg', plugin='MyNode', name='node2')
        ]
    )
    ```

### 2. Opaque Functions
For very dynamic launch logic (e.g., parsing a file to decide what nodes to launch), use `OpaqueFunction`.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the order of operations in `nav2_lifecycle_manager`?
    *   **A:** Configure All -> Activate All. (Usually).
2.  **Q:** How do I pass a parameter from command line?
    *   **A:** `ros2 launch pkg file.launch.py param:=value`. In Python: `LaunchConfiguration('param')`.
3.  **Q:** Why use `get_package_share_directory`?
    *   **A:** Because installed packages are in `/opt/ros/...` or `install/...`, not your source folder. This finds the correct path at runtime.

### Challenge Task
**Task:** Conditional Launch.
1.  Add an argument `enable_rviz`.
2.  Use `Condition=IfCondition(LaunchConfiguration('enable_rviz'))` to launch RViz only when requested.

---

## 📚 Further Reading & References
-   [ROS 2 Launch Tutorials](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Launch-Main.html)
-   [Nav2 Lifecycle Manager](https://navigation.ros.org/configuration/packages/configuring-lifecycle.html)

---

**Day 142 Complete** | Phase 4: ADAS & Robotics Systems | Week 21: System Integration & Capstone
