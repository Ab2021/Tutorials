# Day 4: Launch Files, Parameters, and Configuration
## Phase 4: ADAS & Robotics Systems | Week 1: ROS 2 Fundamentals

---

> **📝 Day 4 Focus:**
> In production ADAS systems, you never run nodes manually with `ros2 run`. You use the **Launch System** to orchestrate complex startups, manage dependencies, and handle failures. We will also master **Parameters** for runtime configuration, enabling dynamic tuning of algorithms without recompilation.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Architect** complex launch systems using Python-based launch files (`.launch.py`) with conditional logic and event handlers.
2.  **Implement** dynamic parameter handling in C++ nodes, including validation callbacks and runtime reconfiguration.
3.  **Manage** large-scale configuration using YAML files and nested parameter structures.
4.  **Orchestrate** multi-node ADAS pipelines (e.g., Perception -> Planning -> Control) with dependency management.
5.  **Debug** launch issues and parameter conflicts using introspection tools.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Python 3:** Strong understanding of Python for launch files.
-   **ROS 2 Nodes:** Understanding of node lifecycle (Day 1-3).
-   **YAML:** Syntax for configuration files.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS with ROS 2 Humble.

### Software Stack
-   **ROS 2 Humble**
-   **Launch Library:** `launch`, `launch_ros`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The ROS 2 Launch System

#### 1.1 Why Python? (vs XML in ROS 1)

ROS 1 used XML (`.launch`). It was declarative but limited.
ROS 2 uses Python (`.launch.py`). It is **imperative** and **programmable**.

**Advantages for ADAS:**
-   **Logic:** "If camera fails, launch simulation."
-   **Introspection:** Query system state before launching.
-   **Modularity:** Use Python classes/functions to generate launch descriptions.
-   **Events:** React to node crashes (e.g., respawn or trigger emergency stop).

#### 1.2 Core Concepts

1.  **LaunchDescription:** The container that holds all actions.
2.  **Action:** Something to do (Run a node, include another file, set a variable).
3.  **Substitution:** A value evaluated at runtime (e.g., `FindPackageShare`, `LaunchConfiguration`).
4.  **Condition:** Predicate to decide if an action runs (`IfCondition`, `UnlessCondition`).
5.  **Event Handler:** Reacts to system events (e.g., `OnProcessExit`).

#### 1.3 Substitutions Deep Dive

Substitutions are lazy-evaluated strings.

-   `LaunchConfiguration('param_name')`: Gets a value passed via command line (`param_name:=value`).
-   `PathJoinSubstitution([root, 'sub', 'file'])`: Cross-platform path joining.
-   `FindPackageShare('pkg_name')`: Finds the `share` directory of a package.
-   `Command(['xacro ', xacro_file])`: Executes a shell command and uses the output (e.g., generating URDF).

#### 1.4 Event-Driven Launching

In ADAS, startup order matters.
*   "Don't start the Planner until the Localizer is ready."
*   "If the Driver crashes, respawn it."

**Lifecycle Integration:**
You can trigger transitions based on events.
-   `OnProcessStart`: When a process starts.
-   `OnProcessExit`: When a process dies.
-   `RegisterEventHandler`: The mechanism to hook into these.

---

### 🔹 Part 2: Parameters & Configuration

#### 2.1 The Parameter Server Model

In ROS 1, there was a global Parameter Server.
In ROS 2, **Parameters are Node-Local**.

-   Each node maintains its own parameters.
-   Nodes can query other nodes' parameters (via services).
-   **Global Parameters?** No. You use a dedicated "Global Parameter Node" if needed, or pass the same YAML to everyone.

#### 2.2 Parameter Types

-   `bool`
-   `integer` (int64)
-   `double` (float64)
-   `string`
-   `byte_array`
-   `bool_array`
-   `integer_array`
-   `double_array`
-   `string_array`

#### 2.3 Dynamic Reconfiguration

ROS 2 allows changing parameters at runtime without restarting the node.
-   **Callback:** `add_on_set_parameters_callback`.
-   **Validation:** You can reject invalid changes (e.g., "Speed limit cannot be negative").

#### 2.4 YAML Configuration

YAML files allow bulk loading of parameters.

```yaml
/perception_node:
  ros__parameters:
    camera_topic: "/camera/image_raw"
    confidence_threshold: 0.75
    class_labels: ["car", "pedestrian", "cyclist"]
```

**Wildcards:**
```yaml
/**:
  ros__parameters:
    use_sim_time: true
```
Applies to all nodes in the namespace.

---

### 🔹 Part 3: Advanced Launch Patterns

#### 3.1 Composition (Components)

Launching nodes as **Components** in a **Component Container**.
-   Reduces overhead (shared process).
-   Enables zero-copy transport.
-   Launch file syntax changes: `LoadComposableNodes` instead of `Node`.

#### 3.2 Opaque Functions

Sometimes you need to run arbitrary Python code to calculate launch arguments *before* the graph is built.
-   `OpaqueFunction`: Allows executing a Python function that returns a list of Actions.

---

## 💻 Implementation: ADAS Launch System

We will build a robust launch system for a "Perception Stack" that includes:
1.  **Sensor Driver** (Simulated).
2.  **Object Detector** (Processing).
3.  **Visualizer** (RViz).
4.  **Parameter Management**.

### 🛠️ Package Setup

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake \
  --dependencies rclcpp std_msgs sensor_msgs \
  --node-name configurable_node \
  adas_launch_config

cd adas_launch_config
mkdir launch config
```

### 📦 Configuration Files

Create `config/perception_params.yaml`:

```yaml
/lidar_driver:
  ros__parameters:
    scan_frequency: 10.0
    port: "/dev/ttyUSB0"
    frame_id: "lidar_link"

/object_detector:
  ros__parameters:
    model_path: "/models/yolo_v5.onnx"
    confidence_threshold: 0.5
    nms_threshold: 0.45
    classes: ["vehicle", "pedestrian", "cyclist", "traffic_sign"]
    debug_mode: true
```

### 👨‍💻 Node Implementation (Parameter Handling)

#### Header: `include/adas_launch_config/detector.hpp`

```cpp
#ifndef ADAS_LAUNCH_CONFIG__DETECTOR_HPP_
#define ADAS_LAUNCH_CONFIG__DETECTOR_HPP_

#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <string>

namespace adas_launch_config
{

class DetectorNode : public rclcpp::Node
{
public:
  explicit DetectorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  // Parameter Callback
  rcl_interfaces::msg::SetParametersResult parameters_callback(
    const std::vector<rclcpp::Parameter> & parameters);

  // Internal State
  double confidence_threshold_;
  std::vector<std::string> classes_;
  bool debug_mode_;

  // Handler
  OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;
};

}  // namespace adas_launch_config

#endif  // ADAS_LAUNCH_CONFIG__DETECTOR_HPP_
```

#### Source: `src/detector.cpp`

```cpp
#include "adas_launch_config/detector.hpp"

namespace adas_launch_config
{

DetectorNode::DetectorNode(const rclcpp::NodeOptions & options)
: Node("object_detector", options)
{
  // 1. Declare Parameters (with default values)
  this->declare_parameter("model_path", "default.onnx");
  this->declare_parameter("confidence_threshold", 0.5);
  this->declare_parameter("nms_threshold", 0.4);
  this->declare_parameter("classes", std::vector<std::string>{"car"});
  this->declare_parameter("debug_mode", false);

  // 2. Get Initial Values
  confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
  classes_ = this->get_parameter("classes").as_string_array();
  debug_mode_ = this->get_parameter("debug_mode").as_bool();

  RCLCPP_INFO(this->get_logger(), "Detector Initialized. Conf: %.2f, Debug: %s",
    confidence_threshold_, debug_mode_ ? "ON" : "OFF");

  // 3. Register Callback for Dynamic Updates
  params_callback_handle_ = this->add_on_set_parameters_callback(
    std::bind(&DetectorNode::parameters_callback, this, std::placeholders::_1));
}

rcl_interfaces::msg::SetParametersResult DetectorNode::parameters_callback(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  for (const auto & param : parameters) {
    if (param.get_name() == "confidence_threshold") {
      double val = param.as_double();
      if (val < 0.0 || val > 1.0) {
        result.successful = false;
        result.reason = "Confidence must be between 0.0 and 1.0";
        RCLCPP_WARN(this->get_logger(), "Rejected confidence: %.2f", val);
      } else {
        confidence_threshold_ = val;
        RCLCPP_INFO(this->get_logger(), "Updated confidence: %.2f", val);
      }
    }
    else if (param.get_name() == "debug_mode") {
      debug_mode_ = param.as_bool();
      RCLCPP_INFO(this->get_logger(), "Updated debug mode: %s", 
        debug_mode_ ? "ON" : "OFF");
    }
  }

  return result;
}

}  // namespace adas_launch_config

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(adas_launch_config::DetectorNode)
```

### 🚀 Launch File Implementation

Create `launch/perception_stack.launch.py`:

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('adas_launch_config')
    
    # 1. Launch Arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug mode'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(pkg_share, 'config', 'perception_params.yaml'),
        description='Path to config file'
    )

    # 2. Nodes
    
    # Simulated LiDAR Driver (using dummy node for example)
    lidar_node = Node(
        package='adas_launch_config', # Assuming we have a dummy driver here
        executable='configurable_node', # Reusing the same executable for demo
        name='lidar_driver',
        parameters=[
            LaunchConfiguration('config_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        remappings=[
            ('/scan', '/lidar/scan')
        ]
    )

    # Object Detector
    detector_node = Node(
        package='adas_launch_config',
        executable='configurable_node',
        name='object_detector',
        parameters=[
            LaunchConfiguration('config_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'debug_mode': LaunchConfiguration('debug')} # Override config with arg
        ],
        output='screen'
    )

    # RViz (Conditional)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_share, 'config', 'view.rviz')],
        condition=IfCondition(LaunchConfiguration('debug'))
    )

    # 3. Event Handling (Example)
    # If detector crashes, log an error (or respawn)
    # respawn_detector = RegisterEventHandler(
    #     OnProcessExit(
    #         target_action=detector_node,
    #         on_exit=[
    #             LogInfo(msg='Detector crashed! Respawning...'),
    #             detector_node # This would restart it
    #         ]
    #     )
    # )

    return LaunchDescription([
        use_sim_time_arg,
        debug_arg,
        config_file_arg,
        lidar_node,
        detector_node,
        rviz_node
    ])
```

---

## 🔬 Lab Exercise: Dynamic Tuning

### Lab Objectives
1.  Launch the `perception_stack`.
2.  Use `ros2 param` CLI to modify thresholds at runtime.
3.  Observe the validation logic rejecting invalid values.
4.  Create a launch file that includes another launch file.

### Part 1: CLI Interaction

```bash
# 1. Build & Launch
colcon build
source install/setup.bash
ros2 launch adas_launch_config perception_stack.launch.py debug:=true

# 2. List Parameters
ros2 param list

# 3. Get Value
ros2 param get /object_detector confidence_threshold

# 4. Set Valid Value
ros2 param set /object_detector confidence_threshold 0.8
# Output: Set parameter successful

# 5. Set Invalid Value
ros2 param set /object_detector confidence_threshold 1.5
# Output: Set parameter failed: Confidence must be between 0.0 and 1.0
```

### Part 2: Include Launch

Create `launch/system.launch.py`:

```python
def generate_launch_description():
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('adas_launch_config'),
                'launch',
                'perception_stack.launch.py'
            ])
        ]),
        launch_arguments={'debug': 'true'}.items()
    )
    
    return LaunchDescription([
        perception_launch
    ])
```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Parameter Not Found
**Symptom:** `ros2 param get` fails.
**Cause:** Parameter not declared in the node constructor.
**Solution:** Always call `declare_parameter` for every parameter you expect to use.

#### 2. Launch File Syntax Error
**Symptom:** `ros2 launch` fails with Python traceback.
**Cause:** Python syntax error or incorrect import.
**Solution:** Run `python3 your_launch_file.py` directly to check for syntax errors (though it won't run ROS logic, it checks parsing).

#### 3. YAML Loading Issues
**Symptom:** Parameters remain default despite YAML.
**Cause:**
-   Node name in YAML (`/object_detector`) doesn't match node name in Launch (`name='object_detector'`).
-   Namespace mismatch.
**Solution:** Ensure the hierarchy in YAML matches the runtime node names exactly.

### Introspection Tools

**1. `ros2 param dump`**
Save current configuration to file.
```bash
ros2 param dump /object_detector > current_config.yaml
```

**2. `rqt_reconfigure`**
GUI for tuning parameters (if `rqt` plugin installed).
```bash
ros2 run rqt_reconfigure rqt_reconfigure
```

---

## ⚡ Optimization & Best Practices

### 1. Parameter Declarations
-   Use `declare_parameter` with a sensible default.
-   Use `ParameterDescriptor` to add metadata (description, range constraints).
    ```cpp
    auto desc = rcl_interfaces::msg::ParameterDescriptor();
    desc.description = "Detection confidence threshold";
    desc.floating_point_range.resize(1);
    desc.floating_point_range[0].from_value = 0.0;
    desc.floating_point_range[0].to_value = 1.0;
    this->declare_parameter("confidence", 0.5, desc);
    ```
    *This allows tools like `rqt` to show sliders!*

### 2. Launch File Organization
-   Keep launch files granular (one per subsystem).
-   Use a top-level `main.launch.py` to compose them.
-   Use `GroupAction` and `PushRosNamespace` to isolate subsystems.

### 3. Respawn
-   For production, use `respawn=True` in the `Node` action.
    ```python
    Node(..., respawn=True, respawn_delay=2.0)
    ```
    *Note: Only available in newer ROS 2 versions (Humble+).*

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** How do you pass a command-line argument to a launch file?
    *   **A:** `ros2 launch pkg file.launch.py arg_name:=value`.

2.  **Q:** What is the difference between `LaunchConfiguration` and a Python variable?
    *   **A:** `LaunchConfiguration` is a substitution evaluated at *launch time* (runtime), while a Python variable is evaluated at *generation time* (parsing).

3.  **Q:** Why is `add_on_set_parameters_callback` important?
    *   **A:** It allows the node to validate parameter changes and react to them (e.g., updating internal state) without restarting.

### Challenge Task
**Task:** Create a "Safe Mode" Parameter.
1.  Add a `safe_mode` (bool) parameter.
2.  If `safe_mode` is set to `true` at runtime, the node should immediately publish a log message "ENTERING SAFE MODE" and perhaps stop processing data.
3.  Launch the node and toggle the parameter via CLI.

---

## 📚 Further Reading & References
-   [ROS 2 Launch Tutorials](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Launch-Main.html)
-   [Using Parameters in a Class (C++)](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Using-Parameters-In-A-Class-CPP.html)
-   [Launch Architecture](https://design.ros2.org/articles/ros_launch.html)

---

**Day 4 Complete** | Phase 4: ADAS & Robotics Systems | Week 1: ROS 2 Fundamentals
