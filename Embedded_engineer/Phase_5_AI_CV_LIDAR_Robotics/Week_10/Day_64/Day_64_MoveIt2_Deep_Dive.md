# Day 64: MoveIt 2 Deep Dive (Architecture & Planners)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 10: Robot Manipulation & Grasping

---

> **📝 Content Creator Instructions:**
> We aren't just moving to a pose anymore. We are planning complex, constrained, high-DOF paths in dynamic environments.
> - **Focus:** Architecture (MoveGroup), Planners (OMPL, PILZ, STOMP), Hybrid Planning, and MoveIt Servo (Real-time).
> - **Code:** A complete C++ Node using `moveit_cpp` API (no MoveGroupInterface python wrappers today, we go deep).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Dissect** the `move_group` node architecture (Monitors, Planners, Checkers).
2.  **Configure** OMPL planners (RRT*, BIT*, PRM) via `ompl_planning.yaml`.
3.  **Implement** Real-Time Servoing (`moveit_servo`) for joystick/teleop control.
4.  **Deploy** a `Hybrid Planning` architecture (Global Planner + Local Solver).
5.  **Debug** "Invalid Start State" and "Motion Plan Failed" errors using the trajectory visualization.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- 6+ DOF Manipulator (Simulation or Real).
- Gamepad (Optional for Servo).

### Software Environment
```bash
sudo apt install ros-humble-moveit
sudo apt install ros-humble-moveit-servo
sudo apt install ros-humble-pilz-industrial-motion-planner
```

### Prior Knowledge
- Day 52 (MoveIt Basics).
- Day 50 (Kinematics).
- C++ Shared Pointers & Inheritance.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The `move_group` Node Architecture

The `move_group` node is the brain. It integrates independent components:
1.  **Planning Pipeline:**
    *   **Motion Planner Adapter:** Pre-processes requests (e.g., Fix Start State).
    *   **Planner:** Generates the path (OMPL, Pilz, CHOMP).
    *   **Trajectory Processing:** Smoothing, Time Parameterization (TOTG - Time Optimal Trajectory Generation).
2.  **Planning Scene Monitor:**
    *   Maintains the "World Model" (Robot State + Octomap + Attached Objects).
    *   Listens to `/joint_states` and Depth Camera buffers.
3.  **Kinematics Plugin:**
    *   Solves IK (KDL, Trac-IK, PickIK, BioIK).

### 🔹 Part 2: OMPL vs. Optimization Planners

*   **Sampling-Based (OMPL):**
    *   *RRTConnect:* Fast, finds *a* solution. Not optimal. Jerky.
    *   *RRT\*:* Asymptotically optimal. Slower.
    *   *BIT\*:* Batch Informed Trees. Best of both worlds.
*   **Optimization-Based (CHOMP/STOMP):**
    *   Starts with a straight line. Deforms it to avoid obstacles.
    *   Good for post-processing OMPL paths.
*   **Deterministic (PILZ):**
    *   LIN (Linear), PTP (Point-to-Point), CIRC (Circular).
    *   Required for Industry (Welding/Gluing) where the path *shape* matters, not just the goal.

### 🔹 Part 3: MoveIt Servo (Real-Time)

Standard planning takes 500ms. Breaking a glass requires reaction in 10ms.
*   **Servo:** Jacobian-based Cartesian control.
*   **Input:** Twist Command ($\dot{x}, \dot{y}, \dot{z}, \omega_x, \omega_y, \omega_z$).
*   **Output:** Joint Velocities ($\dot{q} = J^{-1} \dot{x}$).
*   **Safety:** Collision Checking is done asynchronously. If collision imminent, velocity $\to$ 0.

---

## 💻 Implementation: C++ `moveit_cpp` Interface

We move away from the "Python Interface" (which hides too much) to the direct C++ API used in production.

### 🛠️ Project Structure
```text
day64_moveit_deep/
├── config/
│   └── moveit_cpp.yaml
├── src/
│   └── advanced_motion.cpp
├── launch/
│   └── run_advanced.launch.py
└── CMakeLists.txt
```

### 👨‍💻 Configuration (`config/moveit_cpp.yaml`)

Defines the pipelines.

```yaml
planning_scene_monitor_options:
  name: "planning_scene_monitor"
  robot_description: "robot_description"
  joint_state_topic: "/joint_states"
  attached_collision_object_topic: "/moveit_cpp/planning_scene_monitor"
  publish_planning_scene_topic: "/moveit_cpp/publish_planning_scene"
  wait_for_initial_state_timeout: 10.0

planning_pipelines:
  pipeline_names: ["ompl", "pilz_industrial_motion_planner", "chomp"]

plan_request_params:
  planning_attempts: 1
  planning_pipeline: ompl
  planner_id: "RRTConnectkConfigDefault"
  max_velocity_scaling_factor: 1.0
  max_acceleration_scaling_factor: 1.0
```

### 👨‍💻 Advanced Node (`src/advanced_motion.cpp`)

```cpp
#include <rclcpp/rclcpp.hpp>
#include <moveit/moveit_cpp/moveit_cpp.h>
#include <moveit/moveit_cpp/planning_component.h>
#include <moveit/robot_state/conversions.h>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("moveit_cpp_advanced");

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("advanced_motion_node",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

    // 1. Initialize MoveItCpp
    auto moveit_cpp_ptr = std::make_shared<moveit_cpp::MoveItCpp>(node);
    moveit_cpp_ptr->getPlanningSceneMonitor()->providePlanningSceneService();

    // 2. Planning Component (Lightweight "MoveGroup")
    auto planning_components = std::make_shared<moveit_cpp::PlanningComponent>(
        "panda_arm", moveit_cpp_ptr);

    // 3. Set Goal (State)
    auto robot_model = moveit_cpp_ptr->getRobotModel();
    auto robot_state = moveit_cpp_ptr->getPlanningSceneMonitor()->getStateMonitor()->getCurrentState();
    
    // Define target pose
    geometry_msgs::msg::PoseStamped target_pose;
    target_pose.header.frame_id = "panda_link0";
    target_pose.pose.orientation.w = 1.0;
    target_pose.pose.position.x = 0.28;
    target_pose.pose.position.y = -0.2;
    target_pose.pose.position.z = 0.5;

    planning_components->setGoal(target_pose, "panda_link8");

    // 4. Multi-Pipeline Planning
    // Try OMPL first
    auto plan_solution = planning_components->plan();
    
    if (plan_solution) {
        RCLCPP_INFO(LOGGER, "OMPL Plan Successful");
        planning_components->execute();
    } else {
        RCLCPP_WARN(LOGGER, "OMPL Failed, switching to CHOMP");
        
        // Dynamic Reconfiguration of parameters
        auto params = planning_components->getPlanningPipelineParameters();
        params.planning_pipeline = "chomp";
        // CHOMP usually needs a valid initial guess or it works as optimizer
        // Here we just demontrate switching logic
        
        plan_solution = planning_components->plan(params);
        if(plan_solution) planning_components->execute();
    }

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
```

### 👨‍💻 Real-Time Servo (`launch/servo.launch.py`)

To run servo, we need a separate node that streams calculations.

```python
from launch import LaunchDescription
from launch_ros.actions import Node
import yaml

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def generate_launch_description():
    servo_params = load_yaml('servo_config.yaml')
    
    servo_node = Node(
        package='moveit_servo',
        executable='servo_node_main',
        parameters=[servo_params],
        output='screen'
    )
    
    return LaunchDescription([servo_node])
```

**Servo Config (`servo_config.yaml`):**
```yaml
use_gazebo: false
command_in_type: "speed_units" # m/s
scale:
  linear:  0.6
  rotational: 0.3
  joint: 0.01

incoming_command_topic:  /servo/delta_twist_cmds
status_topic:            /servo/status
```

---

## 🔬 Lab Exercise: The "Welding Seam"

### 1. Lab Objectives
- Use **PILZ Industrial Planner** (`LIN` command).
- **Task:** Trace a perfect square in 3D space.
- **Why?** OMPL (RRTConnect) will wobble between points. PILZ LIN guarantees a straight line in Cartesian space.
- **Code:**
    ```cpp
    planning_components->setPlanningPipelineId("pilz_industrial_motion_planner");
    planning_components->setPlannerId("LIN");
    ```

---

## 🚀 Project: "Hybrid Planning Architecture"

**Goal:** Combine Global Planning (Avoid Labyrinth) + Local Planning (Servo to Target).
1.  **Global Planner:** Finds path around a wall.
2.  **Local Planner:** As the robot executes, the target moves slightly.
3.  **Hybrid Logic:** OMPL generates initial trajectory. `moveit_servo` takes over near the goal for visual servoing alignment.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Fail: Goal is in Collision"
*   **Visually:** Robot looks clear.
*   **Reality:** The padding on the collision checking is too high (default 2cm).
*   **Fix:** Adjust `padding` in `moveit.yaml` or check if `attached_objects` (gripped items) are colliding with self.

#### 2. "Jerky Motion"
*   **Cause:** TOTG (Time Optimal Trajectory Generation) creates bang-bang accleration profiles.
*   **Fix:** Use `TimeParameterization` adapter with `AddTimeParameterization` via `Ruckig` smoothing algorithms (Smoother velocity ramp).

#### 3. "Servo Lag"
*   **Cause:** Low-pass filters on the joint commands.
*   **Fix:** Tune `low_pass_filter_coeff` in servo config. Higher = smoother lay, Lower = responsive jitter.

---

## ⚡ Optimization: Parallel Planning

Why wait for RRT* to converge?
*   **Concept:** Launch 4 planners in parallel threads (RRTConnect, BIT*, PRM, KPIECE).
*   **Race:** First one to find a valid solution wins.
*   **Or:** Wait for 100ms, pick the *shortest* path among all successful plans.
*   **Implementation:** `MoveGroup` supports `ContextManager` for parallel execution.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Difference between `move_group` interfacing and `moveit_cpp`?
    *   **A:** `move_group` is a standalone ROS node you talk to via Actions. `moveit_cpp` allows you to load the MoveIt Core *inside* your own C++ node for lower latency (no serialization overhead).
2.  **Q:** Why use CHOMP?
    *   **A:** It pushes the trajectory away from obstacles using a cost gradient. Good for refining a rough RRT path.
3.  **Q:** When to use Servo?
    *   **A:** Teleoperation, Visual Servoing, or Contact tasks where the goal changes instantly.

### Challenge Task
> **Task:** The Obstacle Course.
> 1. Spawn a dynamic obstacle (swinging pendulum) in Gazebo.
> 2. Use `PlanningSceneMonitor` to update Octomap at 10Hz.
> 3. Plan a path through it.
> 4. If the pendulum enters the path execution, Trigger "Replanning".

---

## 📚 Further Reading
- **MoveIt 2 Tutorials:** "MoveItCpp" section.
- **PickNik Robotics Blog:** " Hybrid Planning".
- **OMPL:** "Geometric vs Control-based planning".

---

**Day 64 Complete**
