# Day 71: Nav2 Architecture Deep Dive
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 11: Navigation 2 (Nav2) Mastery

---

> **📝 Content Creator Instructions:**
> "Move Base" is dead. Long live Nav2.
> - **Focus:** Lifecycle Nodes (Managed Nodes), Behavior Trees as the orchestrator, and the Server-Client architecture (Planner Server, Controller Server).
> - **Code:** A custom Lifecycle Node that integrates into the Nav2 manager logic.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deconstruct** the Nav2 Stack: BT Navigator, Planner Server, Controller Server, Recoveries.
2.  **Implement** a ROS 2 Lifecycle Node (Unconfigured $\to$ Inactive $\to$ Active).
3.  **Configure** DDS (Data Distribution Service) for high-reliability navigation traffic (FastDDS/CycloneDDS).
4.  **Trace** a navigation goal execution flow through the architecture.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Mobile Robot (Simulation or Real) with Lidar.

### Software Environment
```bash
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
sudo apt install ros-humble-rmw-cyclonedds-cpp
```

### Prior Knowledge
- ROS 2 Actions.
- Finite State Machines.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Monolith vs Microservices

*   **MoveBase (ROS 1):** A single giant C++ class. Hard to modify.
*   **Nav2 (ROS 2):** A collection of Action Servers orchestrated by a Behavior Tree.
    *   **Planner Server:** "Calculate Path" (ComputePathToPose).
    *   **Controller Server:** "Follow Path" (FollowPath).
    *   **Smoother Server:** "Smooth Path".
    *   **Behavior Server:** "Spin", "BackUp" (Recovery).
    *   **BT Navigator:** The "Boss". Calls these servers in sequence based on XML logic.

### 🔹 Part 2: Lifecycle Managed Nodes

Robots shouldn't Just Start. They should Boot Up carefully.
*   **Unconfigured:** Node created. Memory allocated.
*   **Configure()**: Load ROS parameters, setup Pub/Sub. (Transition to Inactive).
*   **Activate()**: Enable Pub/Sub. Start processing. (Transition to Active).
*   **Deactivate()**: Pause processing.
*   **Cleanup()**: Destruct.
*   *Nav2 uses `nav2_lifecycle_manager` to boot all servers in strict order (Map -> AMCL -> Planner -> Controller).*

### 🔹 Part 3: DDS Tuning

Nav2 is heavy on traffic (Costmaps, PointClouds).
*   **Issue:** Default DDS (UDP Multicast) drops large packets on weak WiFi.
*   **Fix:** `cyclonedds.xml`. Increase socket buffer size. Use `BEST_EFFORT` for Lidar, `RELIABLE` for Global Plan.

---

## 💻 Implementation: Custom Lifecycle Node

We will write a `SecurityScanner` node that acts as a Lifecycle node. It only scans when Active.

### 🛠️ Project Structure
```text
day71_nav2_arch/
├── src/
│   └── lifecycle_scanner.cpp
├── launch/
│   └── managed_scan.launch.py
└── CMakeLists.txt
```

### 👨‍💻 Lifecycle Node (`src/lifecycle_scanner.cpp`)

```cpp
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class SecurityScanner : public rclcpp_lifecycle::LifecycleNode {
public:
    SecurityScanner() : LifecycleNode("security_scanner") {}

    // 1. Configure: Load Parameters
    CallbackReturn on_configure(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(get_logger(), "Configuring...");
        pub_scan_ = this->create_publisher<sensor_msgs::msg::LaserScan>("scan_filtered", 10);
        sub_scan_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", 10, std::bind(&SecurityScanner::scanCb, this, std::placeholders::_1));
        return CallbackReturn::SUCCESS;
    }

    // 2. Activate: Enable Publishing
    CallbackReturn on_activate(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(get_logger(), "Activating...");
        pub_scan_->on_activate(); // Crucial!
        return CallbackReturn::SUCCESS;
    }

    // 3. Deactivate: Disable Publishing
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(get_logger(), "Deactivating...");
        pub_scan_->on_deactivate();
        return CallbackReturn::SUCCESS;
    }

    // 4. Cleanup: Free Memory
    CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) {
        pub_scan_.reset();
        return CallbackReturn::SUCCESS;
    }

    void scanCb(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
            return; // Don't process if inactive
        }
        // Passthrough processing
        pub_scan_->publish(*msg);
    }

private:
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_scan_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SecurityScanner>()->get_node_base_interface());
    rclcpp::shutdown();
    return 0;
}
```

### 👨‍💻 Management Launch (`launch/managed_scan.launch.py`)

Using `LifecycleManager` (or manual transition).

```python
from launch import LaunchDescription
from launch_ros.actions import Node, LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessStart

def generate_launch_description():
    scanner_node = LifecycleNode(
        package='day71_nav2_arch',
        executable='lifecycle_scanner',
        name='security_scanner',
        namespace=''
    )

    # Automatically transition to Configure, then Activate
    to_configure = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=lambda n: n == scanner_node,
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )

    to_activate = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=lambda n: n == scanner_node,
            transition_id=Transition.TRANSITION_ACTIVATE,
        )
    )

    return LaunchDescription([
        scanner_node,
        RegisterEventHandler(
            OnProcessStart(target_action=scanner_node, on_start=[to_configure]),
        ),
        # In tracking real production, we wait for 'on_configure' success before activating.
        # This is simplified.
    ])
```

---

## 🔬 Lab Exercise: "The Manager"

### 1. Lab Objectives
- Launch the `nav2_bringup` simulation (`tb3_simulation_launch.py`).
- Run `ros2 lifecycle list`.
- **Task:** Manually kill the `planner_server` (transition to Finalized).
- **Observation:** The Navigation stack stops working.
- **Task:** Restart and re-configure just the Planner Server using `ros2 lifecycle set`.
- **Goal:** Understand "Runtime Recovery" without restarting the whole robot.

---

## 🚀 Project: "Watchdog"

**Goal:** Create a Node monitoring node health.
1.  **Monitor:** Check `/bond` topics (Nav2 nodes publish heartbeats).
2.  **Logic:** If `controller_server` misses 5 heartbeats (Crash):
3.  **Action:** E-Stop the robot (Vel=0) and trigger a `Reset` transition on the Controller Server.
4.  **Implementation:** Simple Python script subscribing to `bond`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Action Server Not Available"
*   **Symptom:** BT Navigator prints `Action server not available`.
*   **Cause:** The relevant node (Planner/Controller) crashed or is in `UNCONFIGURED` state.
*   **Fix:** Check `ros2 lifecycle get /planner_server`. Must be `active`.

#### 2. "Costmap 100% CPU"
*   **Symptom:** Robot freezes.
*   **Cause:** Raytracing is expensive. Updating costmap at 50Hz is overkill.
*   **Fix:** Reduce `update_frequency` to 5Hz in `nav2_params.yaml` for Global Costmap.

---

## ⚡ Optimization: Costmap 2D vs Voxel

*   **Costmap2D:** Projects 3D world to 2D grid. Fast. Loses "Overhang" info (Table top is obstacle, but Space under table is free).
*   **Voxel Grid:** Maintains 3D voxels.
*   **Tip:** Use `VoxelLayer` for local costmap (collision avoidance) and `StaticLayer` (2D) for global planner (efficiency).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Role of BT Navigator?
    *   **A:** It parses the XML behavior tree (Strategy) and calls the Action Servers (Tactics).
2.  **Q:** Why separate Planner and Controller?
    *   **A:** Modularity. You can swap Dijkstra (Planner) for RRT* without changing the DWB (Controller).
3.  **Q:** What is "Recovery"?
    *   **A:** An action taken when the robot is stuck. e.g., "Spin" to clear costmap, or "Back up" to unstuck wheels.

### Challenge Task
> **Task:** Dual-Planner Setup.
> 1. Setup Nav2 to load TWO planners: `GridBased` and `StraightLine`.
> 2. In Behavior Tree, check: If `Distance < 2m`, use `StraightLine`. Else use `GridBased`.
> 3. This reduces CPU usage for short movements.

---

## 📚 Further Reading
- **Nav2 Documentation:** "Concepts" section.
- **ROS 2 Design:** "Lifecycle Nodes".

---

**Day 71 Complete**
