# Day 141: ROS 2 Lifecycle Management
## Phase 4: ADAS & Robotics Systems | Week 21: System Integration & Capstone

---

> **📝 Day 141 Focus:**
> In a real robot, you can't just run `ros2 run`. What if the Camera starts before the Driver? What if the Planner crashes? **Lifecycle Nodes** (Managed Nodes) solve this by enforcing a strict state machine: Unconfigured $\to$ Inactive $\to$ Active.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the ROS 2 Lifecycle State Machine.
2.  **Implement** a `LifecycleNode` in Python.
3.  **Manage** state transitions (Configure, Activate, Deactivate, Cleanup).
4.  **Control** the node using the `ros2 lifecycle` CLI.
5.  **Design** a launch file that starts nodes in a specific order.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **ROS 2 Basics:** Nodes, Publishers, Subscribers.
-   **State Machines:** FSM concepts.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **ROS 2:** `rclpy`, `lifecycle_msgs`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Problem with Standard Nodes

-   **Standard Node:** Starts doing everything (Publishing, Subscribing) immediately upon `__init__`.
-   **Chaos:** If the Lidar driver isn't ready, the Perception node crashes. If the Parameter Server isn't up, the Planner uses wrong defaults.
-   **Solution:** Deterministic Startup.

### 🔹 Part 2: The Lifecycle State Machine

1.  **Unconfigured:** Node is created but has no configuration (params).
2.  **Inactive:** Node is configured (params loaded) but not doing work (pubs/subs paused).
3.  **Active:** Node is fully functional.
4.  **Finalized:** Node is destroyed.

**Transitions:**
-   `configure()`: Unconfigured $\to$ Inactive. (Load params, create pubs/subs).
-   `activate()`: Inactive $\to$ Active. (Enable pubs).
-   `deactivate()`: Active $\to$ Inactive. (Disable pubs).
-   `cleanup()`: Inactive $\to$ Unconfigured. (Destroy pubs/subs).

---

## 💻 Implementation: Lifecycle Sensor Driver

**Scenario:**
-   A "Camera Driver" node.
-   It should only publish images when in **Active** state.

### 🛠️ Setup
Create `week21_day141` package.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python week21_day141
cd week21_day141/week21_day141
touch lifecycle_camera.py
```

### 👨‍💻 Code: Lifecycle Node

```python
import rclpy
from rclpy.lifecycle import Node, State, TransitionCallbackReturn
from std_msgs.msg import String
import time

class LifecycleCamera(Node):
    def __init__(self):
        super().__init__('lifecycle_camera')
        self.get_logger().info('Node created: Unconfigured')
        self.pub = None
        self.timer = None

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring...')
        # Create Publisher (but don't activate it yet)
        self.pub = self.create_lifecycle_publisher(String, 'camera/image', 10)
        self.get_logger().info('Publisher created')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activating...')
        # Create Timer to simulate publishing
        self.timer = self.create_timer(1.0, self.publish_image)
        # Super() call is crucial for lifecycle publishers!
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating...')
        # Destroy Timer
        if self.timer:
            self.timer.cancel()
            self.destroy_timer(self.timer)
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Cleaning up...')
        # Destroy Publisher
        self.destroy_publisher(self.pub)
        self.pub = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Shutting down...')
        return TransitionCallbackReturn.SUCCESS

    def publish_image(self):
        # Only runs when Active
        msg = String()
        msg.data = f"Image Frame {time.time()}"
        self.pub.publish(msg)
        self.get_logger().info(f"Published: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = LifecycleCamera()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 📄 Launch File (`launch/camera_lifecycle.launch.py`)

```python
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from launch.actions import EmitEvent
from launch.actions import RegisterEventHandler
from launch_ros.event_handlers import OnStateTransition

def generate_launch_description():
    camera_node = LifecycleNode(
        package='week21_day141',
        executable='lifecycle_camera',
        name='lifecycle_camera',
        namespace='',
        output='screen'
    )

    # Event: When node reaches 'inactive', request 'activate'
    # This auto-starts the node sequence
    
    # For manual control, we just launch the node.
    # For auto-start, we need a LifecycleManager (Day 142).
    
    return LaunchDescription([
        camera_node
    ])
```

---

## 🔬 Lab Exercise: Manual Control

### Lab Objectives
1.  Build and Run the node.
    ```bash
    colcon build --packages-select week21_day141
    ros2 run week21_day141 lifecycle_camera
    ```
2.  **Observation:** The node prints "Node created". It does **not** publish.
3.  **Control via CLI:**
    Open a new terminal:
    ```bash
    # Check state
    ros2 lifecycle get /lifecycle_camera
    # Output: unconfigured

    # Configure
    ros2 lifecycle set /lifecycle_camera configure
    # Output: inactive. Node prints "Configuring..."

    # Activate
    ros2 lifecycle set /lifecycle_camera activate
    # Output: active. Node prints "Published: Image Frame..."

    # Deactivate
    ros2 lifecycle set /lifecycle_camera deactivate
    # Output: inactive. Publishing stops.
    ```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Publisher Not Publishing
**Symptom:** Node is active, but `ros2 topic echo` shows nothing.
**Cause:** Did you use `create_lifecycle_publisher`? Standard `create_publisher` is always active. Lifecycle publishers obey the state machine.
**Solution:** Use `self.create_lifecycle_publisher(...)` and ensure `super().on_activate(state)` is called.

#### 2. Transition Failure
**Symptom:** `configure` fails.
**Cause:** Exception in `on_configure`.
**Solution:** Return `TransitionCallbackReturn.FAILURE` or `ERROR`. The node will go to `ErrorProcessing` state.

---

## ⚡ Optimization & Best Practices

### 1. Error Handling
If a sensor fails to open in `on_configure`:
-   Return `FAILURE`.
-   The system manager can then try to restart it or trigger a fallback (e.g., switch to backup camera).

### 2. Bond Connection
For distributed systems.
-   A **Bond** is a heartbeat between the Lifecycle Manager and the Node.
-   If the Node crashes, the Manager knows immediately (Bond broken) and can take action.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between `Unconfigured` and `Inactive`?
    *   **A:** `Unconfigured`: No memory allocated, no params. `Inactive`: Ready to go (params loaded, hardware connected), but paused.
2.  **Q:** Why use `create_lifecycle_publisher`?
    *   **A:** It automatically mutes the publisher when the node is not in the `Active` state, preventing spam/garbage data during startup.
3.  **Q:** Can I use standard subscribers in a Lifecycle Node?
    *   **A:** Yes, but you usually want to destroy/create them in `activate/deactivate` to save bandwidth, or ignore messages in the callback if not active.

### Challenge Task
**Task:** Auto-Start.
1.  Modify the launch file to automatically configure and activate the node.
2.  Use `EmitEvent` with `ChangeState` transition `TRANSITION_CONFIGURE` and `TRANSITION_ACTIVATE`.

---

## 📚 Further Reading & References
-   [ROS 2 Design: Lifecycle](https://design.ros2.org/articles/node_lifecycle.html)
-   [Managed Nodes Demo](https://github.com/ros2/demos/tree/master/lifecycle)

---

**Day 141 Complete** | Phase 4: ADAS & Robotics Systems | Week 21: System Integration & Capstone
