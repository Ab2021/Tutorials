# Day 102: CI/CD & Automated Testing
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 15: Production-Grade ROS 2

---

> **📝 Content Creator Instructions:**
> Don't merge broken code.
> - **Focus:** Writing Integration Tests using `launch_testing` (The ROS 2 way), Unit Tests (`ament_cmake_gtest`), and running them in GitHub Actions.
> - **Code:** A `test_robot_launch.py` that starts a node, publishes a fake "Lidar Scan", and asserts that the robot publishes a "Velocity Command" within 2 seconds.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Distinguish** Unit Tests (Function level) vs Integration Tests (Node level).
2.  **Write** a `launch_testing` script that spins up a system and validates topic data.
3.  **Config** a GitHub Actions YAML to build and test the ROS 2 workspace on every Push.
4.  **Use** `colcon test` to run the suite locally.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
# Standard ROS 2
pip install pytest
```

### Prior Knowledge
- Python `unittest`.
- Launch Files.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Testing Pyramid

1.  **Unit Tests (Bottom):** Test `calculate_velocity()`. fast. `gtest` or `pytest`.
2.  **Integration Tests (Middle):** Test `Node A` talking to `Node B`. Slow. `launch_testing`.
3.  **System Tests (Top):** Simulation (Gazebo). Very slow.

### 🔹 Part 2: `launch_testing`

ROS 2's framework for "Black Box" testing.
*   **Fixture:** The Launch Description (Nodes to start).
*   **Test:** Python functions that run *while* the nodes are running.
*   **Assertions:** `assertWaitForTopic(topic, timeout)`.

### 🔹 Part 3: CI/CD (Continuous Integration)

*   **Workflow:**
    1.  Dev pushes code.
    2.  Cloud (GitHub Actions) spins up Docker container (Day 101).
    3.  Runs `colcon build`.
    4.  Runs `colcon test`.
    5.  pass/fail badge.

---

## 💻 Implementation: Integration Test

We will test a simple "Obstacle Avoidance" node.
*   **Input:** `/scan`.
*   **Output:** `/cmd_vel`.
*   **Logic:** If scan < 1.0m, stop.

### 🛠️ Project Structure
```text
day102_testing/
├── src/
│   └── safety_stop.py
├── test/
│   └── test_safety_logic.py
└── CMakeLists.txt (or setup.py)
```

### 👨‍💻 Node Under Test (`src/safety_stop.py`)

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class SafetyStop(Node):
    def __init__(self):
        super().__init__('safety_stop')
        self.sub = self.create_subscription(LaserScan, 'scan', self.scan_cb, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)

    def scan_cb(self, msg):
        cmd = Twist()
        # If any range < 1.0, stop. Else move forward.
        min_dist = min(msg.ranges)
        if min_dist < 1.0:
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = 0.5
        self.pub.publish(cmd)

def main():
    rclpy.init()
    rclpy.spin(SafetyStop())
    rclpy.shutdown()
```

### 👨‍💻 Test Script (`test/test_safety_logic.py`)

```python
import os
import unittest
import rclpy
from rclpy.node import Node
import launch
import launch_ros.actions
import launch_testing.actions
import pytest
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

# 1. Generate Launch Description
@pytest.mark.rostest
def generate_test_description():
    node = launch_ros.actions.Node(
        package='day102_testing',
        executable='safety_stop',
        output='screen'
    )
    
    return launch.LaunchDescription([
        node,
        launch_testing.actions.ReadyToTest()
    ]), {'node': node}

# 2. The Test Class
class TestSafetyLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_node')
        self.pub = self.node.create_publisher(LaserScan, 'scan', 10)
        self.received_msgs = []
        self.sub = self.node.create_subscription(
            Twist, 'cmd_vel', 
            lambda msg: self.received_msgs.append(msg), 
            10
        )

    def tearDown(self):
        self.node.destroy_node()

    def test_stop_on_obstacle(self):
        """Test if robot stops when obstacle is near"""
        # Publish Fake Scan (0.5m)
        scan = LaserScan()
        scan.header.frame_id = "laser"
        scan.ranges = [0.5, 0.5, 0.5]
        
        # Spin loop to send/recv
        end_time = self.node.get_clock().now() + rclpy.duration.Duration(seconds=2)
        while self.node.get_clock().now() < end_time:
            self.pub.publish(scan)
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
            if len(self.received_msgs) > 0:
                cmd = self.received_msgs[-1]
                if cmd.linear.x == 0.0:
                    return # SUCCESS
        
        self.fail("Did not receive STOP command!")

    def test_go_clear(self):
        """Test if robot moves when clear"""
        # Publish Fake Scan (2.0m)
        scan = LaserScan()
        scan.header.frame_id = "laser"
        scan.ranges = [2.0, 2.0, 2.0]
        self.received_msgs = [] # Clear history
        
        end_time = self.node.get_clock().now() + rclpy.duration.Duration(seconds=2)
        while self.node.get_clock().now() < end_time:
            self.pub.publish(scan)
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
            if len(self.received_msgs) > 0:
                cmd = self.received_msgs[-1]
                if cmd.linear.x > 0.0:
                    return # SUCCESS

        self.fail("Did not receive MOVE command!")
```

---

## 🔬 Lab Exercise: "The CI Pipeline"

### 1. Lab Objectives
- **Run Locally:** `colcon test --packages-select day102_testing`.
- **View Results:** `colcon test-result --all`.
- **Create:** `.github/workflows/ros2.yaml`.
- **Content:**
    ```yaml
    name: ROS 2 CI
    on: [push]
    jobs:
      build:
        runs-on: ubuntu-latest
        container: osrf/ros:humble-desktop
        steps:
          - uses: actions/checkout@v2
          - run: colcon build
          - run: colcon test
          - run: colcon test-result
    ```
- **Action:** Push code to GitHub. Watch it turn Green (or Red).

---

## 🚀 Project: "Mock Hardware"

**Goal:** Test a Driver without the Hardware.
1.  **Architecture:** Make `DriverClass` abstract.
2.  **Implementation 1:** `RealDriver` (Uses Serial Port).
3.  **Implementation 2:** `MockDriver` (Returns static data).
4.  **Test:** In `test_launch.py`, remap parameters to use `mode: mock`.
5.  **Assert:** Driver publishes standard messages even in CI environment (where serial ports don't exist).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Test timed out"
*   **Cause:** DDS discovery takes time. The test started checking *before* the nodes connected.
*   **Fix:** Wait for subscriber count > 0 before starting assertions. `while pub.get_subscription_count() == 0: sleep`.

#### 2. "Exit code mismatch"
*   **Cause:** Node crashed during test.
*   **Fix:** `launch_testing` captures exit codes. Assert `exit_code == 0`.

---

## ⚡ Optimization: Flaky Tests

Tests that pass 90% of the time.
*   **Cause:** Timing issues, Network load.
*   **Fix:** Use `retries` or increase Timeouts. Avoid `sleep(1)`. Use `wait_for_condition(lambda: check(), timeout=5)`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is `colcon test`?
    *   **A:** A command that finds and executes tests defined in `CMakeLists.txt` or `package.xml` (pytest/gtest/nosetests).
2.  **Q:** Why run tests in Docker?
    *   **A:** To ensure the environment matches production. If I test on Ubuntu 22 and deploy on Ubuntu 20, it might fail. Docker fixes this.
3.  **Q:** What is a "Fixture"?
    *   **A:** The setup required to run a test (e.g., "Start these 3 nodes").

### Challenge Task
> **Task:** Test QoS Compatibility.
> 1. Set Node to `Reliable`.
> 2. Set Test to `Best Effort`.
> 3. Assert that NO messages are received.
> 4. Fix Test to `Reliable`.
> 5. Assert messages received.

---

## 📚 Further Reading
- **ROS 2 Quality:** "Testing Guide".
- **Launch Testing:** Official Examples.

---

**Day 102 Complete**
