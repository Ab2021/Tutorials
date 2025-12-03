# Day 156: Software-in-the-Loop (SIL)
## Phase 4: ADAS & Robotics Systems | Week 23: Testing & Validation

---

> **📝 Day 156 Focus:**
> MIL proved the math works. But does the C++ code work? **Software-in-the-Loop (SIL)** tests the actual production code (compiled binary) against a simulated environment. This catches bugs like Integer Overflow, Memory Leaks, and Segmentation Faults.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Distinguish** MIL (Math) from SIL (Code).
2.  **Wrap** a C++ ROS Node in a SIL Test Harness.
3.  **Simulate** sensor inputs (Mocking) for the compiled node.
4.  **Automate** SIL testing using `launch_testing`.
5.  **Detect** implementation-specific bugs (e.g., float precision).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **C++:** Compilation, Linking.
-   **Day 144:** CI/CD (Unit Tests).

### Hardware Requirements
-   **None:** Simulation based (Host PC).

### Software Stack
-   **ROS 2:** `launch_testing`, `gtest`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why SIL?

-   **Platform Differences:** Python (MIL) handles large numbers automatically. C++ `int` overflows at 2 billion.
-   **Timing:** MIL runs "as fast as possible". SIL can run in "Real-Time" or "Step-Based" mode to check timing constraints.
-   **Integration:** SIL tests the actual interfaces (Topics/Services) used in production.

### 🔹 Part 2: The Test Harness

To test a node without a real car:
1.  **DUT (Device Under Test):** Your compiled ROS node.
2.  **Plant Simulator:** A separate node that mimics the car (Physics).
3.  **Test Manager:** Orchestrates the test (Start, Monitor, Stop, Judge).

### 🔹 Part 3: Automated Regression

-   Every night, the CI server runs the SIL suite.
-   If a developer changes the PID logic and breaks the braking distance, SIL catches it immediately.

---

## 💻 Implementation: SIL Test for AEB

**Scenario:**
-   **Node:** Automatic Emergency Braking (AEB).
-   **Logic:** If `dist < 10m`, publish `brake = 1.0`.
-   **Test:** Simulate a crash scenario and verify the node publishes the brake command.

### 🛠️ Setup
Create `week23_day156` package.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake week23_day156
mkdir -p week23_day156/test
```

### 👨‍💻 Code: The AEB Node (C++)

`src/aeb_node.cpp`

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/bool.hpp"

class AEBNode : public rclcpp::Node {
public:
    AEBNode() : Node("aeb_node") {
        sub_dist_ = this->create_subscription<std_msgs::msg::Float32>(
            "distance", 10, std::bind(&AEBNode::dist_cb, this, std::placeholders::_1));
        pub_brake_ = this->create_publisher<std_msgs::msg::Bool>("brake_cmd", 10);
    }

private:
    void dist_cb(const std_msgs::msg::Float32::SharedPtr msg) {
        std_msgs::msg::Bool cmd;
        if (msg->data < 10.0) {
            cmd.data = true; // BRAKE!
            RCLCPP_WARN(this->get_logger(), "Emergency Braking!");
        } else {
            cmd.data = false;
        }
        pub_brake_->publish(cmd);
    }

    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_dist_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_brake_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AEBNode>());
    rclcpp::shutdown();
    return 0;
}
```

### 👨‍💻 Code: The SIL Test (Python Launch Test)

`test/test_aeb_sil.py`

```python
import os
import pytest
import rclpy
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_testing.actions import ReadyToTest
import launch_testing.markers
from std_msgs.msg import Float32, Bool
import unittest

# This function generates the launch description
@pytest.mark.launch_test
def generate_test_description():
    aeb_node = Node(
        package='week23_day156',
        executable='aeb_node',
        name='aeb_node'
    )

    return LaunchDescription([
        aeb_node,
        ReadyToTest()
    ])

# This class contains the actual tests
class TestAEB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_runner')
        self.pub = self.node.create_publisher(Float32, 'distance', 10)
        self.sub = self.node.create_subscription(Bool, 'brake_cmd', self.callback, 10)
        self.received_msgs = []

    def tearDown(self):
        self.node.destroy_node()

    def callback(self, msg):
        self.received_msgs.append(msg)

    def test_brake_trigger(self):
        # 1. Publish Safe Distance (20m)
        msg = Float32()
        msg.data = 20.0
        self.pub.publish(msg)
        
        # Spin briefly
        rclpy.spin_once(self.node, timeout_sec=0.5)
        
        # Check: Should NOT brake
        if self.received_msgs:
            assert self.received_msgs[-1].data == False, "Braked too early!"
            
        # 2. Publish Danger Distance (5m)
        msg.data = 5.0
        self.pub.publish(msg)
        
        # Spin
        rclpy.spin_once(self.node, timeout_sec=0.5)
        
        # Check: SHOULD brake
        assert len(self.received_msgs) > 0, "No command received"
        assert self.received_msgs[-1].data == True, "Failed to brake!"
```

### 📄 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.8)
project(week23_day156)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(aeb_node src/aeb_node.cpp)
ament_target_dependencies(aeb_node rclcpp std_msgs)

install(TARGETS aeb_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  
  add_launch_test(test/test_aeb_sil.py TARGET aeb_node)
endif()

ament_package()
```

---

## 🔬 Lab Exercise: Running the SIL Test

### Lab Objectives
1.  **Build:**
    ```bash
    colcon build --packages-select week23_day156
    ```
2.  **Test:**
    ```bash
    colcon test --packages-select week23_day156 --event-handlers console_direct+
    ```
3.  **Observation:**
    -   The test launches the C++ node.
    -   The Python test script acts as the "Plant" (sending distance) and "Judge" (checking brake).
    -   Output: `test_brake_trigger ... ok`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Test Timeout
**Symptom:** Test fails with "Timeout".
**Cause:** Node didn't start fast enough, or topic names mismatch.
**Solution:** Increase `timeout_sec`. Check `ros2 topic list` during test (if possible) or use `RCLCPP_INFO` to debug.

#### 2. Flaky Tests
**Symptom:** Test passes sometimes, fails others.
**Cause:** Race conditions. Publishing before subscriber is ready.
**Solution:** Wait for discovery. `while pub.get_subscription_count() == 0: sleep`.

---

## ⚡ Optimization & Best Practices

### 1. Dockerized SIL
Run SIL tests inside the Docker container (Day 143).
-   Ensures the environment (libraries, OS) matches production.
-   "If it passes in Docker, it passes in Cloud."

### 2. Code Coverage
Use `lcov` to measure how much code the SIL tests touch.
-   Goal: 100% Line Coverage for safety-critical nodes (AEB).
-   If `else` branch is never hit, write a test case for it.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Can SIL test hardware faults?
    *   **A:** No. SIL runs on a PC. It cannot simulate a blown fuse or a voltage spike. That requires HIL.
2.  **Q:** Why use Python for the Test Runner if the Node is C++?
    *   **A:** Python is easier for writing test logic (assertions, mocking). The Node doesn't care who sends the ROS messages.
3.  **Q:** What is "Regression Testing"?
    *   **A:** Re-running old tests to ensure new changes didn't break existing functionality.

### Challenge Task
**Task:** Hysteresis Test.
1.  Modify AEB to release brake only if `dist > 12m` (Hysteresis).
2.  Update the SIL test to verify this behavior.
    -   Dist 5 $\to$ Brake ON.
    -   Dist 11 $\to$ Brake ON (Still).
    -   Dist 13 $\to$ Brake OFF.

---

## 📚 Further Reading & References
-   [ROS 2 Launch Testing](https://github.com/ros2/launch/tree/humble/launch_testing)
-   [Google Test (gtest)](https://google.github.io/googletest/)

---

**Day 156 Complete** | Phase 4: ADAS & Robotics Systems | Week 23: Testing & Validation
