# Day 205: Testing & Validation
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 30: Capstone Project Part 2

---

> **📝 Content Creator Instructions:**
> Don't guess. Test.
> - **Focus:** Unit Testing (Python), Integration Testing (launch_testing), and Continuous Integration (CI).
> - **Code:** `test_bringup.py` and `test_fruit_detector.py`.
> - **Concept:** TDD (Test Driven Development) vs "It works on my machine".

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Write** a `pytest` unit test for a pure logic Node (e.g., coordinate converter).
2.  **Implement** a ROS 2 Integration Test using `launch_testing`.
3.  **Configure** a GitHub Actions workflow to auto-run tests on Push.
4.  **Mock** sensor data to test perception logic without Gazebo.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install pytest launch_testing launch_testing_ros
```

### Prior Knowledge
- Unit Testing basics (Asserts).
- ROS 2 Launch.

---

## 📖 Theoretical Verification

### 🔹 Levels of Testing

1.  **Unit Test:** Checks a single function. (e.g., `calculate_grasp_offset(img)`). Fast (ms).
2.  **Integration Test:** Checks node communication. (e.g., "Does `detector` publish to `nav`?"). Medium (sec).
3.  **System Test:** Robot executes mission in Sim. Slow (mins).

### 🔹 Launch Testing

ROS 2 provides a framework to start nodes, wait for them, and assert that:
*   They didn't crash (exit code 0).
*   They published expected data to stdout/topics.

---

## 💻 Implementation: The Test Suite

### 🛠️ Project Structure
```text
agribot_testing/
├── test/
│   ├── unit/
│   │   └── test_math_utils.py
│   └── integration/
│       └── test_bringup_launch.py
└── package.xml (Depend: pytest)
```

### 👨‍💻 Unit Test (`test/unit/test_math_utils.py`)

Testing the 3D projection math without ROS.

```python
import numpy as np
import pytest

# Imagine this function is imported from agribot_perception
def project_2d_to_3d(u, v, z, fx, fy, cx, cy):
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z

def test_projection_center():
    # If pixel is at center (cx, cy), X and Y should be 0
    x, y, z = project_2d_to_3d(320, 240, 1.0, 500, 500, 320, 240)
    assert x == 0.0
    assert y == 0.0
    assert z == 1.0
    
def test_projection_corner():
    # u = cx + fx -> x should be z
    x, y, z = project_2d_to_3d(820, 240, 2.0, 500, 500, 320, 240)
    assert x == 2.0 # (820-320)*2/500 = 500*2/500 = 2
```

### 👨‍💻 Integration Test (`test/integration/test_bringup_launch.py`)

Starts the simulation and checks if nodes come up.

```python
import os
import unittest
import pytest
import rclpy
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_testing.actions import ReadyToTest
from ament_index_python.packages import get_package_share_directory

# This function tells launch_testing what to run
@pytest.mark.launch_test
def generate_test_description():
    pkg_bringup = get_package_share_directory('agribot_bringup')
    
    # Launch the FULL stack (lightweight version, maybe no GUI)
    launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, 'launch', 'sim_complete.launch.py')
        ),
        launch_arguments={'gui': 'false'}.items()
    )

    return LaunchDescription([
        launch_include,
        ReadyToTest()
    ])

# Actual Tests
class TestAgriBotStart(unittest.TestCase):
    def test_nodes_running(self, proc_output):
        # We expect a certain log message to appear
        # e.g., "Nav2 Stack Started"
        # proc_output object captures stdout
        pass 
        
    def test_topics_published(self):
        # We can spin up a temporary node to check topics
        rclpy.init()
        node = rclpy.create_node('test_listener')
        
        # Check /tf
        # ... logic to check topic existence ...
        
        node.destroy_node()
        rclpy.shutdown()
```

---

## 🔬 Lab Exercise: "CI Pipeline"

### 1. Lab Objectives
- **Create:** `.github/workflows/ros2_test.yml`.
- **Logic:**
    1.  Checkout Code.
    2.  Install ROS 2 Humble (Docker container).
    3.  `colcon build`.
    4.  `colcon test`.
- **Commit:** Push code to GitHub.
- **Watch:** See the "Actions" tab turn Green (Success) or Red (Fail).

---

## 🚀 Project Steps

1.  **Metric Tests:** Add a test that ensures the `fruit_detector` processes images at > 10Hz. If it drops to 2Hz, FAIL.
2.  **Linting:** Add `ament_flake8` and `ament_cpplint` to ensure code style (PEP8) is enforced.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Test Output Hidden"
*   **Cause:** `colcon test` swallows output by default.
*   **Fix:** Use `colcon test --event-handlers console_direct+` to see print statements.

#### 2. "Sim Hangs in CI"
*   **Cause:** GitHub Runners have no GPU. Gazebo crashes.
*   **Fix:** Skip simulation tests in CI. Mock the simulation (publish fake sensor data) for unit tests. Only run full sim tests on a robust local server.

---

## ⚡ Optimization: Code Coverage

Measure how much of your code is actually tested.
*   **Tool:** `pytest-cov`.
*   **Goal:** > 80% coverage.
*   **Reality:** Robotics code is hard to cover 100% (hardware drivers). Focus on logic (algorithms).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why Mock sensors?
    *   **A:** Because running Gazebo is slow/heavy. Mocking allows testing the "Planning" logic instantly without needing physics.
2.  **Q:** What is "Regression"?
    *   **A:** When a new feature breaks an old feature. CI prevents this by running ALL tests on EVERY change.

### Challenge Task
> **Task:** "Fuzz Testing".
> 1. Send garbage data (NaN, Infinity, Zero) to the `fruit_detector`.
> 2. Does it crash?
> 3. Does it catch the error and log a warning? (Desired behavior).

---

## 📚 Further Reading
- **ROS 2 Quality of Service (QoS):** Testing network reliability.
- **TDD for Mechanics:** "Test rig" concepts.

---

**Day 205 Complete**
