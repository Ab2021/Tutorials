# Day 144: Continuous Integration (CI/CD) for ROS 2
## Phase 4: ADAS & Robotics Systems | Week 21: System Integration & Capstone

---

> **📝 Day 144 Focus:**
> You wrote the code. It works. But will it work next week after your teammate merges their changes? **CI/CD** automates the verification process. Every time you push code, a cloud server builds it, runs tests, and checks coding style.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Write** Unit Tests using `pytest` (Python) and `gtest` (C++).
2.  **Configure** `ament_lint` to enforce coding standards (PEP8).
3.  **Create** a GitHub Actions workflow for ROS 2.
4.  **Use** `industrial_ci` to simplify ROS CI configuration.
5.  **Analyze** build logs and test results.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Git:** Push/Pull/Merge.
-   **Day 143:** Docker (CI runs in Docker).

### Hardware Requirements
-   **None:** Cloud based (GitHub).

### Software Stack
-   **GitHub Actions:** CI runner.
-   **industrial_ci:** ROS CI helper.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Testing in ROS 2

-   **Unit Tests:** Test a single function/class. Fast.
-   **Integration Tests:** Test interaction between nodes (Launch tests). Slower.
-   **Linters:** Check style (`ament_flake8`, `ament_cpplint`).
-   Command: `colcon test`.

### 🔹 Part 2: GitHub Actions

-   **Workflow:** Defined in `.github/workflows/ros_ci.yml`.
-   **Runner:** A virtual machine (Ubuntu) provided by GitHub.
-   **Steps:** Checkout Code -> Setup ROS -> Build -> Test.

### 🔹 Part 3: Industrial CI

Setting up ROS 2 on a bare runner takes time.
-   **industrial_ci:** A script that spins up a Docker container (using the official ROS images), mounts your repo, and runs `colcon build` and `colcon test` automatically.
-   Configuration via Environment Variables (`ROS_DISTRO`, `UPSTREAM_WORKSPACE`).

---

## 💻 Implementation: Adding Tests & CI

**Scenario:**
-   We want to test our `LifecycleCamera` from Day 141.
-   We want to ensure it builds on GitHub.

### 🛠️ Setup
Go to `week21_day141`.

### 👨‍💻 Code: Unit Test (`test/test_camera.py`)

Create `week21_day141/test/test_camera.py`.

```python
import pytest
import rclpy
from week21_day141.lifecycle_camera import LifecycleCamera

# Basic Unit Test
def test_camera_init():
    rclpy.init()
    node = LifecycleCamera()
    assert node.get_name() == 'lifecycle_camera'
    node.destroy_node()
    rclpy.shutdown()
```

### 📄 Update `setup.py`

Ensure tests are discovered.

```python
# ... inside setup()
tests_require=['pytest'],
# ...
```

### 📄 GitHub Workflow (`.github/workflows/ros_ci.yml`)

Create this file in the root of your repo.

```yaml
name: ROS 2 CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  industrial_ci:
    name: ROS 2 Humble CI
    runs-on: ubuntu-latest
    env:
      ROS_DISTRO: humble
      # If you have dependencies in a .repos file:
      # UPSTREAM_WORKSPACE: 'dependencies.repos'
    steps:
      - uses: actions/checkout@v3
      - uses: ros-industrial/industrial_ci@master
```

---

## 🔬 Lab Exercise: The Red Cross

### Lab Objectives
1.  **Run Tests Locally:**
    ```bash
    colcon test --packages-select week21_day141
    colcon test-result --verbose
    ```
    -   **Observation:** Tests pass (or fail if you have syntax errors).

2.  **Push to GitHub:**
    -   Commit your code and push.
    -   Go to the "Actions" tab in your GitHub repo.
    -   **Observation:** You should see a yellow dot (Running), then a green check (Success) or red cross (Failure).

3.  **Break the Code:**
    -   Add a syntax error to `lifecycle_camera.py`.
    -   Push.
    -   **Result:** CI fails. You get an email. This prevents bad code from merging.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Missing Dependencies
**Symptom:** CI fails with "Package 'cv_bridge' not found".
**Cause:** `industrial_ci` installs dependencies listed in `package.xml`.
**Solution:** Ensure ALL dependencies are in `package.xml` (`<exec_depend>`, `<test_depend>`).

#### 2. Linter Errors
**Symptom:** CI fails on `ament_flake8`.
**Cause:** Code style issues (indentation, whitespace).
**Solution:** Run `ament_flake8 .` locally and fix errors. Or disable linting in CI (not recommended).

---

## ⚡ Optimization & Best Practices

### 1. Caching
Building takes time.
-   Use **ccache** to cache object files between CI runs.
-   `industrial_ci` supports this via `CCACHE_DIR` env var.

### 2. Matrix Build
Test against multiple ROS versions.
```yaml
strategy:
  matrix:
    ros_distro: [humble, iron, rolling]
env:
  ROS_DISTRO: ${{ matrix.ros_distro }}
```

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the command to run tests in ROS 2?
    *   **A:** `colcon test`.
2.  **Q:** Why do we use `industrial_ci` instead of writing our own shell scripts?
    *   **A:** It handles Docker setup, dependency resolution (`rosdep`), and workspace sourcing automatically, saving hundreds of lines of boilerplate.
3.  **Q:** What is a "Linter"?
    *   **A:** A tool that analyzes code for stylistic errors and bugs without executing it (Static Analysis).

### Challenge Task
**Task:** Integration Test.
1.  Write a test that launches the node and checks if it publishes to `/camera/image`.
2.  Use `launch_testing`.

---

## 📚 Further Reading & References
-   [Industrial CI Documentation](https://github.com/ros-industrial/industrial_ci)
-   [ROS 2 Testing Tutorial](https://docs.ros.org/en/humble/Tutorials/Intermediate/Testing/Testing-Main.html)

---

**Day 144 Complete** | Phase 4: ADAS & Robotics Systems | Week 21: System Integration & Capstone
