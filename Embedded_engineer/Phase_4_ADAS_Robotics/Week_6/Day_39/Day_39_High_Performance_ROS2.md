# Day 39: Writing High-Performance ROS 2 Nodes
## Phase 4: ADAS & Robotics Systems | Week 6: Embedded Systems & Real-Time OS

---

> **📝 Day 39 Focus:**
> A naive ROS 2 node is just a `while(1)` loop. A **Production** ROS 2 node is a state machine. It can be configured, activated, deactivated, and cleaned up deterministically. Today, we master **Lifecycle Nodes** and **Multi-Threaded Executors** to build robust, high-performance drivers.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Implement** a Managed (Lifecycle) Node with states: Unconfigured, Inactive, Active, Finalized.
2.  **Control** node transitions using the `ros2 lifecycle` command line tool.
3.  **Optimize** concurrency using **Executors** and **Callback Groups**.
4.  **Differentiate** between Mutually Exclusive and Reentrant Callback Groups.
5.  **Achieve** Zero-Copy data transfer using Intra-Process Communication.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 38:** ROS 2 Architecture.
-   **C++:** Inheritance, Virtual Functions.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS.

### Software Stack
-   **ROS 2:** `rclcpp_lifecycle`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Lifecycle Nodes (Managed Nodes)

Standard Nodes start doing work immediately upon creation. This is bad for complex systems (e.g., don't start the motor driver until parameters are loaded).
**Lifecycle Nodes** have a defined state machine:

1.  **Unconfigured:** Node is created. No resources allocated.
2.  **Inactive:** Configured (Params loaded, Pubs/Subs created). But *not* doing work (not publishing).
3.  **Active:** Doing work (Publishing, Processing).
4.  **Finalized:** Destroyed.

**Transitions:** `configure()`, `activate()`, `deactivate()`, `cleanup()`, `shutdown()`.

### 🔹 Part 2: Executors & Callbacks

In ROS 1, `ros::spin()` was single-threaded.
In ROS 2, **Executors** manage how callbacks are run.

#### 2.1 SingleThreadedExecutor (Default)
-   Runs callbacks one by one.
-   Safe (no race conditions).
-   Blocking: If one callback takes 1s, all others wait.

#### 2.2 MultiThreadedExecutor
-   Runs callbacks in a thread pool.
-   Parallel execution.
-   **Danger:** Race conditions if sharing data.

#### 2.3 Callback Groups
Control which callbacks can run in parallel.
-   **Mutually Exclusive Group:** Callbacks in this group never run in parallel with each other. (Like a Mutex).
-   **Reentrant Group:** Callbacks can run in parallel (even with themselves!).

### 🔹 Part 3: Zero Copy (Intra-Process)

When Node A sends a message to Node B in the *same process*:
-   **Standard:** Serialize -> DDS -> Deserialize. (Slow).
-   **Intra-Process:** Pass a pointer. (Fast).
-   **Requirement:** Use `std::unique_ptr` for messages (Ownership transfer).

---

## 💻 Implementation: Lifecycle & Threading

We will create a package `advanced_nodes` with a Lifecycle Node that simulates a Camera Driver.

### 🛠️ Setup
Create `week6_day39` package.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake week6_day39 --dependencies rclcpp rclcpp_lifecycle std_msgs
cd week6_day39
touch src/camera_driver.cpp
```

### 👨‍💻 Code: camera_driver.cpp

```cpp
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class CameraDriver : public rclcpp_lifecycle::LifecycleNode {
public:
    CameraDriver() : LifecycleNode("camera_driver") {
        RCLCPP_INFO(get_logger(), "Node Created (Unconfigured). Waiting for configuration...");
    }

    // --- State Transitions ---

    CallbackReturn on_configure(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(get_logger(), "Configuring...");
        
        // 1. Create Publisher (Lifecycle Publisher)
        pub_ = this->create_publisher<std_msgs::msg::String>("image_data", 10);
        
        // 2. Create Timer (but don't start it yet? Actually timers start immediately in ROS2 usually, 
        // but for Lifecycle, we usually gate the logic in the callback or create/destroy timer)
        // Better pattern: Create timer here, but logic checks state.
        // Or: Create timer in on_activate.
        
        // Let's use a MultiThreaded logic simulation
        callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        
        RCLCPP_INFO(get_logger(), "Configured. Hardware Initialized.");
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_activate(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(get_logger(), "Activating...");
        
        // Enable Publisher
        pub_->on_activate();
        
        // Start Processing Thread
        active_ = true;
        capture_thread_ = std::thread(&CameraDriver::capture_loop, this);
        
        RCLCPP_INFO(get_logger(), "Active. Publishing Data.");
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(get_logger(), "Deactivating...");
        
        // Stop Thread
        active_ = false;
        if (capture_thread_.joinable()) capture_thread_.join();
        
        // Disable Publisher
        pub_->on_deactivate();
        
        RCLCPP_INFO(get_logger(), "Inactive. Standby.");
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(get_logger(), "Cleaning up...");
        
        // Release Resources
        pub_.reset();
        
        RCLCPP_INFO(get_logger(), "Cleaned up. Back to Unconfigured.");
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_shutdown(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(get_logger(), "Shutting down...");
        active_ = false;
        if (capture_thread_.joinable()) capture_thread_.join();
        pub_.reset();
        return CallbackReturn::SUCCESS;
    }

private:
    void capture_loop() {
        int frame_id = 0;
        while (active_ && rclcpp::ok()) {
            auto msg = std::make_unique<std_msgs::msg::String>();
            msg->data = "Frame " + std::to_string(frame_id++);
            
            // Publish (if active)
            if (pub_->is_activated()) {
                pub_->publish(std::move(msg));
                // Simulate work
                std::this_thread::sleep_for(100ms); 
            }
        }
    }

    rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::String>::SharedPtr pub_;
    std::thread capture_thread_;
    std::atomic<bool> active_{false};
    rclcpp::CallbackGroup::SharedPtr callback_group_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<CameraDriver>();
    
    // Use MultiThreaded Executor
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node->get_node_base_interface());
    
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}
```

### 👨‍💻 Code: CMakeLists.txt

```cmake
add_executable(camera_driver src/camera_driver.cpp)
ament_target_dependencies(camera_driver rclcpp rclcpp_lifecycle std_msgs)

install(TARGETS
  camera_driver
  DESTINATION lib/${PROJECT_NAME})
```

### 🛠️ Build & Run

```bash
cd ~/ros2_ws
colcon build --packages-select week6_day39
source install/setup.bash

# Terminal 1: Run the node
ros2 run week6_day39 camera_driver

# Terminal 2: Control the Lifecycle
# Check state
ros2 lifecycle get /camera_driver
# Configure
ros2 lifecycle set /camera_driver configure
# Activate
ros2 lifecycle set /camera_driver activate
# Listen
ros2 topic echo /image_data
# Deactivate
ros2 lifecycle set /camera_driver deactivate
```

---

## 🔬 Lab Exercise: Executor Starvation

### Lab Objectives
1.  Add a `heavy_computation` callback to the node that sleeps for 2 seconds.
2.  **Scenario A:** Use `SingleThreadedExecutor`.
    -   Trigger the heavy callback.
    -   *Observation:* The `image_data` publishing stops for 2 seconds. (Starvation).
3.  **Scenario B:** Use `MultiThreadedExecutor` and put the heavy callback in a separate `Reentrant` Callback Group.
    -   *Observation:* The `image_data` continues publishing smoothly while the heavy calculation runs in the background.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Transition Failure
**Symptom:** `ros2 lifecycle set ... activate` fails.
**Cause:** Exception thrown inside `on_activate`.
**Solution:** Check logs. Ensure hardware is ready. If `on_activate` returns `FAILURE`, the node goes to `ErrorProcessing` state.

#### 2. Race Conditions
**Symptom:** Random crashes or corrupted data.
**Cause:** Two callbacks accessing the same member variable without a Mutex in a MultiThreadedExecutor.
**Solution:** Use `std::mutex` to protect shared data.

---

## ⚡ Optimization & Best Practices

### 1. Component Containers
Don't run every node as a separate process (Executable).
Compile them as **Components** (Shared Libraries).
Load them into a single **Component Container**.
-   **Benefit:** Enables Zero-Copy (Intra-process) communication automatically.
-   **Command:** `ros2 component load /ComponentManager package node_plugin`.

### 2. Real-Time Kernel (PREEMPT_RT)
For true determinism:
-   Install a Linux kernel with `PREEMPT_RT` patch.
-   Set thread priority to Real-Time (`SCHED_FIFO`).
-   Lock memory (`mlockall`) to prevent page faults.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the benefit of the `Inactive` state?
    *   **A:** The node is fully configured (memory allocated, connections made) but silent. It can switch to `Active` instantly (microseconds) when needed.
2.  **Q:** When should I use a `MutuallyExclusive` callback group?
    *   **A:** When you have a set of callbacks that touch the same data and you don't want to write complex mutex logic.
3.  **Q:** How does Zero-Copy work in ROS 2?
    *   **A:** By using `unique_ptr` to pass ownership of the message memory address directly to the subscriber within the same process, bypassing serialization.

### Challenge Task
**Task:** Error Recovery.
1.  Implement `on_error`.
2.  Simulate a hardware failure in `on_activate` (return `FAILURE`).
3.  Observe the transition to `ErrorProcessing`.
4.  Implement logic in `on_error` to reset hardware and return `SUCCESS` (which goes to `Unconfigured`) or `FAILURE` (which goes to `Finalized`).

---

## 📚 Further Reading & References
-   [ROS 2 Design: Lifecycle](https://design.ros2.org/articles/node_lifecycle.html)
-   [Real-Time Programming in ROS 2](https://docs.ros.org/en/humble/Tutorials/Real-Time-Programming.html)

---

**Day 39 Complete** | Phase 4: ADAS & Robotics Systems | Week 6: Embedded Systems & Real-Time OS
