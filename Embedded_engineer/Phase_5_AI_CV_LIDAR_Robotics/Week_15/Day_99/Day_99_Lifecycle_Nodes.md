# Day 99: ROS 2 Lifecycle Management
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 15: Production-Grade ROS 2

---

> **📝 Content Creator Instructions:**
> Don't just `Ctrl+C`.
> - **Focus:** The difference between Managed Nodes (Lifecycle) and Standard Nodes. Understanding states: Unconfigured, Inactive, Active, Finalized.
> - **Code:** A `LifecycleNode` wrapper for a Camera Driver that ensures the camera device is released when the node is inactive, allowing other nodes to use it.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Diagram** the ROS 2 Lifecycle State Machine.
2.  **Implement** the callbacks: `on_configure`, `on_activate`, `on_deactivate`, `on_cleanup`, `on_shutdown`.
3.  **Control** node states using the command line (`ros2 lifecycle set`).
4.  **Explain** why Lifecycle nodes are critical for "Deterministic Startup" (Configure all $\to$ Activate all).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Webcam (or any resource that needs exclusive access).

### Software Environment
```bash
# Standard ROS 2 Humble
```

### Prior Knowledge
- ROS 2 Nodes (Day 1).
- Service Callbacks.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Startup Race Condition

In ROS 1 (and standard ROS 2 nodes):
*   Node A starts, tries to subscribe to Node B.
*   Node B isn't ready.
*   Node A crashes or logs errors.
*   **Fix:** `LifecycleNodes`.

### 🔹 Part 2: The State Machine

1.  **Unconfigured:** Node created. No memory allocated for heavy buffers.
2.  **Inactive:** Configured (params loaded, publishers created), but *not publishing*.
3.  **Active:** Publishing data. Processing callbacks.
4.  **Finalized:** Destroyed.

### 🔹 Part 3: Deterministic Launch

Instead of "Launch and Pray", we use a Manager:
1.  Load all Nodes (State: Unconfigured).
2.  Trigger `configure()` on all. (Check if hardware exists, allocate RAM).
    *   If any fail $\to$ Abort Launch.
3.  Trigger `activate()` on all.
    *   System starts synchronously.

---

## 💻 Implementation: Managed Camera Driver

We will create a node that wraps `cv2.VideoCapture`.

### 🛠️ Project Structure
```text
day99_lifecycle/
├── src/
│   ├── managed_camera.cpp
│   └── lifecycle_manager.py
└── launch/
    └── managed_system.launch.py
```

### 👨‍💻 Lifecycle Node (`src/managed_camera.cpp`)

```cpp
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

using rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface;

class ManagedCamera : public rclcpp_lifecycle::LifecycleNode
{
public:
  ManagedCamera(const std::string & node_name, bool intra_process_comms = false)
  : LifecycleNode(node_name,
      rclcpp::NodeOptions().use_intra_process_comms(intra_process_comms))
  {
  }

  // 1. Configure: Load params, Create Publisher (but don't publish yet)
  LifecycleNodeInterface::CallbackReturn on_configure(const rclcpp_lifecycle::State &)
  {
    RCLCPP_INFO(get_logger(), "Configuring...");
    pub_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", 10);
    return LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  // 2. Activate: Open Camera, Start Timer
  LifecycleNodeInterface::CallbackReturn on_activate(const rclcpp_lifecycle::State &)
  {
    RCLCPP_INFO(get_logger(), "Activating...");
    pub_->on_activate(); // Enable publisher
    
    cap_.open(0);
    if (!cap_.isOpened()) {
        RCLCPP_ERROR(get_logger(), "Cannot open camera!");
        return LifecycleNodeInterface::CallbackReturn::FAILURE;
    }
    
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(30), 
      std::bind(&ManagedCamera::publish_frame, this));
      
    return LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  // 3. Deactivate: Stop Timer, Release Camera (but keep params/publishers)
  LifecycleNodeInterface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &)
  {
    RCLCPP_INFO(get_logger(), "Deactivating...");
    pub_->on_deactivate();
    timer_->cancel(); // Stop timer
    cap_.release(); // FREES THE HARDWARE
    return LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  // 4. Cleanup: Destroy Publisher, Release Memory
  LifecycleNodeInterface::CallbackReturn on_cleanup(const rclcpp_lifecycle::State &)
  {
    RCLCPP_INFO(get_logger(), "Cleaning up...");
    pub_.reset();
    timer_.reset();
    return LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }
  
  // 5. Shutdown
  LifecycleNodeInterface::CallbackReturn on_shutdown(const rclcpp_lifecycle::State &)
  {
    RCLCPP_INFO(get_logger(), "Shutting down...");
    pub_.reset();
    return LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

private:
  void publish_frame() {
      cv::Mat frame;
      cap_ >> frame;
      if(!frame.empty()) {
          auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
          pub_->publish(*msg);
      }
  }

  rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  cv::VideoCapture cap_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ManagedCamera>("managed_camera"));
  rclcpp::shutdown();
  return 0;
}
```

### 👨‍💻 CLI Control

```bash
# Terminal 1
ros2 run day99_lifecycle managed_camera

# Terminal 2 (Check state)
ros2 lifecycle get /managed_camera
# Output: unconfigured

# Transition
ros2 lifecycle set /managed_camera configure
# Output: inactive

ros2 lifecycle set /managed_camera activate
# Output: active (Camera LED turns on)

ros2 lifecycle set /managed_camera deactivate
# Output: inactive (Camera LED turns off)
```

---

## 🔬 Lab Exercise: "Resource Conflict"

### 1. Lab Objectives
- **Scenario:** Two nodes need the same camera.
- **Node A:** Face Recognition (Managed).
- **Node B:** QR Code Scanner (Managed).
- **Task:**
    1.  Activate Node A. (A opens camera).
    2.  Try Activate Node B. (B fails to open camera $\to$ FAILURE).
    3.  Deactivate Node A.
    4.  Activate Node B. (Success).
- **Result:** You can multiplex hardware access without killing processes.

---

## 🚀 Project: "The System Manager"

**Goal:** Write a Python script (`LifecycleManager`) that automates the transition.
1.  **List:** `['/camera', '/lidar', '/planner']`.
2.  **Step 1:** `change_state(node, 'configure')` for all.
    *   Wait for verify.
3.  **Step 2:** `change_state(node, 'activate')` for all.
4.  **Error Handling:** If `/camera` fails to configure, don't start the `/planner`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Publisher doesn't publish"
*   **Cause:** Forgot `pub_->on_activate()`. Lifecycle publishers are muted by default.
*   **Fix:** Explicitly enable them in the `on_activate` callback.

#### 2. "Transition Failure"
*   **Cause:** Callback returned `FAILURE`.
*   **Rx:** ROS 2 will automatically transition the node to `error_processing` state. You need to handle error recovery or shutdown.

---

## ⚡ Optimization: Respawn

Launch files allow `respawn=True`.
*   If a managed node crashes (segfault), Launch system restarts it.
*   It starts in `Unconfigured`.
*   Your Manager needs to detect this and re-configure/re-activate it.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between `on_cleanup` and `on_shutdown`?
    *   **A:** `on_cleanup` returns the node to `Unconfigured` (clean slate, can be configured again). `on_shutdown` is final (node is about to exit).
2.  **Q:** Why use `Lifecyclepublisher` instead of `Publisher`?
    *   **A:** `LifecyclePublisher` hooks into the state machine. Standard publishers would leak data even when node is "Inactive".
3.  **Q:** Can I use lifecycle for a pure math node?
    *   **A:** Yes, to save CPU. Pausing the node stops the subscription callbacks and timers.

### Challenge Task
> **Task:** Recovery Mode.
> 1. Implement `on_error`.
> 2. Simulate a camera disconnect (unplug USB).
> 3. Verify node goes to Error state.
> 4. Wait 5s, try to re-configure.

---

## 📚 Further Reading
- **ROS 2 Design:** "Lifecycle Nodes".
- **Nav2:** Uses Lifecycle nodes extensively (check `nav2_lifecycle_manager`).

---

**Day 99 Complete**
