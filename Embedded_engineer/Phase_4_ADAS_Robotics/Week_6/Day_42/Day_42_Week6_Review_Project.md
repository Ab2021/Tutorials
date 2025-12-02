# Day 42: Week 6 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 6: Embedded Systems & Real-Time OS

---

> **📝 Day 42 Focus:**
> We have peeled back the layers of the robotic onion: from Python scripts to C++ drivers, from Linux processes to RTOS tasks, and from CPUs to GPUs and Microcontrollers. Today, we integrate everything into a **Real-Time Motor Controller** system that spans the entire stack.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Architect** a hybrid system: High-level ROS 2 (Linux) + Low-level Control (RTOS/Micro-ROS).
2.  **Implement** a robust C++ Lifecycle Node for system management.
3.  **Design** a Real-Time Control Loop using FreeRTOS tasks (simulated).
4.  **Configure** DDS QoS to ensure safety-critical commands are delivered reliably.
5.  **Evaluate** your mastery of Week 6 concepts through a comprehensive assessment.

---

## 📚 Week 6 Review

### 1. Modern C++ (The Language)
-   **Smart Pointers:** `unique_ptr` (Ownership), `shared_ptr` (Sharing).
-   **Move Semantics:** `std::move` avoids copies.
-   **RAII:** Resource management via destructors.

### 2. RTOS (The Timing)
-   **Determinism:** Guarantees deadlines.
-   **Tasks:** Preemptive scheduling based on priority.
-   **IPC:** Queues for data, Mutexes for protection (Priority Inversion risk).

### 3. ROS 2 Architecture (The Middleware)
-   **DDS:** Distributed, Data-Centric.
-   **QoS:** Reliability (Reliable/Best Effort), Durability (Volatile/Transient Local).
-   **Lifecycle:** Managed nodes (Unconfigured -> Active).

### 4. Hardware Acceleration (The Speed)
-   **GPU:** Massive parallelism for Vision/Lidar.
-   **FPGA:** Low latency pipelining.

### 5. Micro-ROS (The Bridge)
-   **XRCE-DDS:** Lightweight protocol for MCUs.
-   **Agent:** Bridge to the main ROS 2 network.

---

## 🛠️ Capstone Project: Real-Time Motor Controller

**Goal:** Build a distributed control system.
1.  **PC (Linux/ROS 2):** Runs a "Trajectory Generator" (Lifecycle Node). Sends velocity commands.
2.  **MCU (Simulated RTOS):** Runs a "PID Controller" (High Priority Task). Reads Encoder, drives Motor.
3.  **Communication:** ROS 2 Topics with custom QoS.

### Architecture
-   **Topic `cmd_vel`:** Reliable. (PC -> MCU).
-   **Topic `motor_status`:** Best Effort. (MCU -> PC).

### Package Structure
Create `week6_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week6_project
cd ~/ros2_ws/src/week6_project
touch src/motor_system.cpp
```

### 👨‍💻 Code: The Integrated System

We will simulate the MCU part using a separate thread in C++ to keep it self-contained on a PC, but the logic is identical to Micro-ROS + FreeRTOS.

```cpp
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "std_msgs/msg/float32.h"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

// --- Mock Hardware (Motor) ---
class MotorHardware {
public:
    void set_voltage(float v) {
        // Simulate physics: dv/dt = (V - k*v) / m
        float dt = 0.001; // 1ms physics step
        float drag = 0.1;
        float mass = 1.0;
        velocity_ += (v - drag * velocity_) / mass * dt;
    }
    float get_velocity() { return velocity_; }
private:
    float velocity_ = 0.0;
};

// --- RTOS Layer (Simulated) ---
class MotorController {
public:
    MotorController() {
        // Start Control Loop Task (1kHz)
        running_ = true;
        control_thread_ = std::thread(&MotorController::control_loop, this);
    }

    ~MotorController() {
        running_ = false;
        if (control_thread_.joinable()) control_thread_.join();
    }

    void set_target(float target) {
        std::lock_guard<std::mutex> lock(mutex_);
        target_velocity_ = target;
    }

    float get_current_velocity() {
        return hardware_.get_velocity();
    }

private:
    void control_loop() {
        // Real-Time Task: 1kHz (1ms period)
        while (running_) {
            auto start = std::chrono::steady_clock::now();

            // 1. Read Sensors
            float current = hardware_.get_velocity();

            // 2. Get Target (Thread Safe)
            float target;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                target = target_velocity_;
            }

            // 3. PID Control
            float error = target - current;
            integral_ += error * 0.001;
            float derivative = (error - prev_error_) / 0.001;
            
            float Kp = 2.0, Ki = 0.5, Kd = 0.1;
            float output = Kp*error + Ki*integral_ + Kd*derivative;
            
            prev_error_ = error;

            // 4. Write Actuator
            hardware_.set_voltage(output);

            // 5. Sleep (Maintain 1kHz)
            std::this_thread::sleep_until(start + 1ms);
        }
    }

    MotorHardware hardware_;
    std::thread control_thread_;
    std::atomic<bool> running_;
    std::mutex mutex_;
    
    float target_velocity_ = 0.0;
    float integral_ = 0.0;
    float prev_error_ = 0.0;
};

// --- ROS 2 Layer (Lifecycle Node) ---
class MotorDriverNode : public rclcpp_lifecycle::LifecycleNode {
public:
    MotorDriverNode() : LifecycleNode("motor_driver") {
        RCLCPP_INFO(get_logger(), "Motor Driver Created.");
    }

    CallbackReturn on_configure(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(get_logger(), "Configuring...");
        
        // QoS: Reliable for Commands
        rclcpp::QoS cmd_qos(10);
        cmd_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
        
        sub_cmd_ = this->create_subscription<std_msgs::msg::Float32>(
            "cmd_vel", cmd_qos,
            std::bind(&MotorDriverNode::cmd_callback, this, std::placeholders::_1));

        // QoS: Best Effort for Status
        rclcpp::QoS stat_qos(10);
        stat_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
        
        pub_status_ = this->create_publisher<std_msgs::msg::Float32>("motor_velocity", stat_qos);
        
        // Initialize Controller
        controller_ = std::make_unique<MotorController>();
        
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_activate(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(get_logger(), "Activating...");
        pub_status_->on_activate();
        
        // Start Telemetry Timer (10Hz)
        timer_ = this->create_wall_timer(100ms, std::bind(&MotorDriverNode::timer_callback, this));
        
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(get_logger(), "Deactivating...");
        pub_status_->on_deactivate();
        timer_->cancel();
        controller_->set_target(0.0); // Safety Stop
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) {
        RCLCPP_INFO(get_logger(), "Cleaning up...");
        controller_.reset(); // Destroy controller
        pub_status_.reset();
        sub_cmd_.reset();
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_shutdown(const rclcpp_lifecycle::State &) {
        controller_.reset();
        return CallbackReturn::SUCCESS;
    }

private:
    void cmd_callback(const std_msgs::msg::Float32::SharedPtr msg) {
        if (controller_) {
            controller_->set_target(msg->data);
            RCLCPP_INFO(get_logger(), "Target Set: %.2f", msg->data);
        }
    }

    void timer_callback() {
        if (controller_) {
            auto msg = std_msgs::msg::Float32();
            msg.data = controller_->get_current_velocity();
            if (pub_status_->is_activated()) {
                pub_status_->publish(msg);
            }
        }
    }

    std::unique_ptr<MotorController> controller_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_cmd_;
    rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float32>::SharedPtr pub_status_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MotorDriverNode>()->get_node_base_interface());
    rclcpp::shutdown();
    return 0;
}
```

### 👨‍💻 Code: CMakeLists.txt

```cmake
add_executable(motor_system src/motor_system.cpp)
ament_target_dependencies(motor_system rclcpp rclcpp_lifecycle std_msgs)

install(TARGETS
  motor_system
  DESTINATION lib/${PROJECT_NAME})
```

---

## 🧪 Verification & Testing

### 1. Lifecycle Management
**Scenario:**
1.  Run node.
2.  `ros2 lifecycle set /motor_driver configure`.
3.  `ros2 lifecycle set /motor_driver activate`.
4.  **Observation:** Node starts publishing `/motor_velocity` (0.0).

### 2. Control Response
**Scenario:**
1.  `ros2 topic pub /cmd_vel std_msgs/msg/Float32 "{data: 10.0}"`.
2.  **Observation:** `/motor_velocity` ramps up to 10.0 (simulating PID response).

### 3. Safety Shutdown
**Scenario:**
1.  `ros2 lifecycle set /motor_driver deactivate`.
2.  **Observation:** Motor target set to 0.0. Publishing stops.

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: C++ & RTOS
1.  **Q:** Why did we use `std::unique_ptr` for the `controller_`?
    *   **A:** To ensure exclusive ownership and automatic cleanup (RAII) when the node is cleaned up or destroyed.
2.  **Q:** Why is the Control Loop in a separate thread?
    *   **A:** To run at a high frequency (1kHz) independent of the ROS 2 callbacks (which might be blocked by networking or logging).

### Section 2: ROS 2
3.  **Q:** What QoS policy ensures that the `cmd_vel` is received even if the network is lossy?
    *   **A:** `Reliability: Reliable`.
4.  **Q:** Why use Lifecycle nodes for hardware drivers?
    *   **A:** To ensure the hardware is initialized only when ready, and safely shut down (motors stopped) when deactivated.

### Section 3: Hardware
5.  **Q:** If we moved the `MotorController` class to an ESP32, what would change?
    *   **A:** The `std::thread` would become a FreeRTOS Task. The `MotorHardware` would write to actual PWM registers. The ROS 2 layer would become Micro-ROS `rclc` code.

---

## 🏆 Conclusion

Congratulations on completing Week 6!
-   You have mastered the **Embedded** side of robotics.
-   You can write high-performance C++, manage real-time tasks, and bridge the gap to ROS 2.

**Next Week:** We move to **Simulation & Modeling**. We will stop writing "Mock Hardware" classes and start using physics-accurate simulators like **Gazebo** and **URDF** to model real robots.

---

**Day 42 Complete** | Phase 4: ADAS & Robotics Systems | Week 6: Embedded Systems & Real-Time OS
