# Day 41: Micro-ROS (ROS 2 on Microcontrollers)
## Phase 4: ADAS & Robotics Systems | Week 6: Embedded Systems & Real-Time OS

---

> **📝 Day 41 Focus:**
> Standard ROS 2 runs on Linux (Raspberry Pi, Jetson). But low-level control (Motor PWM, Encoder counting) happens on Microcontrollers (STM32, ESP32). How do we connect them? **Micro-ROS** puts a first-class ROS 2 node directly on the microcontroller, bridging the gap between the RTOS world and the DDS world.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the Micro-ROS architecture: Client, Agent, and XRCE-DDS.
2.  **Setup** the Micro-ROS development environment for ESP32 (Arduino or ESP-IDF).
3.  **Write** a Micro-ROS node that publishes sensor data (IMU/Encoder) to the ROS 2 network.
4.  **Subscribe** to ROS 2 topics on the microcontroller to control actuators (LEDs/Motors).
5.  **Debug** connection issues between the Micro-ROS Client and Agent.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 37:** RTOS Concepts (FreeRTOS).
-   **Embedded C:** GPIO, UART, WiFi.

### Hardware Requirements
-   **Microcontroller:** ESP32 (Recommended), Teensy 4.0, or STM32 Nucleo.
-   **Connection:** USB Cable (Serial) or WiFi.

### Software Stack
-   **Micro-ROS Arduino Library:** For easy integration.
-   **Micro-ROS Agent:** Docker container running on the PC.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Gap

Standard DDS (FastDDS/CycloneDDS) is too heavy for an MCU (Requires MBs of RAM).
**XRCE-DDS (eXtremely Resource Constrained Environments DDS):**
-   **Client (MCU):** Lightweight. Sends serialized data over Serial/UDP.
-   **Agent (PC):** Heavy lifter. Acts as a bridge. Receives data from Client and republishes it to the global DDS space.

### 🔹 Part 2: Architecture

1.  **Application Layer:** Your code (rclc). Uses C API (not C++ rclcpp) to save memory.
2.  **Middleware Layer:** Micro-XRCE-DDS Client.
3.  **Transport Layer:** UART, UDP, TCP, CAN.
4.  **Hardware:** ESP32, STM32.

### 🔹 Part 3: The `rclc` Executor

On Linux, `rclcpp` uses C++ classes.
On MCU, `rclc` uses C structs.
-   **Executor:** Manages subscriptions and timers.
-   **Allocator:** Custom memory allocator (FreeRTOS heap).

---

## 💻 Implementation: Micro-ROS on ESP32

We will create a system where:
1.  **ESP32:** Publishes `std_msgs/Int32` (Encoder count).
2.  **ESP32:** Subscribes to `std_msgs/Bool` (LED Control).
3.  **PC:** Runs the Agent and visualizes data.

### 🛠️ Setup (PC Side)

1.  **Install Micro-ROS Agent:**
    ```bash
    # Run Agent in Docker (Easiest)
    # For Serial (USB) connection
    docker run -it --rm -v /dev:/dev --privileged --net=host microros/micro-ros-agent:humble serial --dev /dev/ttyUSB0 -b 115200
    
    # OR for WiFi (UDP) connection
    docker run -it --rm --net=host microros/micro-ros-agent:humble udp4 --port 8888
    ```

### 👨‍💻 Code: ESP32 Firmware (Arduino Framework)

*Note: You need to install the `micro_ros_arduino` library in Arduino IDE.*

```cpp
#include <micro_ros_arduino.h>

#include <stdio.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>

#include <std_msgs/msg/int32.h>
#include <std_msgs/msg/bool.h>

// Hardware Pins
#define LED_PIN 2

// ROS Objects
rcl_publisher_t publisher;
rcl_subscription_t subscriber;
std_msgs__msg__Int32 msg_pub;
std_msgs__msg__Bool msg_sub;
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;

// Macros
#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){}}

void error_loop(){
  while(1){
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    delay(100);
  }
}

// --- Callbacks ---

// Timer Callback (Publish Data)
void timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
  RCLC_UNUSED(last_call_time);
  if (timer != NULL) {
    // Simulate Encoder Count
    msg_pub.data++;
    RCSOFTCHECK(rcl_publish(&publisher, &msg_pub, NULL));
  }
}

// Subscription Callback (Control LED)
void subscription_callback(const void * msgin)
{
  const std_msgs__msg__Bool * msg = (const std_msgs__msg__Bool *)msgin;
  digitalWrite(LED_PIN, (msg->data) ? HIGH : LOW);
}

// --- Setup & Loop ---

void setup() {
  // 1. Transport Setup
  set_microros_transports(); // Uses Serial by default
  // set_microros_wifi_transports("SSID", "PASS", "AGENT_IP", 8888); // For WiFi

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  delay(2000);

  allocator = rcl_get_default_allocator();

  // 2. Initialize Support
  RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));

  // 3. Create Node
  RCCHECK(rclc_node_init_default(&node, "esp32_node", "", &support));

  // 4. Create Publisher
  RCCHECK(rclc_publisher_init_default(
    &publisher,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
    "encoder_count"));

  // 5. Create Subscriber
  RCCHECK(rclc_subscription_init_default(
    &subscriber,
    &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Bool),
    "led_control"));

  // 6. Create Timer (10 Hz)
  RCCHECK(rclc_timer_init_default(
    &timer,
    &support,
    RCL_MS_TO_NS(100),
    timer_callback));

  // 7. Create Executor
  RCCHECK(rclc_executor_init(&executor, &support.context, 2, &allocator)); // 2 handles (Timer + Sub)
  RCCHECK(rclc_executor_add_timer(&executor, &timer));
  RCCHECK(rclc_executor_add_subscription(&executor, &subscriber, &msg_sub, &subscription_callback, ON_NEW_DATA));

  msg_pub.data = 0;
}

void loop() {
  // Spin the executor to handle callbacks
  RCSOFTCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100)));
  delay(10);
}
```

---

## 🔬 Lab Exercise: Controlling Hardware from ROS 2

### Lab Objectives
1.  **Flash:** Upload the code to ESP32.
2.  **Start Agent:** Run the Docker command.
    -   *Observation:* You should see `[INFO] Session established`.
3.  **Verify Publisher:**
    -   PC Terminal: `ros2 topic echo /encoder_count`
    -   *Result:* Numbers incrementing.
4.  **Verify Subscriber:**
    -   PC Terminal: `ros2 topic pub /led_control std_msgs/msg/Bool "{data: true}"`
    -   *Result:* LED on ESP32 turns ON.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Waiting for agent..." forever
**Symptom:** ESP32 hangs or Agent shows no connection.
**Cause:**
-   Baud rate mismatch (Default 115200).
-   Wrong Serial Port (`/dev/ttyUSB0` vs `USB1`).
-   Cable is power-only (no data lines).
**Solution:** Check connections. Press Reset button on ESP32 *after* starting the Agent.

#### 2. Memory Allocation Failed
**Symptom:** `rclc_support_init` fails.
**Cause:** ESP32 ran out of heap.
**Solution:** Reduce number of publishers/subscribers. Use `rclc_support_init_with_options` and custom allocator.

#### 3. Data Lag
**Symptom:** Updates are slow.
**Cause:** Serial bottleneck.
**Solution:** Increase baud rate to 921600. Or use WiFi/Ethernet.

---

## ⚡ Optimization & Best Practices

### 1. Custom Transports
If you need CAN bus (CAN-FD):
-   Implement a custom transport function `micro_ros_transport_write` / `read`.
-   Pass it to `rmw_uxrce_transport_set_custom`.

### 2. Quality of Service (QoS)
Even on MCU, you can set Best Effort.
-   `rclc_publisher_init_best_effort(...)`.
-   Saves bandwidth and retries.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why can't we run full ROS 2 on an Arduino Uno?
    *   **A:** Not enough RAM (2KB). Micro-ROS needs at least ~20KB RAM (ESP32 has 520KB).
2.  **Q:** What is the role of the Micro-ROS Agent?
    *   **A:** It acts as a proxy. It translates XRCE-DDS (Serial/UDP) messages from the MCU into standard DDS messages on the ROS 2 network.
3.  **Q:** Can Micro-ROS use FreeRTOS tasks?
    *   **A:** Yes! The `rclc_executor_spin` can run inside a FreeRTOS task.

### Challenge Task
**Task:** IMU Publisher.
1.  Connect an MPU6050 to the ESP32 (I2C).
2.  Read Accel/Gyro data.
3.  Publish `sensor_msgs/Imu` to `/imu/data_raw`.
4.  Visualize in Rviz2 on PC.

---

## 📚 Further Reading & References
-   [Micro-ROS Official Website](https://micro.ros.org/)
-   [Micro-ROS Arduino Library](https://github.com/micro-ROS/micro_ros_arduino)

---

**Day 41 Complete** | Phase 4: ADAS & Robotics Systems | Week 6: Embedded Systems & Real-Time OS
