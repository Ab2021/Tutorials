# Day 175: Project Planning & Architecture (The Masterpiece)
## Phase 3: Camera Systems & ISP | Week 28: The Masterpiece Project

---

## 🎯 Learning Objectives
1.  **Define** the scope of the Final Capstone Project: An Autonomous Mobile Robot (AMR) Vision System.
2.  **Architect** the system: Hardware (Jetson, Camera, Motor Driver), Software (ROS 2 / GStreamer / Python), and Cloud (AWS).
3.  **Select** the components: Camera Module, Compute Board, Chassis, Battery.
4.  **Draft** the Interface Control Document (ICD): Defining messages between subsystems.
5.  **Set up** the Project Repository and Kanban Board.

---

## 📚 Prerequisites & Preparation
*   **Goal:** This is the culmination of 350 days of learning.
*   **Hardware:** Jetson Nano/Orin, MIPI Camera (IMX219/IMX477), Robot Chassis (2WD/4WD), Motor Driver (L298N/PCA9685).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Mission
*   **Scenario:** A "Warehouse Robot" that navigates a floor, detects packages (QR Codes / Boxes), avoids obstacles (People), and reports inventory to the Cloud.
*   **Key Features:**
    1.  **Line Following / Lane Keeping:** Using OpenCV.
    2.  **Obstacle Avoidance:** Using Stereo Vision or AI Detection.
    3.  **Barcode/QR Reading:** Using ZBar.
    4.  **Remote Streaming:** WebRTC to Browser.

### 🔹 Part 2: System Architecture
*   **Perception Layer:**
    *   Input: Camera (MIPI CSI-2).
    *   Processing: ISP -> Zero-Copy -> AI (TensorRT) -> CV (OpenCV).
    *   Output: `ObjectList`, `LaneDeviation`.
*   **Control Layer:**
    *   Input: `ObjectList`, `LaneDeviation`.
    *   Logic: State Machine (Idle, Patrol, Avoid, Stop).
    *   Output: `MotorPWM` (Left, Right).
*   **Communication Layer:**
    *   Protocol: MQTT (Telemetry), WebRTC (Video).
    *   Cloud: AWS IoT Core.

### 🔹 Part 3: Power Budget
*   **Jetson:** 10W.
*   **Motors:** 12V x 2A = 24W (Peak).
*   **Camera:** 0.5W.
*   **Total:** ~35W Peak.
*   **Battery:** 3S LiPo (11.1V) 2200mAh. Runtime ~45 mins.

---

## 💻 Implementation Examples

### Example 1: Directory Structure

Organizing the codebase.

```text
/masterpiece_robot
    /hardware
        motor_driver.py
        camera_driver.py
    /perception
        lane_detector.py
        object_detector.py
        qr_reader.py
    /control
        pid_controller.py
        state_machine.py
    /cloud
        mqtt_client.py
        webrtc_streamer.py
    /config
        robot_config.json
        model_weights.pt
    main.py
    requirements.txt
```

### Example 2: Interface Control Document (JSON Schema)

Defining how modules talk.

```json
// Topic: robot/perception/objects
{
    "timestamp": 1698765432.123,
    "objects": [
        {"id": 1, "class": "person", "bbox": [100, 100, 200, 300], "distance": 1.5},
        {"id": 2, "class": "box", "bbox": [400, 200, 500, 300], "distance": 2.0}
    ]
}

// Topic: robot/control/motors
{
    "left_speed": 0.8,  // -1.0 to 1.0
    "right_speed": 0.8
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Hardware Assembly

**Objective:** Build the rig.

**Steps:**
1.  Mount Jetson on Chassis.
2.  Connect Camera to CSI Port.
3.  Connect Motor Driver to GPIOs (PWM).
4.  Connect Battery via Buck Converter (5V for Jetson, 12V for Motors).
5.  **Test:** Spin motors. Capture image.

### Lab 2: Software Environment

**Objective:** Dockerize it.

**Steps:**
1.  Create `Dockerfile.base` with L4T (Linux for Tegra), OpenCV, PyTorch, TensorRT.
2.  Build the image (this takes hours on Jetson, do it overnight).
3.  **Benefit:** Reproducible environment. No "it works on my machine" issues.

### Lab 3: The "Hello World" of Robotics

**Objective:** Move and See.

**Steps:**
1.  Write a script that moves the robot forward for 1 second.
2.  Simultaneously, record a 1-second video.
3.  Save both.
4.  **Verify:** Did the video shake? (Vibration test).

---

## 🐛 Debugging Architecture

### Debug 1: "Brownout on Start"

**Symptom:** Jetson reboots when motors start.

**Cause:**
*   Motors draw huge Inrush Current. Voltage dips below 4.75V.
*   **Fix:** Use a separate battery for motors, or a high-quality Buck Converter with large capacitors.

### Debug 2: "Latency is too high for control"

**Symptom:** Robot oscillates (wobbles) on the line.

**Cause:**
*   Camera latency (100ms) + Inference (50ms) = 150ms delay.
*   **Fix:** Reduce resolution (QVGA). Increase FPS (60fps). Tune PID controller (reduce P, increase D).

---

## ⚡ Performance Optimization

### Optimization 1: Multiprocessing

*   Python GIL (Global Interpreter Lock) limits threads.
*   Use `multiprocessing.Process` for Perception and Control.
*   Use `SharedMemory` or `ZMQ` for IPC (Inter-Process Communication).

### Optimization 2: Offload to MCU

*   If Jetson GPIO PWM is jittery (it is), use an Arduino/STM32 via USB/UART to handle the motors.
*   Jetson sends "Velocity Command", Arduino runs the PID loop.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is "Dead Reckoning"?** (Estimating position based on wheel encoders. Prone to drift).
2.  **Why use ROS 2?** (Standard middleware for robotics. Handles messaging, transforms, and hardware abstraction. We are building a "Lite" version here).
3.  **What is a "Watchdog" in robotics?** (If the Control loop crashes, the motors must stop immediately. The Motor Driver should have a timeout).

### Practical Challenges

1.  **Gantt Chart:** Create a timeline for the next 5 days. Day 1: Hardware. Day 2: Vision. Day 3: Control. Day 4: Cloud. Day 5: Demo.
2.  **Risk Assessment:** What if the robot falls off a table? (Add Cliff Sensors - IR).

---

## 📚 Further Reading & Resources

### Documentation
*   **ROS 2 Documentation.**
*   **NVIDIA JetBot (Reference Design).**

---

## 🎓 Summary

Today we covered:
- ✅ **Scope:** The Warehouse Robot.
- ✅ **Architecture:** Perception, Control, Cloud.
- ✅ **Hardware:** Jetson + Chassis.
- ✅ **ICD:** JSON messages.
- ✅ **Setup:** Docker & Power.

**Next:** Day 176 - Hardware Integration.

---

**Day 175 Complete** | Phase 3: Camera Systems & ISP | Week 28: The Masterpiece Project


