# Day 48: Hardware-in-the-Loop (HIL) Simulation
## Phase 4: ADAS & Robotics Systems | Week 7: Simulation & Modeling

---

> **📝 Day 48 Focus:**
> Simulation (SIL) is safe. Real testing (Field) is dangerous. **Hardware-in-the-Loop (HIL)** is the bridge. We trick the real hardware (ECU) into thinking it's driving a real car, when it's actually connected to Gazebo. Today, we build a HIL rig using an ESP32 and ROS 2.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Differentiate** between SIL (Software-in-Loop), PIL (Processor-in-Loop), and HIL (Hardware-in-the-Loop).
2.  **Architect** a HIL system: Gazebo (Plant) <-> Serial/CAN <-> ESP32 (Controller).
3.  **Implement** a Serial Bridge in Python to stream Gazebo state to the MCU.
4.  **Write** Firmware for the MCU to run a PID controller on simulated data.
5.  **Analyze** the effects of latency and jitter in HIL systems.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 41:** Micro-ROS (or Serial Communication).
-   **Day 44:** Gazebo Physics.

### Hardware Requirements
-   **Microcontroller:** ESP32 or Arduino.
-   **PC:** Running Gazebo.
-   **Connection:** USB Cable.

### Software Stack
-   **ROS 2:** `pyserial`, `gazebo_ros_pkgs`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The X-in-the-Loop Spectrum

1.  **Model-in-Loop (MIL):** Matlab/Simulink model of controller + plant.
2.  **Software-in-Loop (SIL):** C++ code of controller + Gazebo plant (All on PC).
3.  **Processor-in-Loop (PIL):** C++ code running on Target Architecture (QEMU/Instruction Set Sim) + Gazebo.
4.  **Hardware-in-the-Loop (HIL):** Real ECU hardware + Real I/O (CAN/PWM) + Gazebo Plant.

### 🔹 Part 2: The HIL Architecture

**The Plant (Gazebo):**
-   Simulates Physics ($F=ma$).
-   Outputs: Sensor Data (Speed, Position).
-   Inputs: Actuator Commands (Voltage/Torque).

**The Interface (Bridge):**
-   Converts ROS messages to Raw Bytes (Serial/CAN).
-   Handles synchronization.

**The Controller (ECU):**
-   Reads Sensor Bytes.
-   Runs Control Logic (PID/MPC).
-   Writes Actuator Bytes.

### 🔹 Part 3: Time Synchronization

Gazebo runs in **Sim Time**. The ECU runs in **Real Time**.
-   **Real-Time Factor (RTF):** Ratio of Sim Time to Wall Time.
-   **Requirement:** RTF must be $\approx 1.0$.
-   If Gazebo is slow (RTF 0.5), the ECU will think the car is moving in slow motion, messing up the PID D-term.

---

## 💻 Implementation: HIL Rig

We will build a simple HIL setup:
1.  **Gazebo:** Simulates a 1D mass (or our RoboCar).
2.  **Bridge:** Sends `current_velocity` to ESP32.
3.  **ESP32:** Calculates `cmd_vel` (PID) to reach `target_velocity`.
4.  **Bridge:** Sends `cmd_vel` back to Gazebo.

### 🛠️ Setup
Create `week7_day48` package.

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python week7_day48
mkdir arduino_hil
touch week7_day48/hil_bridge.py
```

### 👨‍💻 Code: ESP32 Firmware (Arduino)

Flash this to your ESP32.

```cpp
// HIL Controller
// Reads: Current Velocity (float)
// Writes: Force Command (float)

float target_velocity = 2.0; // m/s
float Kp = 10.0, Ki = 0.5, Kd = 0.1;
float integral = 0, prev_error = 0;

void setup() {
  Serial.begin(115200);
  pinMode(2, OUTPUT); // Status LED
}

void loop() {
  if (Serial.available() >= 4) {
    // 1. Read Float (4 bytes)
    union { float f; byte b[4]; } u;
    Serial.readBytes(u.b, 4);
    float current_velocity = u.f;

    // 2. PID Control
    float error = target_velocity - current_velocity;
    integral += error * 0.01; // Assume 100Hz loop
    float derivative = (error - prev_error) / 0.01;
    float output = Kp * error + Ki * integral + Kd * derivative;
    prev_error = error;

    // 3. Write Float (4 bytes)
    union { float f; byte b[4]; } out;
    out.f = output;
    Serial.write(out.b, 4);
    
    // Blink LED to show activity
    digitalWrite(2, !digitalRead(2));
  }
}
```

### 👨‍💻 Code: hil_bridge.py (ROS 2 Node)

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import serial
import struct
import time

class HILBridge(Node):
    def __init__(self):
        super().__init__('hil_bridge')
        
        # Serial Connection
        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
            self.get_logger().info("Connected to ESP32")
        except:
            self.get_logger().error("Failed to connect to ESP32")
            exit(1)

        # ROS 2 Interface
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.last_time = time.time()

    def odom_callback(self, msg):
        # 1. Get Velocity from Gazebo
        v_x = msg.twist.twist.linear.x
        
        # 2. Send to ESP32 (Float, Little Endian)
        data = struct.pack('<f', v_x)
        self.ser.write(data)
        
        # 3. Read Response from ESP32
        response = self.ser.read(4)
        if len(response) == 4:
            force = struct.unpack('<f', response)[0]
            
            # 4. Publish Command to Gazebo
            cmd = Twist()
            cmd.linear.x = force # Treating force as velocity command for diff_drive
            # Note: Ideally diff_drive takes velocity, so ESP32 is outputting velocity command.
            # If we were doing force control, we'd need a force plugin.
            # Let's assume ESP32 outputs desired velocity.
            
            self.pub_cmd.publish(cmd)
            # self.get_logger().info(f"Sent: {v_x:.2f} | Recv: {force:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = HILBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 🛠️ Build & Run

1.  **Flash ESP32.**
2.  **Launch Gazebo:** `ros2 launch week7_day44 gazebo.launch.py` (RoboCar).
3.  **Run Bridge:** `python3 week7_day48/hil_bridge.py`.

---

## 🔬 Lab Exercise: Latency Injection

### Lab Objectives
1.  **Baseline:** Run the HIL loop. Observe the car accelerating to 2.0 m/s.
2.  **Inject Latency:** Modify `hil_bridge.py` to sleep for 0.1s before writing to Serial.
3.  **Observation:** The car might oscillate or become unstable.
    -   *Why?* The PID controller on the ESP32 assumes instant feedback. The delay introduces a phase lag, reducing the Phase Margin.
4.  **Fix:** Retune PID (Lower Kp) to handle the latency.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Serial Buffer Overflow
**Symptom:** Lag increases over time.
**Cause:** Python is sending data faster than ESP32 can read, or vice versa.
**Solution:** Flush buffers (`self.ser.reset_input_buffer()`) before reading/writing. Ensure loop rates match.

#### 2. Endianness Mismatch
**Symptom:** Velocity 2.0 becomes 3.4e38.
**Cause:** PC is Little Endian, Network is Big Endian? Usually PC/ARM are both Little Endian.
**Solution:** Use `struct.pack('<f')` (Little Endian) explicitly.

#### 3. Gazebo Real-Time Factor < 1.0
**Symptom:** Physics runs slow.
**Cause:** Computer too slow.
**Solution:** Simplify collision meshes. Disable shadows.

---

## ⚡ Optimization & Best Practices

### 1. High-Speed Interface
UART (115200) is slow (~10kB/s). Latency ~1ms.
For high-performance HIL:
-   Use **Ethernet (UDP)** or **CAN FD**.
-   Use **Micro-ROS** (as learned in Day 41) over USB-CDC (Virtual Serial) which is much faster than physical UART.

### 2. Lockstep Simulation
If Gazebo is slower than real-time:
-   Pause Gazebo.
-   Send state to ECU.
-   Wait for ECU reply.
-   Step Gazebo 1 tick.
-   Repeat.
-   *Note:* Requires ECU to support "Stepped Clock" (hard for real hardware, easier for PIL).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is HIL safer than Field Testing?
    *   **A:** If the controller fails in HIL, the virtual car crashes (zero cost). In Field, a real car crashes ($$$ + safety).
2.  **Q:** What is the main challenge in HIL?
    *   **A:** Latency and Synchronization. The loop (Gazebo -> PC -> Serial -> ECU -> Serial -> PC -> Gazebo) must be fast enough for the control dynamics.
3.  **Q:** Can I simulate a Camera in HIL?
    *   **A:** Yes, but sending images over Serial is impossible. You need Gigabit Ethernet or HDMI injection (Hardware Video Input to ECU).

### Challenge Task
**Task:** Steering HIL.
1.  Update ESP32 to accept `heading_error` and output `steering_angle`.
2.  Update Bridge to calculate heading error (Target Yaw - Current Yaw).
3.  Watch the physical ESP32 steer the virtual car.

---

## 📚 Further Reading & References
-   [dSPACE HIL Systems](https://www.dspace.com/en/inc/home/products/systems/hardware-in-the-loop.cfm) (Industry Standard).
-   [Renode](https://renode.io/) (Virtual HIL / PIL).

---

**Day 48 Complete** | Phase 4: ADAS & Robotics Systems | Week 7: Simulation & Modeling
