# Day 204: Production Hardening
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 30: Capstone Project Part 2

---

> **📝 Content Creator Instructions:**
> Make it break-proof.
> - **Focus:** Converting the Prototype into a Product. Adding Watchdogs, Diagnostics, and Recovery Behaviors.
> - **Code:** `watchdog_monitor.py`. A Lifecycle Manager that checks if nodes are alive (heartbeats) and restarts them.
> - **Concept:** MTBF (Mean Time Between Failures).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Implement** a ROS 2 Lifecycle Node (Managed Node).
2.  **Create** a Diagnostic Aggregator to report system health (`/diagnostics`).
3.  **Code** a Watchdog Timer that triggers an E-Stop if topics go silent.
4.  **Define** Recovery Behaviors (e.g., "Clear Costmaps", "Home Arm").

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
sudo apt install ros-humble-diagnostic-updater ros-humble-lifecycle
```

### Prior Knowledge
- ROS 2 Lifecycle (Day 12).
- Finite State Machines (Day 200).

---

## 📖 Theoretical Hardening

### 🔹 Prototype vs Product

*   **Prototype:** Works 90% of the time. Restarted by pressing Ctrl+C.
*   **Product:** Works 99.99% of the time. Auto-restarts. Logs errors to cloud.

### 🔹 The Watchdog Pattern

Components fail.
*   **Camera:** USB disconnects. ToS (Topic of Silence).
*   **Nav:** Planner stuck in loop.
*   **Solution:** Nodes publish a `Heartbeat` (1Hz). The Watchdog checks the last timestamp. If `Now - Last > 2s`, Kill and Respawn.

### 🔹 Diagnostics

Standardized error reporting using `diagnostic_msgs`.
*   **Level:** OK, WARN, ERROR, STALE.
*   **Message:** "Battery Voltage Low (11.0V)".

---

## 💻 Implementation: The System Monitor

### 🛠️ Project Structure
```text
agribot_diagnostics/
├── src/
│   ├── system_monitor.py
│   └── lifecycle_manager.py
└── launch/
    └── monitor.launch.py
```

### 👨‍💻 System Watchdog (`src/system_monitor.py`)

Subscribes to sensor topics and checks frequency.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import time

class SystemWatchdog(Node):
    def __init__(self):
        super().__init__('system_watchdog')
        
        self.last_cam_time = time.time()
        self.last_lidar_time = time.time()
        
        self.create_subscription(Image, '/camera/color/image_raw', self.cam_cb, 1)
        self.create_subscription(LaserScan, '/scan', self.lidar_cb, 1)
        
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 1)
        self.create_timer(1.0, self.check_health)
        
    def cam_cb(self, msg):
        self.last_cam_time = time.time()
        
    def lidar_cb(self, msg):
        self.last_lidar_time = time.time()
        
    def check_health(self):
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        
        now = time.time()
        
        # Check Camera
        cam_stat = DiagnosticStatus()
        cam_stat.name = "Sensors: Camera"
        if now - self.last_cam_time > 2.0:
            cam_stat.level = DiagnosticStatus.ERROR
            cam_stat.message = "No Data (Disconnected?)"
        else:
            cam_stat.level = DiagnosticStatus.OK
            cam_stat.message = "Running"
        cam_stat.values.append(KeyValue(key="Latency", value=str(now - self.last_cam_time)))
        arr.status.append(cam_stat)
        
        # Check Lidar
        lid_stat = DiagnosticStatus()
        lid_stat.name = "Sensors: Lidar"
        if now - self.last_lidar_time > 2.0:
            lid_stat.level = DiagnosticStatus.ERROR
            lid_stat.message = "No Data"
        else:
            lid_stat.level = DiagnosticStatus.OK
        arr.status.append(lid_stat)
        
        self.diag_pub.publish(arr)

def main():
    rclpy.init()
    node = SystemWatchdog()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
```

### 👨‍💻 Managed Node Example (`src/lifecycle_driver.py`)

A mock driver that supports transitions (Configure, Activate, Deactivate).

```python
# (Detailed Lifecycle Node boilerplate typically handled by C++, 
# but python support exists via lifecycle_py)
# Concept:
# 1. on_configure(): Connect to hardware.
# 2. on_activate(): Start publishing.
# 3. on_deactivate(): Stop publishing.
# 4. on_cleanup(): Disconnect.
```

---

## 🔬 Lab Exercise: "Sabotage"

### 1. Lab Objectives
- **Run:** The `system_monitor`.
- **Sabotage:** Kill the camera node manually (`ros2 lifecycle set /camera_driver deactivate`).
- **Observe:** `/diagnostics` topic showing ERROR for Camera.
- **Recover:** Write a script that listens to `/diagnostics`. If Camera is ERROR, try to respawn/activate it automatically.

---

## 🚀 Project Steps

1.  **Nav2 Recovery:** Configure the behavior tree to use `Spin`, `BackUp`, and `Wait` recoveries.
2.  **MoveIt Watchdog:** If Arm planning fails 5 times in a row, trigger "Re-Homing" (Move to known safe pose).
3.  **Network:** Handle `rtt` (Round Trip Time) spikes. If Wifi lags, slow down the robot.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "False Positives"
*   **Cause:** CPU load spike causes topic delay > 2s. Watchdog kills a healthy node.
*   **Fix:** Increase timeout to 5s. Or use Windowed Average.

#### 2. "Zombie Nodes"
*   **Cause:** Watchdog tries to kill node, but node is stuck in C-level loop and ignores SIGINT.
*   **Fix:** Use `SIGKILL` (Force Kill) after 5s timeout.

---

## ⚡ Optimization: Hardware Watchdog

Software can freeze (OS Kernel Panic).
*   **External Watchdog:** An Arduino/STM32 that listens for a "Tick" pin toggle from the PC.
*   **Action:** If PC stops toggling Tick for 1s, Arduino cuts power to Motors (Relay). **Crucial for Safety.**

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the `lifecycle_manager`?
    *   **A:** A node that orchestrates the state transitions of other Lifecycle nodes (e.g., ensuring Map Server is Active before AMCL starts).
2.  **Q:** Why use `/diagnostics` instead of `print()`?
    *   **A:** Aggregation. A tool like `rqt_robot_monitor` can visualize the health of 100 subsystems in one view.

### Challenge Task
> **Task:** "Battery Monitor".
> 1. Subscribe to `/battery_state` (Sensor data).
> 2. Publish Diagnostic Warning if V < 11.5V.
> 3. Publish Diagnostic Error if V < 10.5V.
> 4. Trigger "Auto-Dock" behavior on Warning.

---

## 📚 Further Reading
- **ROS 2 Design:** "Lifecycle Management".
- **Safety Critical Systems:** ISO 26262 (Automotive Safety) basics.

---

**Day 204 Complete**
