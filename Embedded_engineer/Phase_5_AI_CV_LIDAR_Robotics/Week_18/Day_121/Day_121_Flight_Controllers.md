# Day 121: Flight Controllers (PX4/ArduPilot)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 18: Aerial & Underwater Robotics

---

> **📝 Content Creator Instructions:**
> We don't write PIDs from scratch in production. We use PX4.
> - **Focus:** The PX4 Autopilot ecosystem, MAVLink, MicroXRCE-DDS (The ROS 2 Bridge), and Offboard Control.
> - **Code:** A ROS 2 node `offboard_control.py` that waits for GPS lock, Arms the drone, switches to OFFBOARD mode, and executes a square flight pattern.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the architecture: Flight Controller (Real-time safety) $\leftrightarrow$ Companion Computer (High-level AI/ROS 2).
2.  **Bridge** MAVLink to ROS 2 using `micro_ros_agent` or `MicroXRCE-DDS`.
3.  **Command** the drone using `TrajectorySetpoint` messages (Position/Velocity/Acceleration).
4.  **Handle** Failsafes (RTL - Return to Launch on loss of ROS heartbeat).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Simulation (PX4 SITL + Gazebo).
- Real: Pixhawk 6C / Holybro (Optional).

### Software Environment
```bash
# PX4 Setup
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
./PX4-Autopilot/Tools/setup/ubuntu.sh
pip install px4-msgs
```

### Prior Knowledge
- Coordinate Frames (NED vs ENU). PX4 uses NED internally! ROS uses ENU. The bridge handles conversion.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Split Brain

*   **Autopilot (Cortex M7):** Runs PX4. Hard Real-Time. Handles Stabilization (Rate Controller), GPS Fusion (EKF2), Battery failsafe.
*   **Companion (Jetson/Pi):** Runs ROS 2. Handles Obstacle Avoidance, SLAM, Mission Planning.
*   **Protocol:** MAVLink (Serial).
*   **Bridge:**
    *   Old: `mavros` (MAVLink $\leftrightarrow$ ROS 1/2). Heavy.
    *   New: `MicroXRCE-DDS`. PX4 speaks DDS natively (v1.14+).

### 🔹 Part 2: Offboard Control Logic

To control the drone from ROS:
1.  **Heartbeat:** Maintain $> 2Hz$ stream of Setpoints (Safety requirement).
2.  **Arming:** Send `VehicleCommand(CMD_COMPONENT_ARM_DISARM, 1)`.
3.  **Mode Switch:** Send `VehicleCommand(CMD_DO_SET_MODE, OFFBOARD)`.
4.  **Control:** Update `TrajectorySetpoint`.

### 🔹 Part 3: Frame Conversions

*   **PX4:** FRD (Front-Right-Down) or NED (North-East-Down). Z is Down (Altitude -10m = 10m high).
*   **ROS:** FLU (Front-Left-Up) or ENU (East-North-Up). Z is Up.
*   **The Bridge:** Usually handles the rotation $R_x(180)$. Be careful doing math manually.

---

## 💻 Implementation: Offboard Square

We assume PX4 SITL is running (`make px4_sitl gazebo`).

### 🛠️ Project Structure
```text
day121_px4/
├── src/
│   ├── offboard_control.py
└── launch/
    ├── sitl_bridge.launch.py
```

### 👨‍💻 Offboard Node (`src/offboard_control.py`)

Using `px4_msgs`.

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus, VehicleOdometry
import numpy as np

class OffboardControl(Node):
    def __init__(self):
        super().__init__('offboard_control')

        # QoS for PX4 (Best Effort is critical for high freq telemetry)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.pub_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.pub_trajectory = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.pub_vehicle_command = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Subscribers
        self.sub_status = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.status_cb, qos_profile)
        self.sub_odom = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_cb, qos_profile)

        # State
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.takeoff_height = -5.0 # NED! (5m Up)
        self.cnt = 0
        
        # Waypoints (NED)
        self.waypoints = [
            [0, 0, -5],
            [10, 0, -5],
            [10, 10, -5],
            [0, 10, -5],
            [0, 0, -5]
        ]
        self.wp_idx = 0

        # Timer (10Hz is minimal, 20-50Hz recommended)
        self.create_timer(0.05, self.loop)

    def status_cb(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def odom_cb(self, msg):
        self.current_pos = np.array([msg.position[0], msg.position[1], msg.position[2]])

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_vehicle_command.publish(msg)

    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

    def engage_offboard(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0) # 1=Custom, 6=Offboard

    def loop(self):
        # 1. Publish Heartbeat (Required to allow mode switch)
        offboard_msg = OffboardControlMode()
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_offboard_mode.publish(offboard_msg)

        # 2. Logic (Once per second check state)
        if self.cnt % 20 == 0:
             if self.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                 self.get_logger().info("Requesting Arming...")
                 self.arm()
             elif self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                 self.get_logger().info("Requesting Offboard Mode...")
                 self.engage_offboard()

        # 3. Publish Setpoint
        # Check if reached waypoint
        dist = np.linalg.norm(self.current_pos - self.waypoints[self.wp_idx])
        if dist < 0.5:
            self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
            self.get_logger().info(f"Reached WP. Next: {self.waypoints[self.wp_idx]}")

        target = self.waypoints[self.wp_idx]
        
        traj_msg = TrajectorySetpoint()
        traj_msg.position = [float(target[0]), float(target[1]), float(target[2])]
        traj_msg.yaw = 0.0 # North
        traj_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_trajectory.publish(traj_msg)
        
        self.cnt += 1

def main():
    rclpy.init()
    rclpy.spin(OffboardControl())
```

---

## 🔬 Lab Exercise: "Digital Twin"

### 1. Lab Objectives
- **Launch:** `MicroXRCEAgent` (udp4 8888).
- **Run:** PX4 Simulator (`make px4_sitl gazebo`).
- **Run:** `ros2 run day121_px4 offboard_control`.
- **Observe:** Drone arms, rises to 5m, flies 10x10m square, loops forever.
- **Fail Check:** Kill the ROS node.
- **Observe:** PX4 detects "Offboard signal lost". Failsafe triggers (Default: Land or Return to Launch).

---

## 🚀 Project: "Follow the AprilTag"

**Goal:** Land on a moving target.
1.  **Vision:** Camera detects AprilTag on ground.
2.  **Transform:** Calculate Tag position in NED frame relative to Drone.
3.  **Control:** Set `TrajectorySetpoint.position = [tag.x, tag.y, -2.0]`. Use Velocity control for smoother tracking.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Drone Rejects Offboard"
*   **Cause:** Not sending setpoint stream *before* requesting mode switch.
*   **Fix:** Ensure `pub_trajectory` runs for at least 1 second before `engage_offboard`.

#### 2. "Flyaway Reversed"
*   **Cause:** Sending [0, 0, 5] instead of [0, 0, -5].
*   **Result:** Drone tries to dig 5 meters into the ground (or disarms instantly on ground contact).

---

## ⚡ Optimization: Velocity Smoothing

Step inputs (Square waypoints) cause jerky motion.
*   **S-Curve Profile:** generating smooth Velocity/Acceleration ramps.
*   PX4 has internal parameters `MPC_Acc_Hor`, `MPC_Jerk_Auto`. Tuning these makes the drone fly cinematically.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is NED?
    *   **A:** North East Down. Standard aerospace frame. Z is Down.
2.  **Q:** Why not just use `cmd_vel`?
    *   **A:** You can (Velocity Control), but Position Control is safer for precise waypoint missions. Offboard supports both.
3.  **Q:** What happens if the Companion Computer crashes?
    *   **A:** PX4 Failsafe triggers. Usually "Return to Launch". The drone does not fall out of the sky.

### Challenge Task
> **Task:** Circle Mode.
> 1. Set $x = R \cos(t), y = R \sin(t)$.
> 2. Set $yaw = t + \pi/2$ (Face tangent to circle).
> 3. Fly a smooth circle while facing forward.

---

## 📚 Further Reading
- **PX4 Docs:** "ROS 2 User Guide".
- **MAVSDK:** Python interface alternative to ROS 2.

---

**Day 121 Complete**
