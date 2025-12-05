# Day 61: Digital Twins (Real-Sim Sync)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 9: Simulation & Sim-to-Real

---

> **📝 Content Creator Instructions:**
> A Simulation running in parallel with Reality.
> - **Focus:** MQTT/ROS Bridge, State Synchronization (Shadow Mode), and Predictive Maintenance.
> - **Code:** A system where the Gazebo robot mimics the Real robot's localized pose and joint states in real-time.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Establish** a bi-directional data pipelines between a Physical Robot and a Digital Twin (Gazebo).
2.  **Synchronize** Transforms (TF) and Joint States (`/joint_states`) with minimal latency.
3.  **Visualize** the differentiation between "Planned State" (Sim) and "Actual State" (Real).
4.  **Implement** a "Collision Prediction" monitor (Sim runs 2s ahead of reality).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- A Real Robot (Optional) or a second distinct simulation instance.

### Software Environment
```bash
sudo apt install ros-humble-rosbridge-suite
```

### Prior Knowledge
- TF2 Tree (Day 3).
- MQTT / WebSockets (IoT).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is a Digital Twin?

Not just a 3D model. It's a **Live** model.
*   **Level 1 (Visual):** Rviz visualization of current state.
*   **Level 2 (Shadow):** Sim physics engine mirrors real forces/torques. Used to estimate unmeasurable params (e.g., Payload mass).
*   **Level 3 (Predictive):** Sim runs faster than real-time to predict failures ("If you continue this trajectory, you will overheat in 10s").

### 🔹 Part 2: Data Pipeline

Real Robot (Edge) $\to$ WiFi $\to$ Workstation (Sim).
*   **Bandwidth:** Cannot send raw camera feed (300MB/s). Send perception *results* (Object Poses).
*   **Latency:** Time Delay ($t_{real} \to t_{sim}$) causes instability if we try to close the control loop through Sim.
*   **Clock Sync:** Sim Time vs Real Time (Wall Clock). `use_sim_time=True/False` hell.

---

## 💻 Implementation: The Shadow Bot

The Real Robot publishes `/joint_states`. The Sim Robot subscribes and moves to match.
We use `ros_gz_bridge` (ROS-Gazebo Bridge) or just a custom node if pure ROS.

### 🛠️ Project Structure
```text
day61_digital_twin/
├── launch/
│   └── twin.launch.py
└── src/
    └── particle_sync.py
```

### 👨‍💻 Sync Node (`src/twin_sync.py`)

A node that listens to Real Robot TF and pushes the Sim Robot Model.
*   **Note:** We cannot use `diff_drive_controller` in Sim because we are not *controlling* it, we are *forcing* it to be where the real one is.
*   **approach:** "Kinematic Mode". Overwrite physics state.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose

class DigitalTwinSync(Node):
    def __init__(self):
        super().__init__('digital_twin_sync')
        
        # Subscriber to Real Robot
        self.sub_joints = self.create_subscription(
            JointState, '/real_robot/joint_states', self.joint_cb, 10
        )
        
        # Subscriber to Real Robot Localization (AMCL)
        self.sub_pose = self.create_subscription(
            Pose, '/real_robot/pose', self.pose_cb, 10
        )
        
        # Client to Gazebo
        self.client_set_state = self.create_client(SetEntityState, '/set_entity_state')
        
    def joint_cb(self, msg):
        # In a real implementation, we map joint names 1:1
        pass
        
    def pose_cb(self, msg):
        # Call Gazebo Service to teleport simulated robot
        # This bypasses physics! (Kinematic Shadow)
        req = SetEntityState.Request()
        req.state.name = 'my_bot'
        req.state.pose = msg
        req.state.reference_frame = 'world'
        
        future = self.client_set_state.call_async(req)
```

### 👨‍💻 Predictive Monitor (`src/predictor.py`)

Simulate 2 seconds into future.
Requires a *second* Gazebo instance running faster (RTF=5.0).
*   **Input:** Current Real State. Current Planned Trajectory.
*   **Process:** Reset Sim to Real State. Execute Plan.
*   **Output:** Did Sim crash? If yes, send E-Stop to Real Robot.

```python
# Pseudo-code logic for Predictive Safety
def safety_loop():
    real_state = get_real_state()
    plan = get_nav_plan()
    
    sim_env.reset(real_state)
    sim_env.step(plan, duration=2.0) # Fast forward
    
    if sim_env.check_collision():
        send_real_robot_stop()
```

---

## 🔬 Lab Exercise: The "Ghost"

### 1. Lab Objectives
- Open Rviz.
- Show **Real Robot** (Red Model) using `/real/joint_states`.
- Show **Sim Robot** (Blue Model) using `/sim/joint_states`.
- **Action:** Push the real robot (if possible) or teleop it.
- **Observe:** The Blue Ghost should follow the Red Robot with some lag.
- **Lag Measurement:** Compute $\Delta t = t_{arrival\_sim} - t_{departure\_real}$.

---

## 🚀 Project: "AR Debugging"

**Goal:** Overlay Sim data on Real Camera feed.
1.  **Setup:** Real Camera looking at scene.
2.  **Calibration:** Calibrate Extrinsics (Camera pose wrt World).
3.  **Rendering:** Render the Sim World (e.g., Costmap, Planned Path) from the *same* virtual viewpoint.
4.  **Overlay:** Alpha Blend Sim render on top of Real Video.
5.  **Result:** You see the "Mind" of the robot (Path lines) drawing on the floor in the video.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "TF Loop" (Tree fights)
*   **Symptom:** Rviz flickers between two positions.
*   **Cause:** Both Real Robot and Sim Robot are publishing `base_link` $\to$ `odom` transform onto the *same* `/tf` topic.
*   **Fix:** **Namespacing**.
    *   Real: `/real/tf`, `frame_id=real_map`.
    *   Sim: `/sim/tf`, `frame_id=sim_map`.

#### 2. "Clock Skew"
*   **Symptom:** TF errors "Message too old".
*   **Cause:** Sim uses Simulated Time (starts at 0). Real uses Wall Time (Epoch).
*   **Fix:** Set `use_sim_time=False` for the real robot nodes. Keep strict separation.

---

## ⚡ Optimization: Delta Compression

Sending full JointStates (array of 64-bit floats) at 100Hz is wasteful.
*   **Deadband:** Only send if $\Delta \theta > 0.01$ rad.
*   **Compression:** Quantize to 16-bit integers.
*   Critical for Cloud-based Digital Twins (AWS RoboMaker).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Kinematic vs Dynamic Twin?
    *   **A:** Kinematic only matches Position (Visual). Dynamic matches Forces (Torque). Dynamic requires solving Inverse Dynamics in Sim.
2.  **Q:** Why run Sim faster than Real?
    *   **A:** To predict the future. If Sim runs at 1x speed, it can only tell us what is happening *now* (or past).
3.  **Q:** How to sync "World" state?
    *   **A:** Hard. If a human moves a chair in real life, Sim doesn't know. Need Perception (Object Detection) to update Sim World.

### Challenge Task
> **Task:** Payload Estimation.
> 1. Real Robot lifts a mystery box. Motors work harder.
> 2. Digital Twin (Sim) lifts a 0kg box. Motors work less.
> 3. Error = $\tau_{real} - \tau_{sim}$.
> 4. Use Error to estimate Mass of box (gradient descent on Sim parameter).

---

## 📚 Further Reading
- **Digital Twin for Industry 4.0:** Concepts and Architectures.
- **AWS RoboMaker:** Cloud simulation.

---

**Day 61 Complete**
