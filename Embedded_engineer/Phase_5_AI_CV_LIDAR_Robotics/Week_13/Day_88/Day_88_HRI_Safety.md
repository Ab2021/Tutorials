# Day 88: Human-Robot Interaction (HRI)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 13: Humanoid Robotics

---

> **📝 Content Creator Instructions:**
> A robot that punches you is a bad robot.
> - **Focus:** ISO 13482 Safety Standards, Collision Detection (External Torque Estimation), and Social Navigation (Intent).
> - **Code:** A "Momentum Observer" to estimate external torques $\tau_{ext}$ without force sensors, using Motor Current and Dynamics Model.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Understand** the levels of Collaboration: Coexistence $\to$ Cooperation $\to$ Collaboration.
2.  **Implement** a Collision Detector using Generalized Momentum ($p = M(q)\dot{q}$).
3.  **Design** a "Reflex" behavior: If Hit $\to$ Go Limp (Gravity Compensation).
4.  **Discuss** Psychological HRI: Eye gaze and predictable motion.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Torque-controllable actuator (Simulation works).

### Software Environment
```bash
pip install numpy pinocchio
```

### Prior Knowledge
- Rigid Body Dynamics ($M, C, G$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Safety Levels (ISO/TS 15066)

*   **Safety Rated Monitored Stop:** Robot stops when human enters zone.
*   **Speed and Separation Monitoring:** Robot slows down as human approaches.
*   **Power and Force Limiting (PFL):** Robot can hit human, but force is too low to injure. (Cobots/Humanoids).

### 🔹 Part 2: Detection without Sensors

How do we feel a touch without "Skin"?
*   **Dynamics:** $\tau_{motor} = M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) + \tau_{ext}$.
*   **Residual:** $r = \tau_{motor} - (M\ddot{q} + C\dot{q} + G)$.
*   If model is perfect, $r = \tau_{ext}$.
*   If $r > Threshold$, we collided.

### 🔹 Part 3: Momentum Observer

Evaluating $\ddot{q}$ (acceleration) is noisy.
Better approach: Integrate the momentum equation.
$$ r = K_O \left( p - \int (\tau_{motor} - C^T \dot{q} + G) dt - p(0) \right) $$
*   Smooth, filtered estimate of external torque.
*   Used by DLR and Franka Emika.

---

## 💻 Implementation: Collision Reflex

We will simulate a 1-DOF arm that goes "Limp" when hit.

### 🛠️ Project Structure
```text
day88_hri/
├── src/
│   ├── safety_node.py
│   └── dynamic_model.py
└── launch/
    └── safe_arm.launch.py
```

### 👨‍💻 Collision Monitor (`src/safety_node.py`)

Using a simplified model for demonstration.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class CollisionMonitor(Node):
    def __init__(self):
        super().__init__('collision_monitor')
        
        self.sub_state = self.create_subscription(JointState, '/joint_states', self.state_cb, 10)
        self.pub_cmd = self.create_publisher(Float64MultiArray, '/effort_controller/commands', 10)
        
        # Model Params (1 Link)
        self.mass = 1.0
        self.length = 0.5
        self.com = 0.25
        self.inertia = 0.1
        self.g = 9.81
        
        self.mode = "POSITION" # or "REFLEX"
        self.collision_threshold = 5.0 # Nm
        self.residual = 0.0

    def state_cb(self, msg):
        # Extract q, dq, effort
        idx = msg.name.index('joint1')
        q = msg.position[idx]
        dq = msg.velocity[idx]
        tau_measured = msg.effort[idx]
        
        # 1. Compute Expected Torque (Inverse Dynamics)
        # tau_expected = G(q) (Assuming static/slow)
        # G(q) = m * g * com * sin(q) (Gravity)
        tau_gravity = self.mass * self.g * self.com * np.sin(q)
        
        # Friction model?
        tau_friction = 0.5 * dq
        
        tau_expected = tau_gravity + tau_friction
        
        # 2. Residual
        self.residual = abs(tau_measured - tau_expected)
        
        # 3. Check Collision
        cmd = Float64MultiArray()
        
        if self.residual > self.collision_threshold:
            self.get_logger().warn(f"COLLISION! Res: {self.residual:.2f}")
            self.mode = "REFLEX"
            
        # 4. Control Logic
        if self.mode == "REFLEX":
            # Gravity Compensation Mode (Zero Stiffness)
            # Send torque = Gravity. The arm becomes weightless and compliant.
            cmd.data = [tau_gravity]
        else:
            # Position Control (via high level, or simulated here)
            # For this example, we just hold position
            kp = 50.0
            error = 0.0 - q
            cmd.data = [kp * error + tau_gravity]

        self.pub_cmd.publish(cmd)

def main():
    rclpy.init()
    node = CollisionMonitor()
    rclpy.spin(node)
```

---

## 🔬 Lab Exercise: "The Punch Test"

### 1. Lab Objectives
- **Setup:** Simulated Arm holding position $q=0$.
- **Action:** Apply external force in Gazebo (`ApplyForce` plugin or GUI).
- **Observation:**
    *   `residual` spikes.
    *   Mode switches to `REFLEX`.
    *   The arm stops fighting and yields to the push.
- **Recovery:** After 5 seconds of low residual, switch back to `POSITION`.

---

## 🚀 Project: "Social Navigation"

**Goal:** Humanoid walking in a crowd.
1.  **Costmap:** Add a "Poxemic Layer" (Social Spaces).
2.  **Rule:** Front of human = High Cost (Interaction Zone). Back = Medium Cost.
3.  **Experiment:**
    *   Human walks towards robot.
    *   Robot should turn right (Pass on left - Cultural norm).
    *   Robot should gaze (turn head) towards the human to signal awareness.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "False Positives"
*   **Symptom:** Robot triggers collision when accelerating fast.
*   **Cause:** The "Expected Torque" (Dynamics model) was too simple. Ignored Inertia ($M\ddot{q}$) or Coriolis.
*   **Fix:** Implement the full Rigid Body Dynamics equation in the monitor.

#### 2. "Reflex Oscillation"
*   **Symptom:** Robot switches Reflex -> Position -> Reflex rapidly.
*   **Fix:** Hysteresis. Require `residual < 1.0` for 2 seconds before resetting.

---

## ⚡ Optimization: Capacitive Skin

Dynamics based detection is slow (needs impact).
*   **Capacitive Skin:** Detects proximity (0-10cm) before touch.
*   **Action:** Pre-reflex. Relax stiffness *before* impact.
*   **Result:** "Feather touch" safety.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is Gravity Compensation?
    *   **A:** $\tau_{cmd} = G(q)$. The motors hold the robot's weight. The robot feels "weightless" to the user and can be moved by hand.
2.  **Q:** Why is "Stiffness" relevant to safety?
    *   **A:** High stiffness (Position Control) transfers all impact energy to the human. Low stiffness (impedance) absorbs the energy.
3.  **Q:** What is the "Uncanny Valley"?
    *   **A:** The dip in emotional response when a robot looks *almost* human but slightly wrong (creepy).

### Challenge Task
> **Task:** Handover.
> 1. Robot extends hand with object.
> 2. Wait for external force (Human pulling).
> 3. Detect force Pull > Threshold.
> 4. Open gripper.

---

## 📚 Further Reading
- **Haddadin (2017):** "Robot Collisions: A Survey on Detection, Isolation, and Identification".
- **ISO 10218:** Industrial Robot Safety.

---

**Day 88 Complete**
