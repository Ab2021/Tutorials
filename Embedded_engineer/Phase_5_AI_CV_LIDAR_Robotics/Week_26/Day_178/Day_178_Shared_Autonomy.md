# Day 178: Shared Autonomy
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 26: Human-Robot Collaboration

---

> **📝 Content Creator Instructions:**
> Two pilots, one wheel.
> - **Focus:** Blended control, Virtual Fixtures, Haptic Feedback, and "Guardianship".
> - **Code:** `shared_controller.py`. A teleoperation node that mixes Joystick Input with Autonomous Obstacle Avoidance.
> - **Key Concept:** $U_{final} = \alpha U_{human} + (1-\alpha) U_{robot}$.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Implement** Linear Blending for Shared Control.
2.  **Define** Virtual Fixtures (Forbidden Regions) to assist teleoperation.
3.  **Arbitrate** control authority based on Confidence/Safety.
4.  **Simulate** Force Feedback (Haptics) for remote operators.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Gamepad (Xbox/DualShock) or SpaceMouse.
- Robot simulator (Gazebo).

### Software Environment
```bash
sudo apt install ros-humble-teleop-twist-joy
```

### Prior Knowledge
- Cmd_vel muxing (Day 75).
- Potential Fields.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Spectrum of Autonomy

*   **Teleoperation:** Human 100%.
*   **Shared Control:** Human guides high-level (Go there), Robot handles low-level (Don't hit wall).
*   **Supervisory:** Human monitors, Robot drives.
*   **Full Autonomy:** Human sleeping.

### 🔹 Part 2: Blending Architectures

1.  **Input Mixing:**
    $$ u_{cmd} = \alpha u_{human} + (1-\alpha) u_{auto} $$
    *   $\alpha$: Control Authority.
    *   If Risk High $\to$ $\alpha \to 0$ (Robot takes over).
2.  **Virtual Fixtures (VF):**
    *   **Guidance VF:** Funnel the user toward the goal (Like a ruler for drawing lines).
    *   **Forbidden VF:** Push user away from obstacles (Magnetic repulsion).

### 🔹 Part 3: Haptic Feedback

Instead of just ignoring the user, **Fight Back** (via Joystick Force).
*   $F_{joystick} = -k (Pos_{robot} - Pos_{obs})$.
*   User "feels" the obstacle before seeing it.

---

## 💻 Implementation: The Assisted Teleop

We blend a Joystick input with a simple Potential Field avoidance.

### 🛠️ Project Structure
```text
day178_shared/
├── src/
│   ├── shared_controller.py
│   └── joy_listener.py
└── launch/
    └── teleop_assist.launch.py
```

### 👨‍💻 Shared Controller (`src/shared_controller.py`)

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np

class SharedAutonomy(Node):
    def __init__(self):
        super().__init__('shared_autonomy')
        
        # IO
        self.create_subscription(Twist, '/joy_vel', self.joy_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # State
        self.joy_twist = Twist()
        self.auto_twist = Twist()
        self.safe_alpha = 1.0 # 1.0 = Human, 0.0 = Robot
        
        # Timer
        self.create_timer(0.05, self.control_loop) # 20Hz
        
    def joy_cb(self, msg):
        self.joy_twist = msg
        
    def scan_cb(self, msg):
        # Calculate Repulsive Vector from obstacles
        ranges = np.array(msg.ranges)
        ranges[ranges == float('inf')] = 10.0
        ranges[np.isnan(ranges)] = 10.0
        
        # Find closest point
        min_idx = np.argmin(ranges)
        min_dist = ranges[min_idx]
        
        angle = msg.angle_min + min_idx * msg.angle_increment
        
        # Simple Reactive Avoidance
        # If obstacle is close, push away
        self.auto_twist = Twist()
        
        SAFE_DIST = 1.0
        CRITICAL_DIST = 0.5
        
        if min_dist < SAFE_DIST:
            # Repulsion strength
            # Vector from Obs to Robot is -Vector(Obs)
            # angle_obs = angle
            # angle_repulse = angle + PI
            
            strength = (SAFE_DIST - min_dist) / (SAFE_DIST - CRITICAL_DIST)
            strength = np.clip(strength, 0.0, 1.0)
            
            # Auto command: Rotate away, Stop forward
            # Oppose object direction
            self.auto_twist.linear.x = -0.5 * strength # Back up?
            self.auto_twist.angular.z = -1.0 * np.sign(angle) * strength # Turn away
            
            # Authority Arbitration
            # Closer we are, less authority Human had
            self.safe_alpha = 1.0 - strength
        else:
            self.safe_alpha = 1.0
            self.auto_twist = Twist()

    def control_loop(self):
        final_cmd = Twist()
        
        # Blending logic
        # Linear X: Human wants Fwd. Robot wants Stop/Back.
        final_cmd.linear.x = self.safe_alpha * self.joy_twist.linear.x + \
                             (1 - self.safe_alpha) * self.auto_twist.linear.x
                             
        # Angular Z
        final_cmd.angular.z = self.safe_alpha * self.joy_twist.angular.z + \
                              (1 - self.safe_alpha) * self.auto_twist.angular.z
                              
        # Print status
        if self.safe_alpha < 1.0:
            self.get_logger().info(f"Assisting! Alpha: {self.safe_alpha:.2f}")
            
        self.pub_cmd.publish(final_cmd)

def main():
    rclpy.init()
    node = SharedAutonomy()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Tunnel Run"

### 1. Lab Objectives
- **Sim:** Gazebo World with a narrow crooked hallway.
- **Task:** Drive through using raw Teleop. Count collisions.
- **Task:** Enable `shared_controller.py`.
- **Observe:** "Wall Following" behavior emerges. The robot refuses to hit the wall even if you steer into it.
- **Result:** Faster completion time, fewer crashes.

---

## 🚀 Project: "Virtual Ruler"

**Goal:** Assist drawing straight lines (or driving straight).
1.  **Input:** Joystick $(x, y)$.
2.  **Constraint:** Line equation $y = 0$.
3.  **Projection:** Project user input onto the valid manifold.
    *   $u_{projected} = P \cdot u_{raw}$.
    *   If user pushes $Y$, ignore it. If user pushes $X$, allow it.
4.  **Stiffness:** Allow *some* deviation if user pushes Hard ($F > F_{break}$).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Fighting the User"
*   **Cause:** Robot tries to avoid a small box, Human tries to nudge it over. Robot spins wildly.
*   **Fix:** "Intent Recognition" (Day 177). If Human persists in an unsafe command, maybe they know better? (Or maybe E-Stop). Typically, Safety overrides, but Operationally, allow "Creep mode".

#### 2. "Oscillation"
*   **Cause:** Human corrects Left, Robot corrects Right. Loop.
*   **Fix:** Add Damping to the blending. Don't change $\alpha$ instantly. Smooth it.

---

## ⚡ Optimization: Optimization Based Blending (MPC)

Instead of weighted average, solve:
$$ \min_{u} J = ||u - u_{human}||^2 + w_{safety} \cdot \frac{1}{dist^2} $$
s.t. Dynamics constraint.
*   The optimizer tries to satisfy the Human WISHES as close as possible while respecting Constraints.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is a "Virtual Fixture"?
    *   **A:** A software-generated constraint that guides (Guide VR) or restricts (Forbidden VR) motion, simulating a physical fixture (ruler/barrier).
2.  **Q:** Why is Haptics better than just overriding?
    *   **A:** It keeps the human in the loop. They *feel* the limit, so they stop pushing. Overriding confuses the human ("Why isn't it moving??").

### Challenge Task
> **Task:** Latency Compensation.
> 1. Teleop over Internet (500ms delay).
> 2. Robot runs local safety loop.
> 3. Display "Ghost Robot" (Predictive Display) to user showing where they *will* be.

---

## 📚 Further Reading
- **Dragan et al.:** "Policy Blending for Shared Assistive Control".
- **Haptics:** Force Dimension SDK.

---

**Day 178 Complete**
