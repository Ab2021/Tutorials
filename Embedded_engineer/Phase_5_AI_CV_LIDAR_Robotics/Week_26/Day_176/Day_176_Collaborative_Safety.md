# Day 176: Collaborative Robot Safety
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 26: Human-Robot Collaboration

---

> **📝 Content Creator Instructions:**
> "Safety First" isn't just a slogan; it's a legal requirement.
> - **Focus:** ISO/TS 15066, The 4 Modes of Collaboration, Risk Assessment, and Implementation of Speed and Separation Monitoring (SSM).
> - **Code:** `safety_monitor_ssm.py`. A ROS 2 Lifecycle Node that calculates the dynamic "Protective Separation Distance" based on robot speed and human speed, clamping velocity if violated.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Distinguish** between the 4 modes of Collaborative Operation (SMS, HG, SSM, PFL).
2.  **Calculate** the Minimum Protective Distance $S_p$ according to ISO 15066 formulas.
3.  **Implement** a Speed and Separation Monitoring (SSM) system using Python and ROS 2.
4.  **Perform** a basic Risk Assessment (HARA) for a cobot application.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Lidar or Depth Camera (Or simulation of one) to track Human position.
- Robot Arm (UR5/Franka) simulation.

### Software Environment
```bash
sudo apt install ros-humble-moveit-servo
pip install scipy
```

### Prior Knowledge
- ROS 2 Lifecycle Nodes.
- Kinematics (Jacobians).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Cobot" Misconception

A "Cobot" (e.g., UR5) is not safe by default. It is a tool that *can be used safely* in a collaborative application.
If you strap a knife to a UR5, it's dangerous.

### 🔹 Part 2: ISO/TS 15066: The Bible of Collaboration

Defined 4 modes:
1.  **Safety-rated Monitored Stop (SMS):** Robot stops when Human enters workspace. Resumes when they leave. (Most common, easiest).
2.  **Hand Guiding (HG):** Robot moves only when held by operator (Zero-gravity mode).
3.  **Speed and Separation Monitoring (SSM):** Robot slows down as Human gets closer. Stops before contact.
4.  **Power and Force Limiting (PFL):** Contact is allowed, but energy is low enough to not cause pain/injury.

### 🔹 Part 3: Calculate Safe Distance (SSM)

Formula from ISO 15066:
$$ S_p = S_h + S_r + S_s + C + Z_d + Z_r $$

*   $S_h$: Human speed contribution (assume 1.6 m/s or measured).
*   $S_r$: Robot stopping distance (reaction time * speed + braking distance).
*   $S_s$: Sensor uncertainty.
*   $C$: Intrusion distance (arm reach).
*   $Z_d, Z_r$: Position uncertainties.

**Key Insight:** As robot speed increases, $S_r$ increases quadratically (kinetic energy), so $S_p$ grows. To keep $S_p$ small, you must drive slowly.

---

## 💻 Implementation: The SSM Controller

We will build an SSM Monitor.
*   **Input:** `/human_centroids` (List of points), `/joint_states` (Robot speed).
*   **Logic:** Calculate $S_p$. Compare to measured distance $D$. Scale Max Allowable Velocity.
*   **Output:** `/cmd_vel_limit` (sent to MoveIt Servo).

### 🛠️ Project Structure
```text
day176_safety/
├── src/
│   ├── ssm_monitor.py
│   └── risk_calc_iso15066.py
└── config/
    └── safety_params.yaml
```

### 👨‍💻 Risk Calculator (`src/risk_calc_iso15066.py`)

```python
import math

class ISO15066Calculator:
    def __init__(self):
        # Constants from Standards/Risk Assessment
        self.T_reaction_robot = 0.1 # 100ms system latency
        self.T_reaction_human = 0.0 # Sometimes considered
        self.V_human = 1.6 # m/s (Standard walking speed assumption)
        self.C = 0.2 # m (Intrusion distance, e.g. reaching arm)
        self.Z = 0.05 # m (Uncertainty)
        
    def calculate_stopping_distance(self, v_robot, accel_max):
        """
        Distance traveled while stopping.
        d = v*t + (v^2)/(2a)
        """
        dist_reaction = v_robot * self.T_reaction_robot
        dist_braking = (v_robot**2) / (2 * accel_max)
        return dist_reaction + dist_braking
        
    def calculate_protective_separation(self, v_robot, accel_max):
        """
        Returns required Sp
        """
        S_r = self.calculate_stopping_distance(v_robot, accel_max)
        S_h = self.V_human * (self.T_reaction_robot + (v_robot / accel_max)) 
        # Note: ISO formula variations exist. Simplifying: Human travels during robot breaking time.
        
        Sp = S_h + S_r + self.C + self.Z
        return Sp

def main():
    calc = ISO15066Calculator()
    v_rob = 0.5 # m/s
    acc = 2.0 # m/s^2
    sp = calc.calculate_protective_separation(v_rob, acc)
    print(f"For V_rob={v_rob} m/s, Human Speed={calc.V_human} m/s")
    print(f"Required Separation Sp = {sp:.3f} m")
    
    # Reverse check: If Distance is 1.0m, what is max V_rob?
    # This requires solving quadratic eq.
    
if __name__ == "__main__":
    main()
```

### 👨‍💻 SSM Node (`src/ssm_monitor.py`)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import PointStamped
import numpy as np

class SSMMonitor(Node):
    def __init__(self):
        super().__init__('ssm_monitor')
        
        # Parameters
        self.declare_parameter('max_accel', 2.0)
        self.max_accel = self.get_parameter('max_accel').value
        
        # State
        self.robot_pos = np.array([0.0, 0.0, 0.0]) # Simplified, ideally from TF
        self.human_dist = 999.0
        self.current_rob_vel = 0.0
        
        # Subs
        self.create_subscription(Float64MultiArray, '/human_centroids', self.human_cb, 10)
        self.create_subscription(Float64MultiArray, '/joint_velocities', self.joint_cb, 10) # Simplified
        
        # Pubs
        self.vel_scaling_pub = self.create_subscription(Float64, '/servo_server/velocity_scaling_factor', self.empty_cb, 10) # Mock
        self.vel_scaling_pub = self.create_publisher(Float64, '/servo_server/velocity_scaling_factor', 10)
        
        # Loop
        self.timer = self.create_timer(0.02, self.control_loop) # 50Hz
        
        self.T_react = 0.150 # 150ms
        self.V_human = 1.6
        self.C_Z = 0.25
        
    def empty_cb(self, msg): pass
    
    def joint_cb(self, msg):
        # Estimate TCP velocity from joints (Jacobian would be used here)
        # Using a scalar proxy for demo
        self.current_rob_vel = np.mean(np.abs(msg.data)) 
        
    def human_cb(self, msg):
        # Find closest human point
        points = np.array(msg.data).reshape(-1, 3)
        if len(points) == 0:
            self.human_dist = 999.0
            return
            
        dists = np.linalg.norm(points - self.robot_pos, axis=1)
        self.human_dist = np.min(dists)
        
    def control_loop(self):
        # 1. Calc Required Sp for *Current* Velocity
        # If Sp > current_dist, we are UNSAFE -> TRIGGER STOP
        
        # 2. Calc Max Safe Velocity for *Current* Distance
        # We want to find V_limit such that calc_Sp(V_limit) == self.human_dist
        
        # S_p(v) = v*Tr + v^2/2a + Vh*(Tr + v/a) + C
        # Quadratic: A*v^2 + B*v + C_eq = 0
        # A = 1/(2a)
        # B = Tr + Vh/a
        # C_eq = Vh*Tr + C + Z - D_actual
        
        A = 1.0 / (2 * self.max_accel)
        B = self.T_react + (self.V_human / self.max_accel)
        C_eq = (self.V_human * self.T_react) + self.C_Z - self.human_dist
        
        # Roots of Av^2 + Bv + C = 0
        # v = (-B + sqrt(B^2 - 4AC)) / 2A
        
        delta = B**2 - 4*A*C_eq
        
        scaling = 1.0
        
        if self.human_dist > 5.0:
            scaling = 1.0
        elif delta < 0:
            # No real solution? Means even at v=0 we are violating margins? (Stop immediately)
            # Actually if C_eq is positive, it means D_actual is very small.
            # If C_eq > 0, dist is too small for basic static spacing.
            scaling = 0.0
        else:
            v_max_safe = (-B + np.sqrt(delta)) / (2*A)
            
            # Clamp
            if v_max_safe < 0: v_max_safe = 0.0
            
            # Convert to scaling factor (assuming max robot vel is 1.0 m/s)
            scaling = np.clip(v_max_safe / 1.0, 0.0, 1.0)
            
        # Hysteresis/Smoothing would go here
        
        msg = Float64()
        msg.data = float(scaling)
        self.vel_scaling_pub.publish(msg)
        
        if scaling < 0.1:
            self.get_logger().warn(f"Speed Limit Active! Dist: {self.human_dist:.2f}m -> Scale: {scaling:.2f}")

def main():
    rclpy.init()
    node = SSMMonitor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Invisible Safety Wall"

### 1. Lab Objectives
- **Run:** ROS 2 Simulation with a moving "Human" obstacle (Simulated TF frame).
- **Observe:** As human approaches, `velocity_scaling_factor` drops.
- **Fail Case:** Set `T_reaction` to 0.01. Observe robot doesn't stop in time (in physics sim) and hits human.
- **Correct:** Set `T_reaction` to 0.2 (realistic). Robot stops 5cm before contact.

---

## 🚀 Project: "Safety Bubble Visualization"

**Goal:** RViz Marker Array.
1.  **Subscribe:** `/human_dist`, `/current_vel`.
2.  **Calculate:** $S_p$ (Protective Separation).
3.  **Draw:**
    *   **Red Sphere:** Around robot TCP, radius = $S_p$.
    *   **Green Sphere:** Around Human, radius = $S_h$.
4.  **Visualize:** If Red and Green intersect, it's a violation.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Robot Stops Too Early"
*   **Cause:** $V_h$ set to 1.6 m/s (fast walk) but human is standing still.
*   **Fix:** Use perception to estimate actual $V_h$, but ISO says you *must* assume max reasonable speed unless you have a safety-rated speed tracker.

#### 2. "Latency Jitter"
*   **Cause:** Perception pipeline takes 200ms sometimes.
*   **Fix:** Your `T_reaction` MUST equal the *Worst Case* execution time, not the average. If you miss deadlines, you are unsafe.

---

## ⚡ Optimization: Dynamic Zones

Instead of a single $S_p$, divide workspace into:
*   **Green Zone:** Far away. Full speed.
*   **Yellow Zone:** SSM Active. Speed = f(Distance).
*   **Red Zone:** SMS (Stop). Immediate Halt.

Implementation: Polygon checking (Point-in-Polygon) is faster than calculating quadratic roots every cycle for complex shapes.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Which is safer: Power & Force Limiting (PFL) or Safety Monitored Stop (SMS)?
    *   **A:** Trick question. SMS prevents contact. PFL allows contact but limits injury. PFL is "safer" for collaboration, SMS is "safer" to avoid any touch.
2.  **Q:** Why does robot mass matter?
    *   **A:** It doesn't appear in the $S_p$ formula explicitly, but it determines the Braking Distance (Deceleration capability). Heavy robot = longer stopping dist = larger $S_p$.
3.  **Q:** Can I use a standard webcam for SSM?
    *   **A:** **NO.** You need a PL-d (Performance Level d) Safety Rated sensor (e.g., Sick, Pilz) for a compliant industrial cell.

### Challenge Task
> **Task:** Implement "Directional SSM".
> 1. If human is 1m away but walking *away* from robot, do we need to stop?
> 2. Modify formula to use relative velocity vector $\vec{V}_{rel} = \vec{V}_{rob} - \vec{V}_{human}$.
> 3. Only penalize if $\vec{V}_{rel}$ is closing.

---

## 📚 Further Reading
- **ISO/TS 15066:** The full specification.
- **NIST:** Guidelines for Collaborative Robot Safety.

---

**Day 176 Complete**
