# Day 40: Formation Control (Leader-Follower)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 6: Multi-Robot Systems

---

> **📝 Content Creator Instructions:**
> Swarms are messy. Formations are rigid (Military style).
> - **Focus:** Leader-Follower, Virtual Structure, and Graph Rigidity.
> - **Code:** Controlling 3 robots to maintain a Triangle Formation while maneuvering.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Classify** Formation Strategies: Leader-Follower ($l-\psi$) vs Virtual Structure.
2.  **Derive** the Control Law for a follower to maintain distance $d$ and angle $\phi$ relative to a leader.
3.  **Analyze** String Stability: Does the error amplify as it propagates down the chain (Traffic jams)?
4.  **Implement** a Formation Controller for a convoy of differential drive robots.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Feedback Linearization (Day 22/23).
- Differential Drive Kinematics.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Leader-Follower ($l-\psi$ Control)

Robot $L$ (Leader) moves. Robot $F$ (Follower) tracks $L$.
Define Error in Follower's frame:
$$ e_x = x_L - x_F - d \cos(\theta_F + \psi) $$
$$ e_y = y_L - y_F - d \sin(\theta_F + \psi) $$
*   $d$: Desired distance.
*   $\psi$: Desired bearing angle relative to Follower's heading.
*   **Controller:** Use Input-Output Linearization to drive $e_x, e_y \to 0$.

### 🔹 Part 2: Virtual Structure

Imagine a rigid "Virtual Triangle" moving in space.
1.  User commands the Virtual Structure (VS) velocity $(v, \omega)$.
2.  Each robot calculates where its "slot" in the VS should be.
3.  Each robot tracks its moving reference point.
*   *Pros:* High precision. If one robot lags, the whole VS can slow down (Cooperative).
*   *Cons:* High communication requirement.

### 🔹 Part 3: Graph Rigidity

If we use distance-only constraints (maintain dist $d$ to neighbor), is the shape unique?
*   **Rigid:** (Triangle). 3 agents, 3 edges. Shape is fixed.
*   **Flexible:** (Square). 4 agents, 4 edges. Can shear into a Rhombus.
*   **Fix:** Add diagonal constraint (Cross-bracing) to make square rigid. Formations must be Rigid Graphs to be stable.

---

## 💻 Implementation: Leader-Follower Platoon

Scenario: 3 Robots in a line (Convoy).
$R_0$ (Leader) -> $R_1$ (Follows $R_0$) -> $R_2$ (Follows $R_1$).

### 🛠️ Project Structure
```text
day40_formation/
├── src/
│   ├── robot.py
│   ├── controller.py
└── run_convoy.py
```

### 👨‍💻 Robot Model (`src/robot.py`)

```python
import numpy as np

class Robot:
    def __init__(self, id, x, y, theta):
        self.id = id
        self.state = np.array([x, y, theta])
        self.v = 0.0
        self.w = 0.0
        
    def step(self, v, w, dt=0.1):
        x, y, theta = self.state
        self.state[0] += v * np.cos(theta) * dt
        self.state[1] += v * np.sin(theta) * dt
        self.state[2] += w * dt
        self.v = v
        self.w = w
        return self.state
```

### 👨‍💻 Controller Logic (`src/controller.py`)

Using Feedback Linearization for Follower.

```python
def follower_control(leader_state, follower_state, des_dist, des_angle):
    x_l, y_l, theta_l = leader_state
    x_f, y_f, theta_f = follower_state
    
    # Error in Follower Frame
    # We want Lead to be at (d, psi) from Follower
    # For a convoy, psi=0 (Behind leader? No, psi=0 means leader is IN FRONT)
    
    # Calculate current relative position (Cartesian)
    dx_world = x_l - x_f
    dy_world = y_l - y_f
    
    # Rotate to Follower Frame
    dx_b = np.cos(theta_f)*dx_world + np.sin(theta_f)*dy_world
    dy_b = -np.sin(theta_f)*dx_world + np.cos(theta_f)*dy_world
    
    # Desired position in Body Frame
    target_x = des_dist * np.cos(des_angle)
    target_y = des_dist * np.sin(des_angle)
    
    # Error
    e_x = target_x - dx_b
    e_y = target_y - dy_b # Cross track
    
    # Control Law (Simple P-Controller on Linearized system)
    k_v = 1.0
    k_w = 4.0
    
    # Feedforward (Leader velocity assumption: 0 or communicated)
    v_cmd = k_v * (dx_b - target_x) # Try to close gap
    w_cmd = k_w * (dy_b - target_y) # Try to center bearing
    
    # Saturation (Prevent crazy spins)
    v_cmd = np.clip(v_cmd, 0.0, 2.0)
    w_cmd = np.clip(w_cmd, -1.0, 1.0)
    
    return v_cmd, w_cmd
```

### 👨‍💻 Simulation (`run_convoy.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
from src.robot import Robot
from src.controller import follower_control

r0 = Robot(0, 10, 10, 0) # Leader
r1 = Robot(1, 5, 10, 0) # Follower 1
r2 = Robot(2, 0, 10, 0) # Follower 2

history = []

for t in range(200):
    # Leader moves in S-shape
    v0 = 1.0
    w0 = 0.5 * np.sin(t * 0.1)
    r0.step(v0, w0)
    
    # Follower 1 follows R0
    v1, w1 = follower_control(r0.state, r1.state, des_dist=2.0, des_angle=0.0)
    r1.step(v1, w1)
    
    # Follower 2 follows R1
    v2, w2 = follower_control(r1.state, r2.state, des_dist=2.0, des_angle=0.0)
    r2.step(v2, w2)
    
    history.append([r0.state, r1.state, r2.state])

# Plotting...
```

---

## 🔬 Lab Exercise: The Cutting Corner

### 1. Lab Objectives
- Run the simulation with $R_0$ turning 90 degrees sharply.
- **Observe:** $R_1$ cuts the corner (looks like a triangle trajectory), it does not trace $R_0$'s path exactly.
- **Why?** Because the controller drives *straight towards* the leader.
- **Fix:** **Breadcrumb Following**. $R_0$ leaves virtual points. $R_1$ follows the *path history*, not the current position.

---

## 🚀 Project: "Triangle Formation"

**Goal:** Maintain an equilateral triangle.
1.  **Follower 1:** Follows Leader at $(d=2, \psi=150^\circ)$. (Left Wing).
2.  **Follower 2:** Follows Leader at $(d=2, \psi=-150^\circ)$. (Right Wing).
3.  **Result:** "V" Formation (Geese/Fighter Jets).
4.  **Challenge:** When Leader turns left, Right Wing must speed up, Left Wing must slow down.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "String Instability" (Traffic Jam)
*   **Symptom:** $R_0$ brakes slightly. $R_1$ brakes hard. $R_2$ stops. $R_3$ reverses/crashes. Error amplifies.
*   **Cause:** Controller delay or overshoot.
*   **Fix:** Ensure $\text{Gain} < 1$ in frequency domain (Bode plot). Or communicate Leader velocity to *everyone* immediately (Feedforward).

#### 2. "Collision"
*   **Symptom:** During sharp turn, wingmen collide.
*   **Fix:** Add potential field repulsion between followers (Swarm-like safety layer).

---

## ⚡ Optimization: Distributed Receding Horizon

Combine MPC + Formation.
*   Each robot solves MPC optimization to maintain formation.
*   Exchange predicted trajectories.
*   Cost function includes: $J = ||p - p_{leader}||^2 + ||p - p_{neighbor}||^2$.
*   Produces optimal, smooth formation changes.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the difference between Leader-Follower and Virtual Structure?
    *   **A:** Leader-Follower cascades error (if Leader is wrong, everyone is wrong). Virtual Structure distributes error (everyone agrees on a virtual point).
2.  **Q:** Can we have 0 distance?
    *   **A:** No, collision. But we can have distance limits $[d_{min}, d_{max}]$.
3.  **Q:** Why is "Bearing-Only" formation hard?
    *   **A:** Scale ambiguity. If I only know angle to neighbor, I don't know if they are 1m or 1km away. Requires Observability maneuvers.

### Challenge Task
> **Task:** Rotation.
> 1. Rotate the entire formation 90 degrees around its centroid without breaking shape.
> 2. Requires robots on outside to move much faster than inside.

---

## 📚 Further Reading
- **Formation Control:** Desai et al. (IEEE TRA 2001).
- **String Stability:** PATH Project (Platooning).

---

**Day 40 Complete**
