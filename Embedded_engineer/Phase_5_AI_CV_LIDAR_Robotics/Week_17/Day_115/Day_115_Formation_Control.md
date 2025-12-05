# Day 115: Formation Control (Flocking)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 17: Swarm Robotics

---

> **📝 Content Creator Instructions:**
> Don't crash into your friends.
> - **Focus:** Reynolds' Boids (Separation, Alignment, Cohesion), Leader-Follower approaches, and Potential Fields for multi-agent collision avoidance.
> - **Code:** A `flocking_node` that calculates velocity vectors based on neighbor positions and a global goal, creating organic swarm movement.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Implement** the 3 Rules of Boids: Separation (Personal space), Alignment (Copy heading), Cohesion (Stay together).
2.  **Tune** weighting factors ($K_{sep}, K_{align}, K_{coh}$) to change swarm behavior (Aggressive vs Ordered).
3.  **Simulate** Formation Flying in Gazebo (5 Turtlebots).
4.  **Resolve** Deadlocks using random perturbations.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Simulation (Gazebo) recommended to avoid hardware damage.

### Software Environment
```bash
sudo apt install ros-humble-gazebo-ros-pkgs
```

### Prior Knowledge
- Vector Math (Normalization, Addition).
- Day 113 (Neighbor Discovery).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Boids (Bird-oids)

Generic algo developed by Craig Reynolds (1986).
1.  **Separation:** Steer to avoid crowding local flockmates.
    *   $\vec{v}_{sep} = -\sum_{j \in Neighbors} \frac{\vec{p}_j - \vec{p}_i}{||\vec{p}_j - \vec{p}_i||^2}$
2.  **Alignment:** Steer towards the average heading of local flockmates.
    *   $\vec{v}_{align} = \frac{1}{N} \sum \vec{v}_j$
3.  **Cohesion:** Steer to move toward the average position (center of mass) of local flockmates.
    *   $\vec{v}_{coh} = (\frac{1}{N} \sum \vec{p}_j) - \vec{p}_i$

### 🔹 Part 2: Leader-Follower

Boids have no goal. They just float.
To go somewhere, we add a **Goal Vector**:
*   $\vec{v}_{total} = K_s \vec{v}_{sep} + K_a \vec{v}_{align} + K_c \vec{v}_{coh} + K_g \vec{v}_{goal}$
*   The "Leader" (or virtual waypoint) defines $\vec{v}_{goal}$.

### 🔹 Part 3: Graph Theory

The swarm is a dynamic graph.
*   **Rigidity:** If the graph is fully connected (All-to-All), formation is rigid.
*   **Split:** If neighbors are defined by radius $R$, the swarm can split around obstacles and merge back.

---

## 💻 Implementation: Boids Controller

Calculates `cmd_vel` based on neighbor odometry.

### 🛠️ Project Structure
```text
day115_flocking/
├── src/
│   ├── boids_node.py
└── launch/
    └── flock_sim.launch.py
```

### 👨‍💻 Boids Node (`src/boids_node.py`)

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import math

class BoidsNode(Node):
    def __init__(self):
        super().__init__('boids_node')
        
        self.ns = self.get_namespace().strip('/')
        # Assume we know neighbors from Day 113 or Param
        # Hardcoded for 3 robots: robot_0, robot_1, robot_2
        self.neighbors = ['robot_0', 'robot_1', 'robot_2']
        if self.ns in self.neighbors: self.neighbors.remove(self.ns)
        
        # State
        self.my_pose = np.zeros(2)
        self.my_vel = np.zeros(2)
        self.neighbor_states = {} # rid -> {'pos': np, 'vel': np}
        
        # Subs
        self.create_subscription(Odometry, 'odom', self.odom_cb, 10)
        for rid in self.neighbors:
            self.create_subscription(Odometry, f'/{rid}/odom', 
                                     lambda msg, rid=rid: self.neighbor_cb(msg, rid), 10)
        
        self.pub_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_timer(0.1, self.loop)
        
        # Weights
        self.K_sep = 1.5
        self.K_align = 1.0
        self.K_coh = 0.5
        self.K_goal = 0.8
        
        self.goal = np.array([5.0, 5.0]) # Global Goal

    def odom_cb(self, msg):
        self.my_pose[0] = msg.pose.pose.position.x
        self.my_pose[1] = msg.pose.pose.position.y
        # Simplified: Assume Holonomic or Unicycle processing later

    def neighbor_cb(self, msg, rid):
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        # Approx velocity from orientation/twist (simplified)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y # If holonomic
        self.neighbor_states[rid] = {'pos': pos, 'vel': np.array([vx, vy])}

    def loop(self):
        v_sep = np.zeros(2)
        v_align = np.zeros(2)
        v_coh = np.zeros(2)
        center_mass = np.zeros(2)
        count = 0
        
        for rid, state in self.neighbor_states.items():
            diff = state['pos'] - self.my_pose
            dist = np.linalg.norm(diff)
            
            if dist < 0.1: continue # Too close/self
            if dist > 3.0: continue # Out of range
            
            # Separation (Inverse Square Law)
            v_sep -= diff / (dist**2)
            
            # Alignment
            v_align += state['vel']
            
            # Cohesion
            center_mass += state['pos']
            count += 1
            
        if count > 0:
            v_align /= count
            center_mass /= count
            v_coh = center_mass - self.my_pose
            
        # Goal
        v_goal = self.goal - self.my_pose
        if np.linalg.norm(v_goal) > 0:
            v_goal = v_goal / np.linalg.norm(v_goal)
            
        # Total
        v_total = (self.K_sep * v_sep) + (self.K_align * v_align) + \
                  (self.K_coh * v_coh) + (self.K_goal * v_goal)
                  
        # Cap Max Speed
        speed = np.linalg.norm(v_total)
        max_speed = 0.5
        if speed > max_speed:
            v_total = (v_total / speed) * max_speed
            
        # Publish
        cmd = Twist()
        cmd.linear.x = v_total[0] # Note: This assumes Holonomic! 
        # For non-holonomic, we need to convert V_total (vector) to [v, w]
        # w = atan2(Vy, Vx) - current_yaw
        cmd.linear.y = v_total[1]
        self.pub_vel.publish(cmd)

def main():
    rclpy.init()
    rclpy.spin(BoidsNode())
```

---

## 🔬 Lab Exercise: "The Predator"

### 1. Lab Objectives
- **Scenario:** 5 Boids (+Goal), 1 Predator.
- **Predator Logic:** Moves toward Center of Mass of Swarm.
- **Boid Logic:** Add `Predator Avoidance` Term (Strong Separation from Predator).
- **Observe:** The swarm splits around the predator and reforms behind it.
- **Tuning:** If $K_{coh}$ is too high, the swarm gets eaten rather than splitting.

---

## 🚀 Project: "Virtual Structure"

**Goal:** Maintain a Triangle Formation.
1.  **Logic:** Instead of "Average", we have "Slots".
2.  **Leader:** $(x, y)$.
3.  **Follower 1:** Target $(x - 1, y + 1)$.
4.  **Follower 2:** Target $(x - 1, y - 1)$.
5.  **Error Correction:** $PID(target - current)$.
6.  **Rotation:** If Leader turns, rotate the slots using Rotation Matrix.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Oscillation" (Jitter)
*   **Cause:** Separation force is too strong / non-linear singularities near 0 distance.
*   **Fix:** Linearize separation force or cap it. Add damping.

#### 2. "Crash"
*   **Cause:** Latency. Discovery is 1Hz. Robot moves 0.5m in 1s.
*   **Fix:** Predict neighbor position: $P_{measured} + V_{measured} \times \Delta t$.

---

## ⚡ Optimization: Reciprocal Velocity Obstacles (RVO)

Boids is a "Force" model (Physics).
RVO is a "Geometric" model.
*   "If I keep this velocity, will we crash in $\tau$ seconds?"
*   Selects a new velocity from the "Velocity Space" that is collision-free and closest to Goal velocity.
*   Used in video games and advanced robotics.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What happens if $K_{align}$ is 0?
    *   **A:** Swarm looks like a gas cloud. Robots bounce around but stay together (Cohesion) without moving in a unified direction.
2.  **Q:** Holonomic vs Non-Holonomic for Swarms?
    *   **A:** Holonomic (Omni-wheels) is easier (Direct vector mapping). Non-holonomic (Diff Drive) requires Unicycle controllers, which introduces lag in turning.
3.  **Q:** Emergent Behavior?
    *   **A:** The "V" formation of geese reduces drag. It emerges from simple aerodynamic advantages, not a central plan.

### Challenge Task
> **Task:** Through the Door.
> 1. Swarm width: 3m. Door width: 1m.
> 2. Swarm must elongate (Alignment high, Separation low-ish) to pass through.
> 3. Verify it works automatically!

---

## 📚 Further Reading
- **Reynolds:** "Flocks, Herds, and Schools: A Distributed Behavioral Model".
- **ORCA (RVO2):** High performance collision avoidance library.

---

**Day 115 Complete**
