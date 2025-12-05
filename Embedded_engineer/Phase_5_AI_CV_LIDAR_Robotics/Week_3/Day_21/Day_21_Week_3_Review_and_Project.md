# Day 21: Week 3 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 3: Advanced Navigation & Planning

---

> **📝 Content Creator Instructions:**
> Planning is the "Brain" of the robot. We moved from Random Trees (RRT) to Social Force Fields.
> - **Goal:** Integrate Hybrid A* (Global), MPPI (Local), and Dynamic Obstacle Management.
> - **Code:** Autonomous Valet Parking (AVP) System simulation.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Synthesize** Global Planning (Search) and Local Planning (Optimization) into a 2-Tier Architecture.
2.  **Implement** Hybrid A* logic for Reeds-Shepp (Car-like) paths.
3.  **Coordinate** multiple sub-systems: Perception (Obstacles) -> Planning (Trajectory) -> Control (Steering).
4.  **Execute** a complex "Parking" maneuver with limited space.

---

## 📚 Week 3 Review: The Navigation Stack

| Day | Component | Key Concept | Pros | Cons |
|-----|-----------|-------------|------|------|
| **15** | **RRT*** | Sampling $C_{space}$ | Finds path in high dims | Jagged paths |
| **16** | **TEB/MPPI** | Optimization | Smooth, Dynamic | Local minima |
| **17** | **RL** | Learning | Handles unknown rules | Hard to verify safety |
| **18** | **NBV** | Exploration | Maps unknown areas | Greedy |
| **19** | **MAPF** | Swarm | Cooperative | Scalability ($N!$) |
| **20** | **Social** | Interaction | Human-friendly | Tuning heuristic params |

### The "Standard" Stack
1.  **Global Planner:** RRT* or Hybrid A* (Low Freq, Long Horizon). Generates Waypoints.
2.  **Local Planner:** TEB or MPPI (High Freq, Short Horizon). Follows Waypoints + Avoids Obstacles.

---

## 🚀 Weekly Capstone: "Autonomous Valet Parking (AVP)"

**Scenario:** A car is dropped off at the entrance. It must navigate a parking lot, avoid pedestrians (Day 20), avoid other cars (Day 19), and reverse park into a spot (Day 15/16).

### 🛠️ Project Structure
```text
week3_capstone/
├── map/
│   └── parking_lot.yaml
├── src/
│   ├── global_planner.py (Hybrid A*)
│   ├── local_planner.py  (MPPI)
│   ├── behavior_tree.py  (Decision Making)
│   └── park_maneuver.py
└── launch/
    └── avp_sim.launch.py
```

### 👨‍💻 Code Implementation: Hybrid A* Node

Hybrid A* combines discrete grid search with continuous Reeds-Shepp curves (Forward/Reverse/Turn).

```python
import numpy as np
import heapq

class Node:
    def __init__(self, x, y, yaw, cost, parent, direction):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.cost = cost
        self.parent = parent
        self.direction = direction # 1=fwd, -1=rev

class HybridAStar:
    def __init__(self, costmap):
        self.costmap = costmap
        self.xy_res = 0.5
        self.yaw_res = np.deg2rad(15) 
        
    def search(self, start, goal):
        # start: [x, y, yaw], goal: [x, y, yaw]
        
        open_list = []
        start_node = Node(start[0], start[1], start[2], 0, None, 1)
        
        heapq.heappush(open_list, (0, start_node))
        closed_set = {} # Key: (idx_x, idx_y, idx_yaw)
        
        while open_list:
            cost, current = heapq.heappop(open_list)
            
            if self.calc_dist(current, goal) < 1.0 and abs(current.yaw - goal[2]) < 0.2:
                print("Goal Reached!")
                return self.reconstruct_path(current)
                
            # Expand (Reeds-Shepp Primitives)
            # Left, Straight, Right (Fwd & Rev)
            for steering in [-0.5, 0, 0.5]:
                for direction in [1, -1]:
                    next_node = self.motion_model(current, steering, direction)
                    
                    if not self.check_collision(next_node):
                        idx = self.calc_index(next_node)
                        if idx not in closed_set:
                            # Heuristic: Non-holonomic-without-obstacles (Reeds-Shepp dist)
                            # + Holonomic-with-obstacles (A* 2D)
                            h = self.calc_heuristic(next_node, goal)
                            heapq.heappush(open_list, (next_node.cost + h, next_node))
                            closed_set[idx] = next_node
                            
        return None

    def motion_model(self, node, steering, direction):
        L = 2.5 # Wheelbase
        ds = 0.5 # Step size
        
        x = node.x + direction * ds * np.cos(node.yaw)
        y = node.y + direction * ds * np.sin(node.yaw)
        yaw = node.yaw + direction * (ds / L) * np.tan(steering)
        
        # Determine cost (Penalty for gear switch and reversing)
        penalty = 0
        if direction != node.direction: penalty += 50 # Gear switch cost
        if direction == -1: penalty += 10 # Reversing is harder
        
        return Node(x, y, normalize_angle(yaw), node.cost + ds + penalty, node, direction)
```

### 👨‍💻 Logic: Parking Maneuver

Parking requires high precision. Pure tracking might fail.
**Logic:**
1.  **Approach:** Drive to a point 5m in front of the spot, aligned perpendicular (for parallel) or angled (for perpendicular).
2.  **Stop:** Velocity = 0.
3.  **Switch Gear:** Reverse.
4.  **Back In:** Use Pure Pursuit on the generated reverse curve.
5.  **Fine Tune:** If `dist_to_wall < 0.2`, stop. Pull forward if angle is bad (Wiggle).

---

## 📝 Self-Assessment Quiz

1.  **Architecture:**
    *   Why use Hybrid A* instead of standard A* for a car?
        *   **A:** Standard A* (Grid) yields paths like (0,0) -> (0,1) -> (1,1) which requires the car to slide sideways. Hybrid A* respects the kinematic constraints (Turning Radius) and generates drivable curves.

2.  **Optimization:**
    *   When using MPPI, how do we handle a "moving pedestrian"?
        *   **A:** By predicting their future position (Constant Velocity Model) and marking those cells as high cost in the future timesteps of the rollout.

3.  **Safety:**
    *   What is the "Time to Collision" (TTC) metric?
        *   **A:** $TTC = \frac{\text{Distance}}{\text{RelativeVelocity}}$. If $TTC < 2s$, the robot must trigger Emergency Braking (AEB).

---

## ⏭️ Look Ahead: Week 4
We have Maps (Week 2) and Plans (Week 3). Now we need to **Control** the hardware.
**Week 4: Advanced Control Theory.**
*   LQR / MPC (Formal Control).
*   Lyapunov Stability.
*   Reinforcement Learning for Control (Sim2Real).

---

**Week 3 Complete**
