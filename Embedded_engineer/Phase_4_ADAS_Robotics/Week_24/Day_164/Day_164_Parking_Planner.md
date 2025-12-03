# Day 164: Path Planning in Parking Lots
## Phase 4: ADAS & Robotics Systems | Week 24: Capstone Project - Autonomous Valet Parking

---

> **📝 Day 164 Focus:**
> Highway planning is easy (Go straight). Parking is hard. You need to reverse, turn tight corners, and maybe do a 3-point turn. Standard A* fails here because it ignores the car's turning radius. We need **Hybrid A***.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the difference between Holonomic and Non-Holonomic planning.
2.  **Implement** Hybrid A* (A-Star with Reeds-Shepp curves).
3.  **Generate** a path for Reverse Parking.
4.  **Smooth** the path using Gradient Descent (Voronoi Field).
5.  **Handle** "Stuck" situations (Re-planning).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 134:** A* Algorithm.
-   **Kinematics:** Ackermann Steering, Minimum Turning Radius.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `numpy`, `heapq`.
-   **Libraries:** `ompl` (Optional), or custom implementation.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Non-Holonomic Constraint

-   **Holonomic:** Can move in any direction (Omni-wheel robot).
-   **Non-Holonomic:** Can only move forward/backward along the heading. $\dot{y} \cos\theta - \dot{x} \sin\theta = 0$.
-   **Result:** You can't slide sideways into a spot. You must drive a curve.

### 🔹 Part 2: Hybrid A*

Standard A* searches grid cells $(x, y)$.
Hybrid A* searches continuous states $(x, y, \theta)$.
-   **Node Expansion:** Instead of "North/South", we apply steering inputs: "Max Left", "Straight", "Max Right".
-   **Heuristic:**
    1.  **Non-Holonomic:** Length of Reeds-Shepp path (ignoring obstacles).
    2.  **Holonomic:** 2D A* distance (ignoring kinematics, respecting obstacles).
    -   $H = \max(H_{RS}, H_{2D})$.

### 🔹 Part 3: Reeds-Shepp Curves

-   Optimal paths for a car that can reverse.
-   Combinations of Straight (S), Left (L), Right (R).
-   Examples: `L S L`, `L R L` (3-point turn).
-   Used as the "Analytic Expansion" step in Hybrid A* (Try to connect to goal directly if close).

---

## 💻 Implementation: Simple Parking Planner

**Scenario:**
-   Start: $(0, 0, 0^\circ)$.
-   Goal: $(10, 10, 90^\circ)$ (Parallel Park).
-   Obstacles: Wall at $x=5$.

### 🛠️ Setup
Create `week24_capstone/planning`.

### 👨‍💻 Code: Hybrid A* (Simplified)

```python
import numpy as np
import math
import heapq
import matplotlib.pyplot as plt

# Vehicle Constants
WB = 3.0  # Wheelbase
MAX_STEER = np.deg2rad(30.0)
MIN_RADIUS = WB / math.tan(MAX_STEER)

class Node:
    def __init__(self, x, y, theta, cost, parent, steer):
        self.x = x
        self.y = y
        self.theta = theta
        self.cost = cost
        self.parent = parent
        self.steer = steer
        
    def __lt__(self, other):
        return self.cost < other.cost

def motion_model(x, y, theta, steer, dist=1.0):
    # Bicycle Model
    x += dist * math.cos(theta)
    y += dist * math.sin(theta)
    theta += dist * math.tan(steer) / WB
    return x, y, theta

def hybrid_a_star(start, goal, obstacles):
    # start/goal: [x, y, theta]
    
    open_set = []
    heapq.heappush(open_set, Node(start[0], start[1], start[2], 0.0, None, 0.0))
    
    visited = {} # (idx_x, idx_y, idx_theta) -> cost
    
    while open_set:
        current = heapq.heappop(open_set)
        
        # Check Goal (Approx)
        dist = math.hypot(current.x - goal[0], current.y - goal[1])
        angle_diff = abs(current.theta - goal[2])
        if dist < 1.0 and angle_diff < 0.5:
            print("Goal Reached!")
            return reconstruct_path(current)
            
        # Expand
        # Try different steering angles: Left, Straight, Right
        # And Directions: Forward, Reverse
        steer_inputs = [-MAX_STEER, 0, MAX_STEER]
        directions = [1, -1]
        
        for d in directions:
            for delta in steer_inputs:
                nx, ny, nth = motion_model(current.x, current.y, current.theta, delta, dist=1.0 * d)
                
                # Check Obstacles (Simple Circle Check)
                if not check_collision(nx, ny, obstacles):
                    new_cost = current.cost + 1.0 + abs(delta)*0.1 # Penalize steering
                    if d < 0: new_cost += 1.0 # Penalize reversing
                    if d != np.sign(current.steer) and current.parent: new_cost += 2.0 # Penalize gear switch
                    
                    # Discretize for Visited set
                    idx = (int(nx), int(ny), int(math.degrees(nth)/10))
                    
                    if idx not in visited or new_cost < visited[idx]:
                        visited[idx] = new_cost
                        # Heuristic: Euclidean dist
                        h = math.hypot(nx - goal[0], ny - goal[1])
                        heapq.heappush(open_set, Node(nx, ny, nth, new_cost + h, current, delta))
                        
    print("No Path Found")
    return None

def check_collision(x, y, obstacles):
    for ox, oy, r in obstacles:
        if math.hypot(x - ox, y - oy) < r + 1.0: # 1.0m Safety margin
            return True
    return False

def reconstruct_path(node):
    path_x, path_y = [], []
    while node:
        path_x.append(node.x)
        path_y.append(node.y)
        node = node.parent
    return path_x[::-1], path_y[::-1]

def main():
    # Obstacles (x, y, radius)
    obstacles = [
        (5.0, 5.0, 1.0),
        (5.0, 2.0, 1.0)
    ]
    
    start = [0, 0, 0]
    goal = [10, 10, np.deg2rad(90)]
    
    path_x, path_y = hybrid_a_star(start, goal, obstacles)
    
    if path_x:
        plt.plot(path_x, path_y, '-r', label='Path')
        # Draw Obstacles
        for ox, oy, r in obstacles:
            circle = plt.Circle((ox, oy), r, color='k')
            plt.gca().add_patch(circle)
            
        plt.plot(start[0], start[1], 'go', label='Start')
        plt.plot(goal[0], goal[1], 'bo', label='Goal')
        plt.legend()
        plt.axis('equal')
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Reverse Park

### Lab Objectives
1.  **Run the script.**
    -   **Observation:** The path curves around the obstacles.
2.  **Force Reverse:**
    -   Place a wall of obstacles in front of the goal, leaving only the back open.
    -   **Result:** The planner should generate a "Cusp" (Gear change) to back into the spot.
    -   *Note: The simplified code above might struggle with complex maneuvers. Full Hybrid A* requires Reeds-Shepp heuristics.*

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Path is Jerky
**Symptom:** Zig-zag steering.
**Cause:** Grid discretization is too coarse.
**Solution:** **Path Smoothing**. Use Gradient Descent to move points away from obstacles and minimize curvature ($\kappa$).

#### 2. Planner Stuck
**Symptom:** "No Path Found" in tight spots.
**Cause:** Node expansion step (arc length) is too large.
**Solution:** Reduce step size (e.g., 0.5m). Use Analytic Expansion (Reeds-Shepp) more often.

---

## ⚡ Optimization & Best Practices

### 1. Pre-computed Primitives
Instead of calculating motion model online:
-   Generate a lookup table of valid short trajectories (Primitives) offline.
-   Load them at runtime for speed.

### 2. Multi-Resolution Search
-   Run standard A* (2D) first to find the rough corridor.
-   Run Hybrid A* only within that corridor.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is Reeds-Shepp better than Dubins?
    *   **A:** Dubins curves only allow forward motion. Reeds-Shepp allows reverse, which is essential for parking.
2.  **Q:** What is a "Cusp"?
    *   **A:** A point in the path where velocity changes sign (Forward $\leftrightarrow$ Reverse) and steering can change instantaneously (since speed is 0).
3.  **Q:** How do we handle collision checking for a car shape?
    *   **A:** Represent the car as 3 overlapping circles. Check distance from obstacle to each circle center. Faster than polygon intersection.

### Challenge Task
**Task:** Parallel Parking.
1.  Set up obstacles to form a "Parallel Parking" slot (Cars front and back).
2.  Tune the cost function to prefer reversing into the spot.
3.  Visualize the 3-point turn.

---

## 📚 Further Reading & References
-   [Practical Search Techniques in Path Planning (Thrun)](https://ai.stanford.edu/~ddolgov/papers/dolgov_gpp_stair08.pdf)
-   [Reeds-Shepp Curves](https://pypi.org/project/reeds-shepp/)

---

**Day 164 Complete** | Phase 4: ADAS & Robotics Systems | Week 24: Capstone Project - Autonomous Valet Parking
