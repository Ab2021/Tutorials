# Day 15: Sampling-Based Planning (RRT*, BIT*)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 3: Advanced Navigation & Planning

---

> **📝 Content Creator Instructions:**
> A* is great for grids, but robots live in Continuous High-Dimensional Spaces (Configuration Space).
> - **Focus:** RRT, RRT*, Bi-directional RRT, and Batch Informed Trees (BIT*).
> - **Code:** Implementation of RRT* from scratch in Python with visualization.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** Configuration Space ($C$-space) and the difference between Workspace and C-space.
2.  **Implement** the Rapidly-exploring Random Tree (RRT) algorithm.
3.  **Upgrade** RRT to RRT* to achieve asymptotic optimality (finding the shortest path over time).
4.  **Analyze** the "Narrow Passage Problem" and how Sampling strategies mitigate it.
5.  **Integrate** RRT* with OMPL (Open Motion Planning Library).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib scipy
# Optional: OMPL Python bindings
```

### Prior Knowledge
- Graph Search (A*, Dijkstra).
- Kinematics (Forward/Inverse).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Configuration Space

Why not just use A*?
*   A* requires a discrete generic graph/grid.
*   A 6-DOF Robot arm has a 6-dimensional continuous space. Discretizing it ($100$ bins per joint) results in $100^6 = 10^{12}$ states. Too big!

**Sampling-Based Planning:**
Instead of discretizing the whole world, we **Sample** random points in $C_{free}$ and connect them.
*   **Completeness:** Probabilistically Complete (If a path exists, we will find it eventually).

### 🔹 Part 2: RRT (Rapidly-exploring Random Tree)

**Algorithm:**
1.  Initialize Tree $T$ with $q_{start}$.
2.  Sample random state $q_{rand}$.
3.  Find nearest neighbor in Tree $q_{near}$.
4.  Steer from $q_{near}$ towards $q_{rand}$ by distance $\delta$ to create $q_{new}$.
5.  If collision-free, add $q_{new}$ to $T$.
6.  Repeat until $q_{goal}$ is reached.

*   **Bias:** The Voronoi bias of RRT naturally causes it to explore large unvisited regions.

### 🔹 Part 3: RRT* (Optimal RRT)

Standard RRT path is "jagged" and sub-optimal. RRT* adds **Rewiring**:
1.  **Choose Parent:** When adding $q_{new}$, check all neighbors in radius $r$. Pick the parent that gives the lowest total Cost to Start.
2.  **Rewire Neighbors:** After adding $q_{new}$, check if neighbors would have a lower cost if they switched their parent to $q_{new}$. If so, rewire them.
*   **Result:** The tree "straightens out" over time, approaching the straight-line optimal path.

### 🔹 Part 4: Advanced Variants

*   **Bi-RRT:** Grow two trees (Start->Goal, Goal->Start) and meet in the middle. Much faster for narrow passages.
*   **BIT* (Batch Informed Trees):** Combines A* (Heuristic Search) with Sampling. Only samples in the "Informed Subset" (Ellipse between Start and Goal).

---

## 💻 Implementation: RRT* Visualization

We will build a 2D RRT* planner.

### 🛠️ Project Structure
```text
day15_planning/
├── src/
│   ├── rrt_star.py
│   └── collision_check.py
└── run_simulator.py
```

### 👨‍💻 Code Implementation (`src/rrt_star.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
import math

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, obstacle_list, rand_area):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.node_list = [self.start]
        self.expand_dis = 1.0 # Step size
        self.goal_sample_rate = 0.1 # Bias towards goal
        self.max_iter = 500
        self.connect_circle_dist = 5.0 # Rewire radius

    def plan(self):
        for i in range(self.max_iter):
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]
            
            new_node = self.steer(nearest_node, rnd, self.expand_dis)
            
            if self.check_collision(new_node, self.obstacle_list):
                near_inds = self.find_near_nodes(new_node)
                
                # 1. Choose Parent
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    # 2. Rewire
                    self.rewire(new_node, near_inds)

            # Check goal
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.goal, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None

    def choose_parent(self, new_node, near_inds):
        if not near_inds: return None
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node, math.inf)
            if t_node and self.check_collision(t_node, self.obstacle_list):
                 costs.append(near_node.cost + self.calc_dist(near_node, new_node))
            else:
                 costs.append(float("inf"))
                 
        min_cost = min(costs)
        if min_cost == float("inf"): return None
        
        min_ind = near_inds[costs.index(min_cost)]
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node, math.inf)
            if not edge_node: continue
            
            no_collision = self.check_collision(edge_node, self.obstacle_list)
            improved_cost = new_node.cost + self.calc_dist(new_node, near_node)
            
            if no_collision andimproved_cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = improved_cost
```

*(Note: Helper functions `steer`, `check_collision`, etc. omitted for brevity but required in full implementation)*

---

## 🔬 Lab Exercise: The Maze Runner

### 1. Lab Objectives
- Define a complex maze with circular Obstacles.
- Run RRT and RRT* side-by-side.
- **Observe:** RRT finds a path quickly but it zig-zags. RRT* takes longer but finds a smooth, near-optimal path.

### 2. Visualization Steps
1.  Plot Obstacles (Black Circles).
2.  Plot Tree Edges (Green Lines).
3.  Plot Final Path (Red Line).
4.  Animate the growth of the tree.

### 3. Analysis Question
- Why does RRT* continue running even after finding the goal?
- *Answer:* To refine the path. It keeps sampling and rewiring to lower the total path cost (asymptotic optimality).

---

## 🚀 Project: "Arm Motion Planning"

**Goal:** Plan a path for a 2-Link Robot Arm to reach a target without hitting an obstacle.
**C-Space:** $\theta_1 \in [0, 2\pi], \theta_2 \in [0, 2\pi]$.
**Transform:** Map Obstacle $(x,y)$ into C-Space Obstacles.

### 1. Implementation
Instead of sampling $(x, y)$, sample $(\theta_1, \theta_2)$.
`check_collision` involves running Forward Kinematics to see if any link intersects the obstacle in workspace.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Taking Forever" (Narrow Passage)
*   **Symptom:** RRT Samples everywhere but fails to go through the door.
*   **Cause:** Probability of sampling exactly in the doorway is small.
*   **Fix:** Use **Gaussian Sampling** (Sample near obstacles) or **Bridge Sampling**.

#### 2. "Collision at Corners"
*   **Symptom:** Robot cuts corner too close and crashes.
*   **Cause:** Robot modeled as a Point. Real robot has radius.
*   **Fix:** Inflate Obstacles by `Robot_Radius + Safety_Margin` (Configuration Space approach).

---

## ⚡ Optimization: KD-Trees

Nearest Neighbor search is $O(N)$ (linear scan).
With 10,000 nodes, this kills performance.
**KD-Tree:** Reduces search to $O(\log N)$.
*   **Task:** Replace linear list search with `scipy.spatial.KDTree`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Is RRT* guaranteed to find the absolute best path in finite time?
    *   **A:** No. It is *asymptotically* optimal. It converges to optimal as Time $\to \infty$.
2.  **Q:** What is the difference between Holonomic and Non-Holonomic planning?
    *   **A:** Holonomic (Omni-drive) can move in any direction. Non-Holonomic (Car) has constraints (cannot move sideways). RRT must use a specific `steer()` function (Dubins Curves) for cars.

### Challenge Task
> **Task:** Implement Bi-Directional RRT.
> 1. Grow Tree A from Start.
> 2. Grow Tree B from Goal.
> 3. In every step, try to connect `new_node_A` to `nearest_node_B`.
> 4. If connected, path found!

---

## 📚 Further Reading
- **RRT:** LaValle (1998).
- **RRT*:** Karaman and Frazzoli (IJRR 2011).
- **OMPL:** kavrakilab.org/OMPL

---

**Day 15 Complete**
