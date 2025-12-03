# Day 134: Global Planning (A*, Dijkstra)
## Phase 4: ADAS & Robotics Systems | Week 20: Path Planning & Decision Making

---

> **📝 Day 134 Focus:**
> We know where we are (Localization) and what is around us (Perception). Now, **Where do we go?** Global Planning finds a route from A to B. We start with the classic graph search algorithms: **Dijkstra** and **A***.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Compare** Breadth-First Search (BFS), Dijkstra, and A*.
2.  **Define** the Heuristic Function ($h(n)$) and its admissibility.
3.  **Implement** the A* algorithm on a 2D Grid Map.
4.  **Visualize** the Open Set and Closed Set expansion.
5.  **Analyze** the effect of heuristic weight on path quality vs. speed.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Data Structures:** Priority Queue (Min-Heap).
-   **Graphs:** Nodes, Edges, Weights.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `heapq`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Search Problem

Given a Grid Map (0=Free, 1=Obstacle).
Start: $(x_s, y_s)$. Goal: $(x_g, y_g)$.
Find a sequence of cells that minimizes cost (Distance).

### 🔹 Part 2: Dijkstra's Algorithm

-   Explores nodes based on **Cost-So-Far** ($g(n)$).
-   Guarantees shortest path.
-   Expands in a circle (inefficient).

### 🔹 Part 3: A* (A-Star) Algorithm

-   Explores nodes based on **Total Estimated Cost** ($f(n) = g(n) + h(n)$).
-   **$h(n)$ (Heuristic):** Estimated cost from $n$ to Goal.
    -   Euclidean Distance: $\sqrt{\Delta x^2 + \Delta y^2}$.
    -   Manhattan Distance: $|\Delta x| + |\Delta y|$.
-   **Admissibility:** $h(n)$ must never overestimate the true cost. If it does, A* is not optimal.
-   Expands towards the goal (efficient).

---

## 💻 Implementation: A* Planner

**Scenario:**
-   $50 \times 50$ Grid.
-   Random Obstacles.
-   Find path from Top-Left to Bottom-Right.

### 🛠️ Setup
Create `week20_day134` and `astar_planner.py`.

```bash
mkdir -p ~/ros2_ws/src/week20_day134
cd ~/ros2_ws/src/week20_day134
touch astar_planner.py
```

### 👨‍💻 Code: A* Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import heapq

class Node:
    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost # g(n)
        self.parent_index = parent_index

class AStarPlanner:
    def __init__(self, ox, oy, resolution, rr):
        self.resolution = resolution
        self.rr = rr # Robot Radius
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node
        
        # Priority Queue: (f_score, grid_index)
        pq = []
        heapq.heappush(pq, (self.calc_heuristic(start_node, goal_node), self.calc_grid_index(start_node)))

        while True:
            if not open_set:
                print("Error: Cannot verify grid map.")
                return [], []

            c_id = heapq.heappop(pq)[1]
            current = open_set[c_id]

            # Show animation
            # plt.plot(self.calc_grid_position(current.x, self.min_x),
            #          self.calc_grid_position(current.y, self.min_y), "xc")
            # plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Goal Found!")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove from open set, add to closed set
            del open_set[c_id]
            closed_set[c_id] = current

            # Expand neighbors
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If goal found (approx)
                if node.x == goal_node.x and node.y == goal_node.y:
                    goal_node.parent_index = current.parent_index
                    goal_node.cost = current.cost
                    # Important: Don't break here, let priority queue handle optimality
                
                if n_id in closed_set:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                    heapq.heappush(pq, (node.cost + self.calc_heuristic(node, goal_node), n_id))
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node
                        # Update priority (lazy update: push duplicate, handle later or ignore)
                        heapq.heappush(pq, (node.cost + self.calc_heuristic(node, goal_node), n_id))

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        return rx, ry

    def calc_heuristic(self, n1, n2):
        w = 1.0 # Weight
        d = w * np.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_pos):
        return index * self.resolution + min_pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)
        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
            return False
        if self.obstacle_map[node.x][node.y]:
            return False
        return True

    def calc_obstacle_map(self, ox, oy):
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = np.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, np.sqrt(2)],
                  [-1, 1, np.sqrt(2)],
                  [1, -1, np.sqrt(2)],
                  [1, 1, np.sqrt(2)]]
        return motion

def main():
    # Start/Goal
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # Obstacles
    ox, oy = [], []
    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")
    plt.grid(True)
    plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    plt.plot(rx, ry, "-r")
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Heuristic Effect

### Lab Objectives
1.  Run the script.
2.  **Observation:** The Red path snakes around the walls efficiently.
3.  **Experiment:**
    -   Set `w = 0.0` in `calc_heuristic`.
    -   **Result:** This becomes **Dijkstra**. It will explore *every* direction equally (Circle). It will be much slower (more nodes visited) but still optimal.
    -   Set `w = 5.0`.
    -   **Result:** This becomes **Greedy Best-First Search**. It runs super fast towards the goal but might hit a dead end or find a sub-optimal path.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Path Through Walls
**Symptom:** The path cuts corners or goes through obstacles.
**Cause:** `robot_radius` is too small or collision check is buggy.
**Solution:** Inflate obstacles (Configuration Space). The path planner assumes a point robot, so obstacles must be expanded by the robot radius.

#### 2. Infinite Loop
**Symptom:** Planner never finishes.
**Cause:** Goal is unreachable (enclosed by walls).
**Solution:** Add a timeout or max iteration count.

---

## ⚡ Optimization & Best Practices

### 1. Hybrid A*
Standard A* produces jagged paths (grid constrained). Cars can't turn 90 degrees instantly.
-   **Hybrid A*:** Uses continuous coordinates $(x, y, \theta)$ inside grid cells.
-   Uses Reeds-Shepp or Dubins curves as the motion model.
-   Produces drivable, smooth paths.

### 2. JPS (Jump Point Search)
Optimization for uniform grids.
-   Instead of stepping 1 by 1, "jump" until you hit an obstacle or a turning point.
-   Orders of magnitude faster than A*.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What makes A* "optimal"?
    *   **A:** It is optimal if the heuristic is **admissible** (never overestimates) and **consistent**.
2.  **Q:** What is the difference between Global and Local Planning?
    *   **A:** Global: Finds route on a map (A*). Local: Avoids dynamic obstacles and follows the route (MPC/DWA).
3.  **Q:** Why use a Priority Queue?
    *   **A:** To always expand the most promising node (lowest $f$-score) first. $O(1)$ access to min.

### Challenge Task
**Task:** 8-Connectivity.
1.  The current code uses 8-connectivity (diagonal moves allowed).
2.  Change `get_motion_model` to only allow 4-connectivity (Up, Down, Left, Right).
3.  Observe the "Manhattan" style path.

---

## 📚 Further Reading & References
-   [Introduction to A*](https://www.redblobgames.com/pathfinding/a-star/introduction.html)
-   [Hybrid A* Paper (Stanford Junior)](https://ai.stanford.edu/~ddolgov/papers/dolgov_gpp_stair08.pdf)

---

**Day 134 Complete** | Phase 4: ADAS & Robotics Systems | Week 20: Path Planning & Decision Making
