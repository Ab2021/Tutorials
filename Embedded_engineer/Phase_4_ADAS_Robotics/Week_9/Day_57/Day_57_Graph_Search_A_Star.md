# Day 57: Graph Search Algorithms (A*, Dijkstra)
## Phase 4: ADAS & Robotics Systems | Week 9: Path Planning & Navigation

---

> **📝 Day 57 Focus:**
> We know where we are (Localization). We know where the walls are (Mapping). Now, how do we get to the goal? **Path Planning** is the brain of the robot. Today, we start with the classics: **Dijkstra** and **A***, the algorithms that power everything from Google Maps to video game AI.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Represent** a 2D grid map as a Graph (Nodes & Edges).
2.  **Explain** Dijkstra's Algorithm: Expanding the wavefront of cost.
3.  **Implement** A* (A-Star): Adding a Heuristic ($h$) to guide the search.
4.  **Compare** Euclidean vs. Manhattan heuristics and their effect on performance.
5.  **Visualize** the "Open Set" and "Closed Set" to see how the algorithm thinks.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Data Structures:** Priority Queue (Min-Heap).
-   **Math:** Euclidean Distance.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `matplotlib`, `heapq`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Configuration Space

-   **Workspace:** The physical world (x, y).
-   **Configuration Space (C-Space):** The set of all possible robot poses.
-   **Grid Map:** Discretized C-Space.
    -   0 = Free.
    -   1 = Obstacle.

### 🔹 Part 2: Dijkstra's Algorithm

Finds the shortest path from Start to Goal in a weighted graph.
-   **Cost ($g$):** Cost to reach current node from Start.
-   **Strategy:** Always expand the node with the lowest $g$.
-   **Result:** Explores in a circle (uniform wavefront) until it hits the goal. Guaranteed optimal.

### 🔹 Part 3: A* (A-Star) Algorithm

Dijkstra is slow because it explores in all directions. A* is smart.
-   **Heuristic ($h$):** Estimated cost from current node to Goal.
-   **Total Cost ($f$):** $f(n) = g(n) + h(n)$.
-   **Strategy:** Always expand the node with the lowest $f$.
-   **Result:** Beelines towards the goal.
-   **Admissibility:** If $h$ never overestimates the true cost, A* is guaranteed optimal.

---

## 💻 Implementation: A* on Grid Map

**Scenario:**
-   **Map:** 50x50 Grid with random obstacles.
-   **Start:** (10, 10).
-   **Goal:** (40, 40).
-   **Movement:** 8-connected (Up, Down, Left, Right, Diagonals).

### 🛠️ Setup
Create `week9_day57` and `a_star.py`.

```bash
mkdir -p ~/ros2_ws/src/week9_day57
cd ~/ros2_ws/src/week9_day57
touch a_star.py
```

### 👨‍💻 Code: A* Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import heapq
import math

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
        # 1. Setup
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node
        
        # Priority Queue for efficient retrieval of lowest cost
        pq = [] 
        heapq.heappush(pq, (0, self.calc_grid_index(start_node)))

        while True:
            if not open_set:
                print("Open set is empty..")
                break

            # 2. Pop lowest cost node
            cost, c_id = heapq.heappop(pq)
            
            if c_id in open_set:
                current = open_set[c_id]
                del open_set[c_id]
                closed_set[c_id] = current
            else:
                continue

            # 3. Check Goal
            if current.x == goal_node.x and current.y == goal_node.y:
                print("Goal Found!")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # 4. Expand Neighbors
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # Check Bounds & Obstacles
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    # New Node
                    open_set[n_id] = node
                    # f = g + h
                    priority = node.cost + self.calc_heuristic(node, goal_node)
                    heapq.heappush(pq, (priority, n_id))
                else:
                    # Existing Node: Check if this path is better
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node
                        priority = node.cost + self.calc_heuristic(node, goal_node)
                        heapq.heappush(pq, (priority, n_id))

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
        w = 1.0 # Weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_pos):
        pos = index * self.resolution + min_pos
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x: return False
        if py < self.min_y: return False
        if px >= self.max_x: return False
        if py >= self.max_y: return False

        # Collision Check
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
                    d = math.hypot(iox - x, ioy - y)
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
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]
        return motion

def main():
    # Start/Goal
    sx = 10.0
    sy = 10.0
    gx = 50.0
    gy = 50.0
    grid_size = 2.0
    robot_radius = 1.0

    # Obstacles
    ox, oy = [], []
    for i in range(60): # Borders
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
    for i in range(40): # Wall
        ox.append(20.0)
        oy.append(i)
    for i in range(40): # Wall
        ox.append(40.0)
        oy.append(60.0 - i)

    if True:  # plotting
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    if True:  # plotting
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()

if __name__ == '__main__':
    main()
```

---

## 🔬 Lab Exercise: Heuristic Tuning

### Lab Objectives
1.  Run the code.
2.  **Observation:** The red line finds the shortest path around the walls.
3.  **Experiment A:** Set `w = 0.0` in `calc_heuristic`.
    -   *Result:* This becomes Dijkstra's Algorithm. It will explore *everywhere* (slow) but find the optimal path.
4.  **Experiment B:** Set `w = 5.0` (Greedy Best First Search).
    -   *Result:* It runs super fast, beelining for the goal, but the path might be sub-optimal (hit a wall and trace around it inefficiently).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Path Through Walls
**Symptom:** Robot cuts corners or goes through obstacles.
**Cause:** `robot_radius` is too small or collision check is buggy.
**Solution:** Inflate obstacles by the robot radius (Configuration Space).

#### 2. Infinite Loop
**Symptom:** Code hangs.
**Cause:** Goal is unreachable (enclosed by walls).
**Solution:** Add a timeout or check if `open_set` is empty (which the code does).

---

## ⚡ Optimization & Best Practices

### 1. Hybrid A*
Grid A* produces jagged paths (Manhattan-like). Real cars can't turn 90 degrees instantly.
**Hybrid A*:**
-   Nodes are continuous $(x, y, \theta)$.
-   Motion model uses Reeds-Shepp or Dubins curves (Kinematics).
-   Result: Smooth, drivable paths.

### 2. JPS (Jump Point Search)
Optimization for uniform grids.
-   Instead of stepping 1 cell at a time, "jump" until you hit an obstacle or a turning point.
-   Orders of magnitude faster than A*.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What makes a heuristic "Admissible"?
    *   **A:** It never overestimates the cost to the goal. Euclidean distance is admissible (straight line is shortest). Manhattan distance is admissible *only* if you can't move diagonally.
2.  **Q:** Why use a Priority Queue?
    *   **A:** To efficiently retrieve the node with the lowest $f$ cost ($O(1)$ or $O(\log N)$). Sorting a list every time is $O(N \log N)$.
3.  **Q:** What is the difference between Open Set and Closed Set?
    *   **A:** Open Set = Candidates to explore. Closed Set = Already explored (don't visit again).

### Challenge Task
**Task:** 4-Connected vs 8-Connected.
1.  Modify `get_motion_model` to remove diagonals.
2.  Run A*.
3.  Observe the "Staircase" effect.
4.  Compare path length vs 8-connected.

---

## 📚 Further Reading & References
-   [Introduction to A* (Red Blob Games)](https://www.redblobgames.com/pathfinding/a-star/introduction.html) - Best visual guide.
-   [PythonRobotics A*](https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning/AStar)

---

**Day 57 Complete** | Phase 4: ADAS & Robotics Systems | Week 9: Path Planning & Navigation
