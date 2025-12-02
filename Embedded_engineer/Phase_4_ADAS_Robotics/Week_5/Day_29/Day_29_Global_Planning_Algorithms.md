# Day 29: Global Planning Algorithms (A*, RRT)
## Phase 4: ADAS & Robotics Systems | Week 5: Path Planning & Decision Making

---

> **📝 Day 29 Focus:**
> We know where we are (Localization) and what the world looks like (Mapping). Now, we need to find a path from Start to Goal. This is **Global Planning**. Today, we master the classics: Graph Search algorithms (A*) and Sampling-based algorithms (RRT).

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Differentiate** between Workspace (Physical) and Configuration Space (C-Space).
2.  **Implement** Dijkstra and A* algorithms on a Grid Map.
3.  **Design** an admissible Heuristic for A* (Euclidean vs Manhattan).
4.  **Implement** RRT (Rapidly-exploring Random Tree) for continuous spaces.
5.  **Analyze** the trade-offs: Completeness vs Optimality vs Speed.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Data Structures:** Graphs, Priority Queues (Heaps), Trees.
-   **Geometry:** Euclidean distance, Collision checking.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows/Mac with Python).

### Software Stack
-   **Python Libraries:** `numpy`, `matplotlib`, `heapq`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Planning Problem

Given:
-   **Start State:** $q_{start}$
-   **Goal State:** $q_{goal}$
-   **Map:** Obstacles $O$.

Find a continuous path $\tau : [0, 1] \to C_{free}$ such that $\tau(0) = q_{start}$ and $\tau(1) = q_{goal}$.

#### 1.1 Configuration Space (C-Space)
The robot is not a point; it has size and shape.
-   **Workspace:** The physical 2D/3D world.
-   **C-Space:** The set of all possible robot configurations (positions/angles).
-   **C-Obstacle:** Configurations where the robot collides with an obstacle.
-   **Idea:** Inflate obstacles by the robot radius. Then treat the robot as a point in C-Space.

---

### 🔹 Part 2: Graph Search (Grid-Based)

Discretize C-Space into a grid.

#### 2.1 Dijkstra's Algorithm
-   Explores uniformly in all directions (like a wavefront).
-   Guaranteed to find the shortest path.
-   **Cost:** $g(n)$ = cost from start to node $n$.
-   **Priority:** Minimize $g(n)$.

#### 2.2 A* (A-Star)
-   Uses a **Heuristic** to guide the search towards the goal.
-   **Cost:** $f(n) = g(n) + h(n)$.
    -   $g(n)$: Cost so far.
    -   $h(n)$: Estimated cost to goal.
-   **Heuristic Admissibility:** $h(n)$ must never overestimate the true cost.
    -   Euclidean Distance ($\sqrt{\Delta x^2 + \Delta y^2}$) is admissible.
    -   Manhattan Distance ($|\Delta x| + |\Delta y|$) is admissible only for 4-connected grids.

---

### 🔹 Part 3: Sampling-Based Planning

For high-dimensional spaces (e.g., a 7-DOF arm), grids are too expensive ($N^7$).
We use random sampling.

#### 3.1 RRT (Rapidly-exploring Random Tree)
1.  Initialize Tree with $q_{start}$.
2.  Sample a random point $q_{rand}$ in space.
3.  Find nearest node in tree $q_{near}$.
4.  Steer from $q_{near}$ towards $q_{rand}$ by step size $\delta$ to create $q_{new}$.
5.  If path is collision-free, add $q_{new}$ to tree.
6.  Repeat until $q_{new}$ is close to $q_{goal}$.

**Pros:** Probabilistically Complete (will find a path if one exists). Fast in high dimensions.
**Cons:** Path is jagged and non-optimal.

#### 3.2 RRT* (Optimal RRT)
Adds a "Rewiring" step.
-   When adding $q_{new}$, check if it can connect to neighbors with a lower cost.
-   If yes, rewire the tree.
-   **Result:** Converges to the optimal path over time.

---

## 💻 Implementation: A* and RRT

We will implement two scripts: `astar_grid.py` and `rrt_continuous.py`.

### 🛠️ Setup
Create `week5_day29`.

```bash
mkdir -p ~/ros2_ws/src/week5_day29
cd ~/ros2_ws/src/week5_day29
touch astar_grid.py rrt_continuous.py
```

### 👨‍💻 Code: A* on Grid

```python
import numpy as np
import matplotlib.pyplot as plt
import heapq

class AStar:
    def __init__(self, grid, start, goal):
        self.grid = grid # 0=Free, 1=Obstacle
        self.start = start
        self.goal = goal
        self.rows, self.cols = grid.shape
        
        # Priority Queue: (f_score, x, y)
        self.open_set = []
        heapq.heappush(self.open_set, (0, start[0], start[1]))
        
        self.came_from = {}
        
        self.g_score = {start: 0}
        self.f_score = {start: self.heuristic(start, goal)}
        
        # 8-connected neighbors
        self.motions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def heuristic(self, a, b):
        # Euclidean distance
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def reconstruct_path(self, current):
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        return path[::-1]

    def plan(self):
        while self.open_set:
            _, cx, cy = heapq.heappop(self.open_set)
            current = (cx, cy)
            
            if current == self.goal:
                return self.reconstruct_path(current)
            
            for dx, dy in self.motions:
                neighbor = (cx + dx, cy + dy)
                
                # Check bounds
                if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                    # Check obstacle
                    if self.grid[neighbor[0], neighbor[1]] == 1:
                        continue
                        
                    # Cost: 1 for straight, sqrt(2) for diagonal
                    move_cost = np.sqrt(dx**2 + dy**2)
                    tentative_g = self.g_score[current] + move_cost
                    
                    if neighbor not in self.g_score or tentative_g < self.g_score[neighbor]:
                        self.came_from[neighbor] = current
                        self.g_score[neighbor] = tentative_g
                        f = tentative_g + self.heuristic(neighbor, self.goal)
                        self.f_score[neighbor] = f
                        heapq.heappush(self.open_set, (f, neighbor[0], neighbor[1]))
                        
        return None # No path found

def run_astar():
    # Create Grid Map (50x50)
    grid = np.zeros((50, 50))
    
    # Add Obstacles
    grid[10:40, 20:25] = 1 # Vertical wall
    grid[10:20, 30:40] = 1 # Block
    
    start = (5, 5)
    goal = (45, 45)
    
    planner = AStar(grid, start, goal)
    path = planner.plan()
    
    # Visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.T, cmap='Greys', origin='lower')
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'rx', markersize=10, label='Goal')
    
    if path:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='A* Path')
    else:
        print("No path found!")
        
    plt.legend()
    plt.title("A* Grid Planning")
    plt.show()

if __name__ == "__main__":
    run_astar()
```

### 👨‍💻 Code: RRT in Continuous Space

```python
import numpy as np
import matplotlib.pyplot as plt
import math

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=3.0, goal_sample_rate=5):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.obstacle_list = obstacle_list # [(x, y, radius), ...]
        self.node_list = []

    def plan(self):
        self.node_list = [self.start]
        
        for i in range(500): # Max iterations
            # 1. Sample Random Node
            if np.random.randint(0, 100) > self.goal_sample_rate:
                rnd = Node(np.random.uniform(self.min_rand, self.max_rand),
                           np.random.uniform(self.min_rand, self.max_rand))
            else:
                rnd = Node(self.goal.x, self.goal.y) # Bias towards goal
                
            # 2. Find Nearest Node
            dists = [(node.x - rnd.x)**2 + (node.y - rnd.y)**2 for node in self.node_list]
            nearest_ind = dists.index(min(dists))
            nearest_node = self.node_list[nearest_ind]
            
            # 3. Steer
            theta = math.atan2(rnd.y - nearest_node.y, rnd.x - nearest_node.x)
            new_node = Node(nearest_node.x + self.expand_dis * math.cos(theta),
                            nearest_node.y + self.expand_dis * math.sin(theta))
            new_node.parent = nearest_node
            
            # 4. Check Collision
            if not self.check_collision(new_node, self.obstacle_list):
                continue
                
            self.node_list.append(new_node)
            
            # 5. Check Goal
            dx = new_node.x - self.goal.x
            dy = new_node.y - self.goal.y
            d = math.hypot(dx, dy)
            
            if d <= self.expand_dis:
                print("Goal Found!")
                self.goal.parent = new_node
                return self.generate_course()
                
        return None

    def check_collision(self, node, obstacle_list):
        for (ox, oy, size) in obstacle_list:
            dx = ox - node.x
            dy = oy - node.y
            d = dx * dx + dy * dy
            if d <= size**2:
                return False # Collision
        return True # Safe

    def generate_course(self):
        path = [[self.goal.x, self.goal.y]]
        node = self.goal
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y])
        return path

def run_rrt():
    # Obstacles (x, y, radius)
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1)
    ]
    
    rrt = RRT(start=[0, 0], goal=[10, 10],
              rand_area=[-2, 15], obstacle_list=obstacle_list)
    path = rrt.plan()
    
    # Visualization
    plt.figure(figsize=(8, 8))
    
    # Draw Obstacles
    for (ox, oy, size) in obstacle_list:
        circle = plt.Circle((ox, oy), size, color='k')
        plt.gca().add_patch(circle)
        
    # Draw Tree
    for node in rrt.node_list:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")
            
    # Draw Path
    if path:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], '-r', linewidth=3, label='RRT Path')
        
    plt.plot(0, 0, "xr")
    plt.plot(10, 10, "xr")
    plt.axis("equal")
    plt.grid(True)
    plt.title("RRT Path Planning")
    plt.show()

if __name__ == "__main__":
    run_rrt()
```

---

## 🔬 Lab Exercise: Heuristic Impact

### Lab Objectives
1.  Run `astar_grid.py`.
2.  **Experiment:** Change the heuristic to return `0` always.
    -   *Result:* A* becomes Dijkstra. It explores a perfect circle.
3.  **Experiment:** Multiply heuristic by 5.0 ($h(n) \times 5$).
    -   *Result:* Greedy Best-First Search. Very fast, beelines to goal, but might hit a wall and get stuck or find a non-optimal path.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Infinite Loop
**Symptom:** Code hangs.
**Cause:** Goal is unreachable (surrounded by obstacles), and there is no termination condition for "Open Set Empty".
**Solution:** Ensure `while open_set:` loop terminates if queue is empty.

#### 2. Path through Walls
**Symptom:** Path cuts corners.
**Cause:** Diagonal movement allows squeezing through a 0-width gap between two diagonal obstacles.
**Solution:** Check neighbors of neighbors or inflate obstacles.

#### 3. RRT Jaggedness
**Symptom:** Path looks like lightning.
**Cause:** Random sampling.
**Solution:** Apply path smoothing (e.g., shortcutting) after finding the path.

---

## ⚡ Optimization & Best Practices

### 1. Hybrid A*
Standard A* produces discrete grid steps (non-holonomic robots can't turn 90 deg instantly).
**Hybrid A*** uses continuous coordinates $(x, y, \theta)$ inside grid cells and uses a kinematic model (Reeds-Shepp curves) for transitions. Used in parking.

### 2. Bidirectional Search
Search from Start to Goal AND Goal to Start simultaneously.
-   Meet in the middle.
-   Drastically reduces search space ($2 \times r^2 < (2r)^2$).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What makes a heuristic "Admissible"?
    *   **A:** It never overestimates the cost to reach the goal.
2.  **Q:** Why is RRT preferred over A* for a 6-DOF robot arm?
    *   **A:** A* requires a grid. A 6D grid is massive ($100^6$ cells). RRT samples space sparsely and finds a path quickly without filling the grid.
3.  **Q:** What is the difference between RRT and RRT*?
    *   **A:** RRT* rewires the tree to minimize path cost, converging to the optimal solution. RRT just finds *a* solution.

### Challenge Task
**Task:** Path Smoothing.
1.  Take the RRT path.
2.  Pick two random points on the path.
3.  Check if a straight line connects them without collision.
4.  If yes, replace the jagged segment with the straight line.
5.  Repeat 100 times.

---

## 📚 Further Reading & References
-   [Planning Algorithms (LaValle)](http://planning.cs.uiuc.edu/) - The Bible of Planning.
-   [PythonRobotics Repo](https://github.com/AtsushiSakai/PythonRobotics) - Implementations of every algorithm.

---

**Day 29 Complete** | Phase 4: ADAS & Robotics Systems | Week 5: Path Planning & Decision Making
