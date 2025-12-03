# Day 135: Local Planning (RRT/RRT*)
## Phase 4: ADAS & Robotics Systems | Week 20: Path Planning & Decision Making

---

> **📝 Day 135 Focus:**
> A* is great for 2D grids. But what if you have a 18-DOF robot arm? Or a car with non-holonomic constraints? Grids fail (Curse of Dimensionality). **RRT (Rapidly-exploring Random Trees)** solves this by randomly sampling the space.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the principle of Sampling-Based Planning.
2.  **Implement** the RRT algorithm (Sample -> Nearest -> Steer -> Check).
3.  **Upgrade** to RRT* (Rewiring) for asymptotic optimality.
4.  **Visualize** the tree growth in a maze.
5.  **Compare** RRT vs A* in terms of speed and path quality.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Geometry:** Euclidean Distance.
-   **Probability:** Uniform Sampling.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The RRT Algorithm

Goal: Grow a tree from Start to Goal.
1.  **Sample:** Pick a random point $x_{rand}$ in space. (With 5% chance, pick Goal).
2.  **Nearest:** Find the closest node $x_{near}$ in the existing tree.
3.  **Steer:** Move from $x_{near}$ towards $x_{rand}$ by distance $\Delta q$. This is $x_{new}$.
4.  **Collision Check:** If path to $x_{new}$ is clear, add $x_{new}$ to tree.
5.  **Repeat:** Until $x_{new}$ is close to Goal.

### 🔹 Part 2: RRT* (The Optimal Version)

RRT finds *a* path, but it's usually jagged and long. RRT* fixes this.
-   **Choose Parent:** When adding $x_{new}$, look at all neighbors. Pick the parent that gives the lowest total cost to Start.
-   **Rewire:** After adding $x_{new}$, check if $x_{new}$ can be a better parent for existing neighbors. If so, rewire them to connect to $x_{new}$.
-   **Result:** As $N \to \infty$, the path converges to the optimal solution.

---

## 💻 Implementation: RRT Planner

**Scenario:**
-   2D Space with circular obstacles.
-   Start: (0, 0). Goal: (6, 10).

### 🛠️ Setup
Create `week20_day135` and `rrt_planner.py`.

```bash
mkdir -p ~/ros2_ws/src/week20_day135
cd ~/ros2_ws/src/week20_day135
touch rrt_planner.py
```

### 👨‍💻 Code: RRT Implementation

```python
import math
import random
import matplotlib.pyplot as plt
import numpy as np

class RRT:
    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=3.0, path_resolution=0.5, goal_sample_rate=5, max_iter=500):
        self.start = self.Node(start[0], start[1])
        self.goal = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self, animation=True):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.goal, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node
        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.goal.x
        dy = y - self.goal.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.goal.x, self.goal.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(math.radians(d)) for d in deg]
        yl = [y + size * math.sin(math.radians(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def check_collision(node, obstacleList):
        if node is None:
            return False
        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]
            if min(d_list) <= size**2:
                return False  # collision
        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

def main():
    print("Start " + __file__)
    # [x, y, size]
    obstacleList = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1)
    ]  # [x,y,size]
    
    # Set Initial parameters
    rrt = RRT(
        start=[0, 0],
        goal=[6, 10],
        rand_area=[-2, 15],
        obstacle_list=obstacleList
    )
    path = rrt.planning(animation=False) # Set True to see animation

    if path is None:
        print("Cannot find path")
    else:
        print("Found path!!")
        # Draw final path
        rrt.draw_graph()
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    main()
```

---

## 🔬 Lab Exercise: The Random Tree

### Lab Objectives
1.  Run the script.
2.  **Observation:** The Green tree spreads out like lightning. It explores empty spaces rapidly.
3.  **Experiment:**
    -   Set `goal_sample_rate = 0`.
    -   **Result:** The tree grows blindly. It might never find the goal.
    -   Set `goal_sample_rate = 50`.
    -   **Result:** The tree rushes towards the goal. If there's a wall in between, it gets stuck trying to go through it.
    -   **Lesson:** Balance exploration (Random) and exploitation (Goal Bias).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Narrow Passages
**Symptom:** RRT fails to find a path through a small door.
**Cause:** The probability of sampling a point *inside* the doorway is tiny.
**Solution:** Use **RRT-Connect** (Grow two trees, one from Start, one from Goal, and try to meet).

#### 2. Jerky Path
**Symptom:** The path zig-zags wildly.
**Cause:** Random sampling.
**Solution:** Post-processing. Apply **Path Smoothing** (e.g., B-Splines or Shortcut heuristic).

---

## ⚡ Optimization & Best Practices

### 1. Kinodynamic RRT
Standard RRT assumes you can move in any direction (Holonomic). Cars cannot (Non-Holonomic).
-   **Solution:** In the `steer` function, instead of drawing a straight line, simulate the vehicle dynamics ($x, y, \theta, v, \phi$) for time $\Delta t$.
-   Ensures the path is drivable.

### 2. Informed RRT*
Once a solution is found, restrict sampling to an **Ellipsoid** defined by the Start, Goal, and current path length.
-   Focuses optimization on the relevant region.
-   Converges much faster.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is RRT "Rapidly-exploring"?
    *   **A:** Because the Voronoi bias means nodes with large empty spaces around them are more likely to be selected as "Nearest", pulling the tree into unexplored areas.
2.  **Q:** What is the main difference between RRT and RRT*?
    *   **A:** RRT* includes a "Rewiring" step that optimizes the tree structure, guaranteeing asymptotic optimality.
3.  **Q:** When should I use RRT over A*?
    *   **A:** High-dimensional spaces (Robot Arms, 6DOF) or continuous spaces where discretization is difficult.

### Challenge Task
**Task:** RRT-Connect.
1.  Create two RRT instances: `rrt_start` and `rrt_goal`.
2.  Grow `rrt_start` towards a random point. Let new node be `q_new`.
3.  Grow `rrt_goal` towards `q_new`.
4.  If they connect, merge paths.
5.  This is much faster for narrow passages.

---

## 📚 Further Reading & References
-   [RRT Paper (LaValle)](http://msl.cs.uiuc.edu/~lavalle/papers/Lav98c.pdf)
-   [PythonRobotics RRT*](https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning/RRTStar)

---

**Day 135 Complete** | Phase 4: ADAS & Robotics Systems | Week 20: Path Planning & Decision Making
