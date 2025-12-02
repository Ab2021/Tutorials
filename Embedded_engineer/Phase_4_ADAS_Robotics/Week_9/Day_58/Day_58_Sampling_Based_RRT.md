# Day 58: Sampling-Based Planning (RRT, RRT*)
## Phase 4: ADAS & Robotics Systems | Week 9: Path Planning & Navigation

---

> **📝 Day 58 Focus:**
> A* is great for 2D grids. But what if you have a 7-DOF robot arm? A grid would have $100^7$ cells. Impossible. **Sampling-Based Planners** like **RRT (Rapidly-exploring Random Tree)** solve this by randomly exploring the space. They are fast, probabilistic, and essential for high-dimensional robotics.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the "Curse of Dimensionality" and why grids fail for arms/drones.
2.  **Implement** the RRT Algorithm: Sample -> Nearest -> Steer -> Check.
3.  **Upgrade** to RRT*: Adding "Rewiring" to asymptotically approach the optimal path.
4.  **Visualize** the tree growth and how it explores free space.
5.  **Compare** RRT (Fast, Suboptimal) vs RRT* (Slower, Optimal).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Geometry:** Euclidean Distance.
-   **Probability:** Random Sampling.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The RRT Algorithm

Designed to handle high-dimensional spaces with complex obstacles.
**Core Loop:**
1.  **Sample:** Pick a random point $q_{rand}$ in C-Space.
2.  **Nearest:** Find the closest existing node $q_{near}$ in the tree.
3.  **Steer:** Move from $q_{near}$ towards $q_{rand}$ by a small step $\delta$. This is $q_{new}$.
4.  **Collision Check:** Is the path from $q_{near}$ to $q_{new}$ collision-free?
5.  **Add:** If yes, add $q_{new}$ to the tree. Link it to $q_{near}$.

**Bias:** Occasionally (e.g., 5% of the time), sample the **Goal** instead of a random point. This pulls the tree towards the destination.

### 🔹 Part 2: RRT* (The Optimizer)

RRT finds *a* path, but it's usually jagged and long. RRT* fixes this.
**Rewiring:**
1.  When adding $q_{new}$, look at all neighbors within radius $r$.
2.  **Choose Parent:** Connect $q_{new}$ to the neighbor that gives the lowest total cost from Start.
3.  **Rewire Neighbors:** Check if passing through $q_{new}$ would lower the cost for any neighbor. If so, change their parent to $q_{new}$.

### 🔹 Part 3: Probabilistic Completeness

-   **Complete:** Finds a solution if one exists (A*).
-   **Probabilistically Complete:** As $N \to \infty$, the probability of finding a solution $\to 1$. (RRT).
-   **Asymptotically Optimal:** As $N \to \infty$, the path converges to the optimal solution. (RRT*).

---

## 💻 Implementation: RRT & RRT*

**Scenario:**
-   **Map:** 2D space with circular obstacles.
-   **Start:** (0, 0).
-   **Goal:** (6, 10).

### 🛠️ Setup
Create `week9_day58` and `rrt.py`.

```bash
mkdir -p ~/ros2_ws/src/week9_day58
cd ~/ros2_ws/src/week9_day58
touch rrt.py
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

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=3.0, path_resolution=0.5, goal_sample_rate=5, max_iter=500):
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
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
            # 1. Sample
            rnd_node = self.get_random_node()
            
            # 2. Nearest
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            
            # 3. Steer
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            
            # 4. Collision Check
            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)
            
            # Check Goal
            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
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
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # Stop drawing if closed
        if plt.gcf().number != 1: return 

        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        # Draw Obstacles
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        # Draw Tree
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        # Draw Goal/Start
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
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
    # (x, y, radius)
    obstacleList = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1)
    ] 
    
    # RRT
    rrt = RRT(
        start=[0, 0],
        goal=[6, 10],
        rand_area=[-2, 15],
        obstacle_list=obstacleList
    )
    path = rrt.planning(animation=True)

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

## 🔬 Lab Exercise: RRT vs RRT*

### Lab Objectives
1.  Run the RRT code.
2.  **Observation:** The green tree grows rapidly. The final red path is jagged and random. It works, but it's ugly.
3.  **Challenge:** Implement RRT*.
    -   Add a `cost` attribute to Node.
    -   In `planning`, after finding `new_node`:
        -   Find `near_nodes` (within radius).
        -   **Choose Parent:** Find which near node gives lowest cost to `new_node`.
        -   **Rewire:** Check if `new_node` can be a better parent for any `near_node`.
    -   *Result:* The tree will look more like a lightning bolt (straight lines). The path will converge to a straight line.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Stuck in Obstacles
**Symptom:** Tree stops growing.
**Cause:** `expand_dis` is too large (jumps over obstacles) or too small (takes forever).
**Solution:** Tune step size. Ensure collision check covers the *entire path segment*, not just endpoints.

#### 2. Goal Unreachable
**Symptom:** Max iterations reached.
**Cause:** Narrow passage problem (Bug trap).
**Solution:** Use **Bi-directional RRT** (Grow two trees, one from Start, one from Goal, and meet in the middle).

---

## ⚡ Optimization & Best Practices

### 1. KD-Tree
Finding `nearest_node` is $O(N)$.
Using a **KD-Tree** makes it $O(\log N)$.
Essential for large trees ($N > 5000$).

### 2. Informed RRT*
Once a solution is found, restrict sampling to an **Ellipsoid** defined by the Start, Goal, and current path length.
-   Focuses samples on improving the existing solution rather than exploring useless space.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is RRT better than A* for a 7-DOF arm?
    *   **A:** A* needs a grid. A 7D grid is too big. RRT samples the space sparsely, finding a path without filling the whole volume.
2.  **Q:** What is "Rewiring"?
    *   **A:** The process in RRT* where we check if a new node offers a shorter path to existing nodes. If so, we change their parent pointer to the new node.
3.  **Q:** Does RRT guarantee the shortest path?
    *   **A:** No. RRT* does (asymptotically).

### Challenge Task
**Task:** Dubins RRT.
1.  Replace the straight-line `steer` function with **Dubins Curves** (Car-like kinematics: Turn Left, Straight, Turn Right).
2.  Now the generated path is drivable by a non-holonomic robot (car).

---

## 📚 Further Reading & References
-   [LaValle's RRT Page](http://msl.cs.uiuc.edu/rrt/)
-   [OMPL (Open Motion Planning Library)](https://ompl.kavrakilab.org/)

---

**Day 58 Complete** | Phase 4: ADAS & Robotics Systems | Week 9: Path Planning & Navigation
