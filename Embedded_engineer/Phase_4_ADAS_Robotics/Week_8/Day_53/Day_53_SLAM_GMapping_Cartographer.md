# Day 53: SLAM (Simultaneous Localization and Mapping)
## Phase 4: ADAS & Robotics Systems | Week 8: State Estimation & Sensor Fusion

---

> **📝 Day 53 Focus:**
> Localization assumes you have a map. Mapping assumes you know your location. What if you have neither? This is the **SLAM (Simultaneous Localization and Mapping)** problem—the "Holy Grail" of mobile robotics. Today, we explore how robots build maps while exploring the unknown.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the SLAM problem: $P(m, x_{1:t} | z_{1:t}, u_{1:t})$.
2.  **Explain** Occupancy Grid Mapping (Inverse Sensor Model).
3.  **Compare** Filter-based SLAM (GMapping/FastSLAM) vs. Graph-based SLAM (Cartographer/Karto).
4.  **Implement** a simple Occupancy Grid Mapper in Python.
5.  **Understand** the concept of Loop Closure and why it fixes map drift.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 50:** Bayes Filter.
-   **Day 52:** Particle Filter.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Chicken and Egg

-   **Mapping:** Given Poses $x$, find Map $m$. (Easy).
-   **Localization:** Given Map $m$, find Poses $x$. (Easy - MCL/EKF).
-   **SLAM:** Find both $m$ and $x$. (Hard).

### 🔹 Part 2: Occupancy Grid Mapping

We represent the world as a grid of cells.
Each cell $m_i$ is a binary random variable (Occupied/Free).
-   **Log-Odds Representation:** $l_i = \log \frac{P(m_i)}{1 - P(m_i)}$.
-   **Update Rule:** $l_{t,i} = l_{t-1,i} + \text{inverse\_sensor\_model}(m_i, x_t, z_t) - l_0$.
-   **Inverse Sensor Model:**
    -   If cell is in front of the hit: Decrease probability (Free).
    -   If cell is at the hit: Increase probability (Occupied).
    -   If cell is behind the hit: Unknown (Don't change).

### 🔹 Part 3: SLAM Approaches

#### 3.1 GMapping (FastSLAM)
-   Uses **Rao-Blackwellized Particle Filter**.
-   Each particle carries its *own map*.
-   Robot Pose is estimated by PF. Map is built for each particle.
-   **Pros:** Efficient for small environments.
-   **Cons:** Memory hungry (Map $\times$ Particles). Particle depletion.

#### 3.2 Cartographer (Graph SLAM)
-   **Local SLAM:** Builds small "Submaps" using scan matching.
-   **Global SLAM:** Optimizes a "Pose Graph".
-   **Loop Closure:** When the robot returns to a known place, it adds a constraint edge to the graph and optimizes the whole trajectory to close the loop (fix drift).

---

## 💻 Implementation: Occupancy Grid Mapping

**Scenario:**
-   **Robot:** Moves in a straight line.
-   **Sensor:** 1D Range Sensor (measures distance to wall).
-   **Task:** Build a 2D Grid Map from known poses (Mapping with Known Poses).

### 🛠️ Setup
Create `week8_day53` and `grid_mapping.py`.

```bash
mkdir -p ~/ros2_ws/src/week8_day53
cd ~/ros2_ws/src/week8_day53
touch grid_mapping.py
```

### 👨‍💻 Code: Grid Mapping Algorithm

```python
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Configuration ---
GRID_SIZE = 100 # 100x100 cells
RESOLUTION = 0.5 # meters per cell
L_OCC = np.log(0.9 / 0.1) # Log odds for Occupied
L_FREE = np.log(0.4 / 0.6) # Log odds for Free
L_PRIOR = np.log(0.5 / 0.5) # 0

class GridMap:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE)) # Log-odds map
        
    def to_probability(self):
        return 1.0 - (1.0 / (1.0 + np.exp(self.grid)))

    def update(self, x_rob, y_rob, theta_rob, scan_ranges):
        # x_rob, y_rob: Robot position (meters)
        # theta_rob: Robot heading (radians)
        # scan_ranges: List of ranges (meters)
        
        # Robot Grid Index
        cx = int(x_rob / RESOLUTION)
        cy = int(y_rob / RESOLUTION)
        
        # Ray Casting (Bresenham or DDA)
        angle_step = np.deg2rad(1.0) # 1 degree resolution
        start_angle = -np.pi/4 # -45 deg
        
        for i, r in enumerate(scan_ranges):
            if r > 10.0: continue # Max range
            
            angle = theta_rob + start_angle + i * angle_step
            
            # End Point (Hit)
            hit_x = x_rob + r * math.cos(angle)
            hit_y = y_rob + r * math.sin(angle)
            
            hx = int(hit_x / RESOLUTION)
            hy = int(hit_y / RESOLUTION)
            
            # Trace Ray
            self.trace_ray(cx, cy, hx, hy)
            
            # Mark Hit as Occupied
            if 0 <= hx < GRID_SIZE and 0 <= hy < GRID_SIZE:
                self.grid[hx, hy] += L_OCC

    def trace_ray(self, x0, y0, x1, y1):
        # Bresenham's Line Algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if x == x1 and y == y1: break
            
            # Mark Free
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                self.grid[x, y] += L_FREE
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

def main():
    gmap = GridMap()
    
    # Ground Truth Map (Walls at x=10, x=40)
    # Robot moves from (5, 25) to (45, 25)
    
    plt.figure(figsize=(8, 8))
    
    for t in range(40):
        # Robot Pose
        x_rob = 5.0 + t * 1.0
        y_rob = 25.0
        theta_rob = 0.0
        
        # Simulate Lidar (Detects walls at x=10 and x=40)
        ranges = []
        for i in range(90): # 90 rays (-45 to +45)
            angle = theta_rob + np.deg2rad(i - 45)
            
            # Ray Intersection with x=10
            d1 = 100.0
            if math.cos(angle) != 0:
                d = (10.0 - x_rob) / math.cos(angle)
                if d > 0: d1 = d
            
            # Ray Intersection with x=40
            d2 = 100.0
            if math.cos(angle) != 0:
                d = (40.0 - x_rob) / math.cos(angle)
                if d > 0: d2 = d
                
            r = min(d1, d2)
            ranges.append(r)
            
        # Update Map
        gmap.update(x_rob, y_rob, theta_rob, ranges)
        
        # Visualization
        if t % 5 == 0:
            plt.cla()
            plt.imshow(gmap.to_probability().T, origin='lower', cmap='Greys', extent=[0, GRID_SIZE*RESOLUTION, 0, GRID_SIZE*RESOLUTION])
            plt.plot(x_rob, y_rob, 'or') # Robot
            plt.title(f"Step {t}")
            plt.pause(0.1)
            
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Unknown World

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   Initially, the map is Gray (0.5 probability, Unknown).
    -   As the robot moves, the area in front becomes White (Free).
    -   The walls at x=10 and x=40 become Black (Occupied).
    -   The area behind the walls remains Gray (Shadowed).
3.  **Experiment:**
    -   Change `L_OCC` to a smaller value.
    -   *Result:* The walls take longer to appear (requires more hits). This is useful for filtering out dynamic obstacles (people walking by).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Map Smearing
**Symptom:** Walls look thick or blurry.
**Cause:** Poor localization (Pose error).
**Solution:** This code assumes *Known Poses*. In real SLAM, if the pose is wrong, the map is wrong. This is why SLAM is hard.

#### 2. Ray Casting Artifacts
**Symptom:** Dotted lines instead of solid free space.
**Cause:** Angular resolution is too low (gaps between rays at long distance).
**Solution:** Increase ray density or use a "Model-based" update (Polygon filling).

---

## ⚡ Optimization & Best Practices

### 1. Octomap
Grids are memory inefficient for 3D ($1000^3$ cells).
**Octomap:** Uses an Octree structure.
-   Large empty spaces are stored as a single large node.
-   Detail is only added where obstacles exist.
-   Standard for 3D SLAM.

### 2. TSDF (Truncated Signed Distance Field)
Instead of Probability, store the Distance to the nearest obstacle.
-   Used in KinectFusion/Dense SLAM.
-   Allows for smooth surface reconstruction (Meshing).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the "Loop Closure" problem?
    *   **A:** When a robot returns to a previously visited location, the accumulated drift might make it think it's in a new place. Loop closure algorithms detect the match and "snap" the trajectory back, correcting the map.
2.  **Q:** Why use Log-Odds?
    *   **A:** Probabilities are bounded [0, 1]. Adding them is tricky. Log-Odds are $(-\infty, \infty)$. We can just add/subtract evidence linearly.
3.  **Q:** What is the difference between GMapping and Cartographer?
    *   **A:** GMapping is a Particle Filter (Filter-based). Cartographer is an Optimization problem (Graph-based). Cartographer is generally more robust for large loops.

### Challenge Task
**Task:** Noisy Poses.
1.  Add noise to `x_rob` in the `main` loop (but not in the `update` call).
2.  Observe how the map becomes blurry.
3.  This demonstrates why Mapping needs Localization (SLAM), not just Mapping.

---

## 📚 Further Reading & References
-   [OpenSLAM.org](https://openslam-org.github.io/)
-   [Cartographer Documentation](https://google-cartographer.readthedocs.io/en/latest/)

---

**Day 53 Complete** | Phase 4: ADAS & Robotics Systems | Week 8: State Estimation & Sensor Fusion
