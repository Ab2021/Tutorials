# Day 118: Occupancy Grid Mapping
## Phase 4: ADAS & Robotics Systems | Week 17: Sensor Fusion

---

> **📝 Day 118 Focus:**
> Tracking objects is great for cars, but what about walls, curbs, and trees? For navigation, we need a map of **Free Space**. The **Occupancy Grid Map (OGM)** divides the world into cells and estimates the probability that each cell is occupied.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the Binary Bayes Filter for mapping.
2.  **Convert** Probability to Log-Odds (to enable additive updates).
3.  **Define** the Inverse Sensor Model (Ray Casting).
4.  **Implement** a Lidar Mapping algorithm in Python.
5.  **Visualize** the Grid Map evolving over time.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Probability:** Bayes Rule.
-   **Geometry:** Bresenham's Line Algorithm (Ray Tracing).

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Grid

We divide the world into a 2D grid (e.g., $10cm \times 10cm$ cells).
Each cell $m_i$ is a binary random variable:
-   $P(m_i = 1)$: Occupied.
-   $P(m_i = 0)$: Free.

### 🔹 Part 2: Log-Odds Notation

Bayes Rule involves multiplication, which is slow and unstable for small floats.
We use **Log-Odds ($l$)**:
$$ l(x) = \log \left( \frac{P(x)}{1 - P(x)} \right) $$
-   $P=0.5 \implies l=0$ (Unknown).
-   $P=0.99 \implies l \approx 4.6$ (Occupied).
-   $P=0.01 \implies l \approx -4.6$ (Free).

**Update Rule:**
$$ l_t(m_i) = l_{t-1}(m_i) + \text{inverse\_sensor\_model}(z_t) - l_0 $$
It becomes simple addition!

### 🔹 Part 3: Inverse Sensor Model

Given a Lidar ray that hits an obstacle at distance $d$:
1.  **Cells along the ray ($r < d$):** Probability decreases (Free). Update: $l = l - 0.4$.
2.  **Cell at the hit ($r \approx d$):** Probability increases (Occupied). Update: $l = l + 0.8$.
3.  **Cells behind ($r > d$):** No information. Update: $l = l + 0$.

---

## 💻 Implementation: Lidar Mapper

**Scenario:**
-   Robot at $(50, 50)$.
-   Lidar scans a wall at $x=80$.
-   Task: Update the grid.

### 🛠️ Setup
Create `week17_day118` and `occupancy_grid.py`.

```bash
mkdir -p ~/ros2_ws/src/week17_day118
cd ~/ros2_ws/src/week17_day118
touch occupancy_grid.py
```

### 👨‍💻 Code: OGM Algorithm

```python
import numpy as np
import matplotlib.pyplot as plt

class OccupancyGrid:
    def __init__(self, width, height, resolution):
        self.w = width
        self.h = height
        self.res = resolution
        
        # Grid dimensions
        self.nx = int(width / resolution)
        self.ny = int(height / resolution)
        
        # Log-Odds Grid (Initialized to 0 -> P=0.5)
        self.grid = np.zeros((self.nx, self.ny))
        
        # Parameters
        self.l_occ = np.log(0.9 / 0.1)  # ~2.2
        self.l_free = np.log(0.4 / 0.6) # ~-0.4
        self.l_max = 5.0  # Clamp
        self.l_min = -5.0 # Clamp

    def world_to_grid(self, x, y):
        gx = int(x / self.res)
        gy = int(y / self.res)
        return gx, gy

    def bresenham_ray(self, x0, y0, x1, y1):
        # Returns list of grid cells along the line
        cells = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        cx, cy = x0, y0
        
        while True:
            cells.append((cx, cy))
            if cx == x1 and cy == y1: break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy
                
        return cells

    def update(self, robot_x, robot_y, scan_ranges, scan_angles):
        gx0, gy0 = self.world_to_grid(robot_x, robot_y)
        
        for r, theta in zip(scan_ranges, scan_angles):
            # Calculate Hit Point
            hit_x = robot_x + r * np.cos(theta)
            hit_y = robot_y + r * np.sin(theta)
            
            gx1, gy1 = self.world_to_grid(hit_x, hit_y)
            
            # Check bounds
            if not (0 <= gx1 < self.nx and 0 <= gy1 < self.ny):
                continue
                
            # Ray Tracing
            cells = self.bresenham_ray(gx0, gy0, gx1, gy1)
            
            # Update Cells
            for (cx, cy) in cells[:-1]: # Free Space
                if 0 <= cx < self.nx and 0 <= cy < self.ny:
                    self.grid[cx, cy] += self.l_free
                    
            # Update Hit Cell (Occupied)
            cx, cy = cells[-1]
            if 0 <= cx < self.nx and 0 <= cy < self.ny:
                self.grid[cx, cy] += self.l_occ
                
        # Clamp
        np.clip(self.grid, self.l_min, self.l_max, out=self.grid)

    def plot(self):
        # Convert Log-Odds to Probability
        # P = 1 - 1 / (1 + exp(l))
        prob_map = 1 - 1 / (1 + np.exp(self.grid))
        
        plt.figure(figsize=(8, 8))
        plt.imshow(prob_map.T, origin='lower', cmap='Greys', extent=[0, self.w, 0, self.h])
        plt.colorbar(label="Occupancy Probability")
        plt.title("Occupancy Grid Map")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.show()

def main():
    # 100x100m world, 0.5m resolution
    ogm = OccupancyGrid(100, 100, 0.5)
    
    # Robot at center
    rx, ry = 50.0, 50.0
    
    # Simulate Lidar Scan (360 degrees)
    angles = np.linspace(0, 2*np.pi, 360)
    ranges = []
    
    for a in angles:
        # Wall at x=80
        # Ray: x = 50 + r*cos(a) = 80 -> r = 30/cos(a)
        if -np.pi/4 < a < np.pi/4:
            r = 30.0 / np.cos(a)
        else:
            r = 40.0 # Max range
        ranges.append(r)
        
    print("Updating Map...")
    ogm.update(rx, ry, ranges, angles)
    
    print("Plotting...")
    ogm.plot()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Moving Robot

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   White area (P=0) around the robot (Free Space).
    -   Black line (P=1) at $x=80$ (The Wall).
    -   Gray area (P=0.5) behind the wall (Unknown).
3.  **Experiment:**
    -   Move the robot to $(60, 50)$ and scan again.
    -   **Result:** The Free Space expands. The map becomes more complete.
    -   This is **SLAM** (Simultaneous Localization and Mapping) in action (assuming we know the location).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Discretization Errors
**Symptom:** Thin walls have holes (Aliasing).
**Cause:** Bresenham's line skips cells if resolution is too coarse.
**Solution:** Use a finer resolution (e.g., 5cm) or "Super-Cover" line algorithms that visit every touched cell.

#### 2. Dynamic Objects
**Symptom:** A moving car leaves a trail of "Occupied" cells (Smearing).
**Cause:** OGM assumes a static world.
**Solution:** Use a **Decay Factor**. If a cell is not observed occupied for a while, slowly lower its probability back to 0.5.

---

## ⚡ Optimization & Best Practices

### 1. Octomap (3D)
For 3D mapping (Drones), a 3D grid is too memory intensive ($N^3$).
-   **Octree:** A tree structure.
-   Start with one big cube. If it's partially occupied, split into 8 smaller cubes. Repeat.
-   Efficient storage for sparse environments.

### 2. GPU Acceleration
Ray tracing is parallelizable.
-   Use CUDA to process 1000 rays simultaneously.
-   Libraries: `nvblox` (NVIDIA Isaac ROS).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why Log-Odds?
    *   **A:** To turn probability multiplication into addition. Faster and avoids underflow.
2.  **Q:** What is the Inverse Sensor Model?
    *   **A:** A function $P(m | z)$ that tells us the probability of occupancy given a measurement. (Usually: High at hit, Low along ray).
3.  **Q:** What happens if I scan the same wall 100 times?
    *   **A:** The log-odds accumulate until they hit `l_max`. The map becomes very confident (P=0.9999).

### Challenge Task
**Task:** Noise Filtering.
1.  Add random noise to Lidar ranges ($\pm 0.5m$).
2.  Run the update 10 times.
3.  Observe that the wall becomes "thick" and fuzzy.
4.  This represents the uncertainty of the sensor in the map.

---

## 📚 Further Reading & References
-   [Probabilistic Robotics (Thrun)](https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf)
-   [Octomap Library](https://octomap.github.io/)

---

**Day 118 Complete** | Phase 4: ADAS & Robotics Systems | Week 17: Sensor Fusion
