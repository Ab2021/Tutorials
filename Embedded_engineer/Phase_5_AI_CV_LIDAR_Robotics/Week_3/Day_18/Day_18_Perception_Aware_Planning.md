# Day 18: Perception-Aware Planning
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 3: Advanced Navigation & Planning

---

> **📝 Content Creator Instructions:**
> Usually, planning assumes perception is a solved problem given to it. But sometimes we must **Move to See**.
> - **Focus:** Active Perception, Next-Best-View (NBV), and Entropy reduction.
> - **Code:** Implementation of an Information Gain-based Exploration planner.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** Active Perception and the Exploration-Exploitation trade-off in robotics.
2.  **Calculate** Information Gain (Entropy Reduction) for a candidate sensor pose.
3.  **Implement** a Next-Best-View (NBV) planner for autonomous mapping.
4.  **Execute** Frontier-Based Exploration to map an unknown environment.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Lidar or Depth Camera.

### Software Environment
```bash
pip install numpy scipy matplotlib scikit-image
```

### Prior Knowledge
- Occupancy Grids.
- Raycasting.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: "Moving to See"

Standard Planning: Minimize Distance ($J = \int ds$).
Perception-Aware Planning: Minimize Uncertainty ($J = H(\text{Map})$).

**The Problem:**
A robot standing in a hallway has a partially known map.
*   **Frontier:** The boundary between "Known Free" space and "Unknown" space.
*   To complete the map, the robot must move to a pose where its sensor *covers* the unknown space.

### 🔹 Part 2: Information Theory Metrics

**Entropy ($H$):** Measure of Uncertainty.
For a binary grid cell $c_i$:
$$ H(c_i) = -p \log p - (1-p) \log (1-p) $$
*   If $p=0.5$ (Unknown), Entropy is Max (1.0).
*   If $p=0.0$ (Free) or $p=1.0$ (Occupied), Entropy is Min (0.0).

**Information Gain ($IG$):**
Reduction in Entropy expected from a new measurement $z$.
$$ IG(z) = H(M) - H(M | z) $$

### 🔹 Part 3: Next-Best-View (NBV)

Algorithm:
1.  Generate candidate poses $P = \{p_1, ..., p_K\}$.
2.  For each candidate, simulate a sensor scan (Raycast into the current map).
3.  Count how many "Unknown" voxels the ray hits.
4.  Select $p^* = \arg \max \text{InformationGain}(p) - \lambda \cdot \text{Cost}(p)$.

---

## 💻 Implementation: Autonomous Explorer

We will build a Frontier-Based Explorer from scratch.

### 🛠️ Project Structure
```text
day18_exploration/
├── src/
│   ├── grid_map.py
│   ├── raycaster.py
│   └── planner.py
└── run_sim.py
```

### 👨‍💻 Code Implementation (`src/planner.py`)

```python
import numpy as np
from scipy.ndimage import binary_dilation

class FrontierPlanner:
    def __init__(self, map_wrapper):
        self.map = map_wrapper # GridMap Class
        
    def find_frontiers(self):
        """
        Frontier = Free Cell adjacent to Unknown Cell.
        """
        grid = self.map.grid # 0=Free, 1=Occupied, 0.5=Unknown
        
        # Masks
        free_mask = (grid == 0)
        unknown_mask = (grid == 0.5)
        
        # Dilate free space
        dilated_free = binary_dilation(free_mask)
        
        # Intersection: Unknown cells that touch Free cells
        frontier_mask = unknown_mask & dilated_free
        
        # Cluster frontiers (Connected Components)
        # Returns list of centroids [x, y]
        return self._cluster_frontiers(frontier_mask)

    def select_goal(self, robot_pos, frontiers):
        """
        Score = Information (Size of cluster) - Cost (Distance)
        """
        best_score = -np.inf
        best_goal = None
        
        for f in frontiers:
            dist = np.linalg.norm(f - robot_pos)
            # Simple Utility
            utility = 1.0 # Could be size of frontier cluster
            
            score = utility - 0.1 * dist
            
            if score > best_score:
                best_score = score
                best_goal = f
                
        return best_goal
```

### 👨‍💻 Helper: Raycasting Information Gain (`src/raycaster.py`)

```python
import numpy as np

def compute_ig(robot_pose, grid, sensor_range):
    """
    Simulate rays. Count how many Unknown cells (0.5) we hit.
    """
    ig = 0
    angles = np.linspace(-np.pi/2, np.pi/2, 30) # 180 FOV
    
    for angle in angles:
        global_angle = robot_pose[2] + angle
        
        # March Ray
        for r in range(1, sensor_range):
            x = int(robot_pose[0] + r * np.cos(global_angle))
            y = int(robot_pose[1] + r * np.sin(global_angle))
            
            if not is_inside(grid, x, y): break
            
            val = grid[y, x]
            if val == 1: break # Hit obstacle
            if val == 0.5:
                ig += 1 # Gained info!
                
    return ig
```

---

## 🔬 Lab Exercise: Mapping the Void

### 1. Lab Objectives
- Use a 2D Simulator (matplotlib based).
- Robot starts in center of grey (Unknown) map.
- Robot must autonomously pick goals to explore the whole map.
- **Metric:** $\%$ of Map Known vs. Time.

### 2. Step-by-Step Guide
1.  Initialize Grid (all 0.5).
2.  Add Wall Obstacles (Ground Truth).
3.  Simulate: `scan = GroundTruth(pose)` -> `grid.update(scan)`.
4.  Plan: `goal = planner.select_goal()`.
5.  Move: `pose = navigate(goal)`.
6.  Loop until Frontiers is empty.

### 3. Expected Output
- The robot spirals out, visiting the closest unknown areas first.
- It enters rooms, scans them, and exits.
- Finally, it stops when the map is complete.

---

## 🚀 Project: "Active Object Search"

**Goal:** Find the "Red Ball" in the apartment.
**Difference:** The goal is not to map *everything*, but to find a specific object.
**Heuristic:**
*   If we see a "Door", probability of finding new objects is high behind it.
*   Prioritize Frontiers near "Door" detections (Semantic NBV).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Robot Oscillates between two goals"
*   **Symptom:** Goes halfway to A, then decides B is closer, turns around, repeats.
*   **Cause:** Re-planning every step with slightly changing costs.
*   **Fix:** **Hysteresis**. Stick to the current goal until reached, or until utility drops significantly ($>20\%$).

#### 2. "Getting Stuck"
*   **Symptom:** No frontiers found (map is closed), but robot thinks it's done even if holes exist.
*   **Cause:** Frontiers too small (1 pixel).
*   **Fix:** Filter frontiers by min cluster size ($>5$ pixels).

---

## ⚡ Optimization: RRT-of-Trees

Instead of evaluating *every* pixel, sample random viewpoints (RRT) and evaluate IG only on the nodes of the tree.
*   **Result:** Rapid exploration of large 3D spaces (Receding Horizon NBV).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is a "Frontier"?
    *   **A:** The edge between Known Free space and Unknown space. It represents the potential for new information.
2.  **Q:** Why not just move randomly?
    *   **A:** Random walk covers space in $O(T^2)$. Directed exploration covers space in $O(T)$. Efficiency matters for battery life.

### Challenge Task
> **Task:** Implement 3D Exploration (Octomap).
> 1. Use `octomap` library.
> 2. Calculate volumetric Information Gain.
> 3. Fly a drone to map a 3D statue.

---

## 📚 Further Reading
- **Frontier-Based Exploration:** Yamauchi (1997).
- **Receding Horizon NBV:** Bircher et al. (ICRA 2016).
- **OctoMap:** Hornung et al.

---

**Day 18 Complete**
