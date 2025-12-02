# Day 26: Localization with HD Maps (NDT)
## Phase 4: ADAS & Robotics Systems | Week 4: Localization & Mapping (SLAM)

---

> **📝 Day 26 Focus:**
> Building a map is hard (SLAM). Using a map is easier (Localization). In autonomous driving, we usually build the map offline (Mapping Mode) and then just localize against it online (Localization Mode). Today, we implement **NDT (Normal Distributions Transform)**, the standard algorithm for robust Lidar localization.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Contrast** SLAM (Unknown Map) with Localization (Known Map).
2.  **Explain** the NDT representation: Dividing space into cells and modeling points as Gaussian distributions.
3.  **Derive** the NDT Score Function and its gradient (Newton's Method).
4.  **Implement** NDT Localization in Python to align a Lidar scan to a global map.
5.  **Analyze** the benefits of NDT over ICP (Smooth cost function, no nearest neighbor search).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 22:** ICP and Point Clouds.
-   **Calculus:** Hessian Matrix (Second derivatives).
-   **Probability:** Multivariate Gaussian (Day 8).

### Hardware Requirements
-   **Map Data:** A pre-built point cloud map (e.g., from Day 22).
-   **Scan Data:** A single Lidar scan.

### Software Stack
-   **Python Libraries:** `numpy`, `scipy`, `open3d`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The NDT Representation

ICP matches points to points. This is noisy and requires expensive Nearest Neighbor search.
NDT represents the map as a **Grid of Probability Distributions**.

1.  Divide the map space into fixed-size cells (e.g., 1m x 1m).
2.  For each cell containing points:
    -   Calculate Mean $\mu$.
    -   Calculate Covariance $\Sigma$.
3.  The probability of finding a point at $x$ in cell $k$ is:
    $$ p(x) \propto e^{-\frac{1}{2} (x-\mu_k)^T \Sigma_k^{-1} (x-\mu_k)} $$

**Result:** A continuous, differentiable probability surface (PDF) representing the geometry.

---

### 🔹 Part 2: NDT Matching

Goal: Find the pose $T = [t_x, t_y, \theta]$ that maximizes the likelihood of the current scan points lying on the high-probability regions of the map.

**Score Function:**
$$ f(T) = \sum_{i=1}^{N} p(T(x_i)) $$
Where $x_i$ are the points in the current scan.

**Optimization:**
We use **Newton's Method** to find $T$ that maximizes $f(T)$.
$$ \Delta T = -H^{-1} g $$
-   $g$: Gradient (First derivative of score w.r.t pose).
-   $H$: Hessian (Second derivative).

**Why NDT?**
-   **No Nearest Neighbors:** We just look up which cell the transformed point falls into. $O(1)$.
-   **Smooth:** The Gaussian tails provide a gradient even if the point is slightly off the wall. ICP has a "hard" cost function.

---

## 💻 Implementation: NDT Localization

We will implement a simplified 2D NDT in Python.
-   **Map:** A set of 2D points.
-   **Grid:** 2D cells.
-   **Scan:** A rotated/translated subset of the map.

### 🛠️ Setup
Create `week4_day26` and `ndt_localization.py`.

```bash
mkdir -p ~/ros2_ws/src/week4_day26
cd ~/ros2_ws/src/week4_day26
touch ndt_localization.py
```

### 👨‍💻 Code: 2D NDT Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

class NDTGrid:
    def __init__(self, map_points, cell_size=1.0):
        self.cell_size = cell_size
        self.cells = {} # (ix, iy) -> (mean, cov)
        
        # 1. Bin points into cells
        bins = {}
        for p in map_points:
            ix = int(np.floor(p[0] / cell_size))
            iy = int(np.floor(p[1] / cell_size))
            if (ix, iy) not in bins:
                bins[(ix, iy)] = []
            bins[(ix, iy)].append(p)
            
        # 2. Compute Gaussian for each cell
        for k, points in bins.items():
            if len(points) < 5: continue # Need enough points
            
            pts = np.array(points)
            mean = np.mean(pts, axis=0)
            cov = np.cov(pts.T)
            
            # Avoid singular covariance
            if np.linalg.det(cov) < 1e-5:
                continue
                
            self.cells[k] = (mean, cov, inv(cov))

    def get_score(self, points):
        score = 0
        for p in points:
            ix = int(np.floor(p[0] / self.cell_size))
            iy = int(np.floor(p[1] / self.cell_size))
            
            if (ix, iy) in self.cells:
                mu, cov, inv_cov = self.cells[(ix, iy)]
                d = p - mu
                score += np.exp(-0.5 * d.T @ inv_cov @ d)
        return score

def transform_points(points, T):
    # T = [tx, ty, theta]
    tx, ty, theta = T
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (R @ points.T).T + np.array([tx, ty])

def run_ndt():
    # 1. Create Map (Corridor)
    # Two parallel walls
    wall1 = np.column_stack((np.linspace(0, 20, 100), np.zeros(100)))
    wall2 = np.column_stack((np.linspace(0, 20, 100), np.ones(100) * 5))
    map_points = np.vstack((wall1, wall2))
    
    # Add noise to map
    map_points += np.random.normal(0, 0.05, map_points.shape)
    
    # Build NDT Grid
    ndt = NDTGrid(map_points, cell_size=2.0)
    
    # 2. Create Scan (Robot at x=2, y=1, theta=0.2)
    true_pose = np.array([2.0, 1.0, 0.2])
    
    # Scan sees a local window of the map
    scan_local = map_points.copy() 
    # Inverse transform to put points in robot frame
    # (Simplified: just taking a subset and rotating it)
    # Let's simulate a scan by taking map points, transforming them to robot frame
    
    # R_inv * (P_world - t)
    c, s = np.cos(true_pose[2]), np.sin(true_pose[2])
    R_inv = np.array([[c, s], [-s, c]])
    scan_robot = (R_inv @ (map_points - true_pose[:2]).T).T
    
    # Add noise to scan
    scan_robot += np.random.normal(0, 0.05, scan_robot.shape)
    
    # 3. Optimization Loop (Simple Hill Climbing / Gradient Descent)
    # Initial Guess (Dead Reckoning)
    current_pose = np.array([1.8, 0.8, 0.0]) # Slightly off
    
    history = [current_pose.copy()]
    
    # For simplicity, we use numerical gradient here instead of analytical
    # In production, use analytical Jacobian/Hessian
    
    learning_rate = 0.1
    
    print(f"Initial Score: {ndt.get_score(transform_points(scan_robot, current_pose))}")
    
    for i in range(50):
        # Calculate Gradient numerically
        grad = np.zeros(3)
        eps = 1e-3
        
        base_score = ndt.get_score(transform_points(scan_robot, current_pose))
        
        # d/dx
        pose_dx = current_pose + [eps, 0, 0]
        grad[0] = (ndt.get_score(transform_points(scan_robot, pose_dx)) - base_score) / eps
        
        # d/dy
        pose_dy = current_pose + [0, eps, 0]
        grad[1] = (ndt.get_score(transform_points(scan_robot, pose_dy)) - base_score) / eps
        
        # d/dtheta
        pose_dt = current_pose + [0, 0, eps]
        grad[2] = (ndt.get_score(transform_points(scan_robot, pose_dt)) - base_score) / eps
        
        # Update (Gradient Ascent because we maximize score)
        current_pose += learning_rate * grad
        history.append(current_pose.copy())
        
        if np.linalg.norm(grad) < 1e-4:
            print(f"Converged at step {i}")
            break
            
    print(f"Final Pose: {current_pose}")
    print(f"True Pose:  {true_pose}")
    
    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(map_points[:,0], map_points[:,1], 'k.', alpha=0.2, label='Map')
    
    # Initial Guess
    p_init = transform_points(scan_robot, history[0])
    plt.plot(p_init[:,0], p_init[:,1], 'r.', alpha=0.5, label='Initial Guess')
    
    # Final
    p_final = transform_points(scan_robot, current_pose)
    plt.plot(p_final[:,0], p_final[:,1], 'g.', label='Aligned Scan')
    
    plt.legend()
    plt.title("NDT Localization")
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    run_ndt()
```

---

## 🔬 Lab Exercise: Grid Resolution

### Lab Objectives
1.  Run the script.
2.  **Experiment:** Change `cell_size`.
    -   **Small (0.5m):** Cells might be empty or have singular covariance (flat line). NDT becomes unstable.
    -   **Large (5.0m):** The Gaussian is too blurry. Precision drops.
3.  **Observation:** NDT is robust to initial error. Try setting `current_pose` far away. It should still pull it in (unlike ICP which might get stuck in local minima).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Singular Covariance
**Symptom:** `np.linalg.inv` fails.
**Cause:** All points in a cell lie on a straight line (rank deficient).
**Solution:** Check `det(cov)`. If small, add a small value to the diagonal (regularization) or discard the cell.

#### 2. Local Minima
**Symptom:** Converges to wrong pose.
**Cause:** Grid structure creates discontinuities at cell boundaries.
**Solution:** Use **Multi-Resolution NDT** (coarse grid for rough alignment, fine grid for precision) or overlapping cells.

#### 3. "Jumping"
**Symptom:** Pose oscillates.
**Cause:** Learning rate too high.
**Solution:** Use Newton's Method (Hessian) instead of simple Gradient Ascent. It adjusts step size automatically.

---

## ⚡ Optimization & Best Practices

### 1. PCL Implementation
The Point Cloud Library (PCL) has a highly optimized C++ implementation: `pcl::NormalDistributionsTransform`.
-   It uses a voxel grid data structure.
-   It implements the analytic derivatives (Newton's method).

### 2. GPU NDT
Autoware uses a CUDA-accelerated NDT.
-   Calculating the score and gradient for thousands of points is parallelizable.
-   Essential for 10Hz localization with large maps.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is NDT faster than ICP?
    *   **A:** No Nearest Neighbor search ($O(1)$ lookup vs $O(N \log N)$).
2.  **Q:** What does the Covariance Matrix in NDT represent?
    *   **A:** The shape and orientation of the surface within that cell (e.g., flat wall, corner, blob).
3.  **Q:** How does NDT handle dynamic objects (cars walking by)?
    *   **A:** If they don't match the map's Gaussians, they contribute low score (low probability). They are naturally filtered out (robust).

### Challenge Task
**Task:** 3D NDT.
1.  Use `open3d` to load 3D clouds.
2.  Implement 3D voxel grid.
3.  Use `pcd.get_voxel_center_coordinate` to bin points.
4.  Visualize the 3D ellipsoids (Covariances).

---

## 📚 Further Reading & References
-   [The Normal Distributions Transform (Biber & Strasser)](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.4.3626&rep=rep1&type=pdf) - Original Paper.
-   [Autoware NDT Documentation](https://autowarefoundation.github.io/autoware.universe/main/localization/ndt_scan_matcher/)

---

**Day 26 Complete** | Phase 4: ADAS & Robotics Systems | Week 4: Localization & Mapping (SLAM)
