# Day 131: Graph SLAM (Pose Graph Optimization)
## Phase 4: ADAS & Robotics Systems | Week 19: Localization & SLAM

---

> **📝 Day 131 Focus:**
> Odometry (Visual or Lidar) always drifts. If you drive around the block and return to the start, your map might say you are 5 meters away. **Graph SLAM** fixes this by detecting the "Loop Closure" and bending the whole trajectory to make ends meet.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the SLAM Frontend (Odometry/Loop Detection) and Backend (Optimization).
2.  **Construct** a Pose Graph (Nodes = Poses, Edges = Constraints).
3.  **Explain** the Loop Closure constraint.
4.  **Implement** a simple 2D Pose Graph Optimization using `g2o` (or Python equivalent).
5.  **Visualize** the "Before" (Drift) and "After" (Optimized) maps.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Least Squares:** Minimizing error.
-   **Day 130:** Odometry.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `numpy`, `scipy.optimize` (for custom solver).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Graph

-   **Nodes ($x_i$):** Robot poses at time $i$ ($x, y, \theta$).
-   **Edges ($z_{ij}$):** Measurements connecting nodes.
    -   **Odometry Edge ($z_{i, i+1}$):** "I moved 1m forward from $i$ to $i+1$."
    -   **Loop Closure Edge ($z_{i, j}$):** "I see the same landmark at $j$ that I saw at $i$."

### 🔹 Part 2: The Error Function

For each edge, the error is the difference between the *expected* relative pose (from nodes) and the *measured* relative pose (from edge).
$$ e_{ij} = z_{ij}^{-1} (x_i^{-1} x_j) $$
We want to find $x^*$ that minimizes the sum of squared errors (weighted by information matrix $\Omega$):
$$ F(x) = \sum e_{ij}^T \Omega_{ij} e_{ij} $$

### 🔹 Part 3: Optimization

This is a **Non-Linear Least Squares** problem.
We solve it using iterative methods:
-   **Gauss-Newton** or **Levenberg-Marquardt**.
-   Libraries: **g2o**, **Ceres**, **GTSAM**.

---

## 💻 Implementation: Custom PGO Solver

**Scenario:**
-   Robot moves in a square ($10 \times 10$).
-   Odometry is noisy. The square doesn't close.
-   Loop Closure detected at the end (Start $\approx$ End).
-   Task: Optimize the path.

### 🛠️ Setup
Create `week19_day131` and `pose_graph.py`.

```bash
mkdir -p ~/ros2_ws/src/week19_day131
cd ~/ros2_ws/src/week19_day131
touch pose_graph.py
```

### 👨‍💻 Code: 2D Graph Optimization

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# --- Helper Functions ---
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_transform(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

def get_pose_from_transform(T):
    x = T[0, 2]
    y = T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return np.array([x, y, theta])

# --- The Problem ---
class PoseGraph:
    def __init__(self):
        self.nodes = [] # List of [x, y, theta] (Initial Guess)
        self.edges = [] # List of (i, j, dx, dy, dtheta)

    def add_node(self, pose):
        self.nodes.append(pose)

    def add_edge(self, i, j, measurement):
        self.edges.append((i, j, measurement))

    def error_function(self, x_flat):
        # Reshape x_flat back to (N, 3)
        poses = x_flat.reshape(-1, 3)
        residuals = []
        
        for (i, j, z) in self.edges:
            # Current estimates
            xi = poses[i]
            xj = poses[j]
            
            # Expected relative transform from xi to xj
            # T_i_j = T_i^-1 * T_j
            Ti = get_transform(*xi)
            Tj = get_transform(*xj)
            T_rel_est = np.linalg.inv(Ti) @ Tj
            z_est = get_pose_from_transform(T_rel_est)
            
            # Measurement
            z_meas = np.array(z)
            
            # Error
            err_x = z_est[0] - z_meas[0]
            err_y = z_est[1] - z_meas[1]
            err_theta = normalize_angle(z_est[2] - z_meas[2])
            
            residuals.extend([err_x, err_y, err_theta])
            
        # Anchor the first node (Constraint to 0,0,0)
        # Otherwise the whole graph floats
        residuals.extend(poses[0] * 100.0) 
        
        return np.array(residuals)

    def optimize(self):
        x0 = np.array(self.nodes).flatten()
        print("Optimizing...")
        res = least_squares(self.error_function, x0, verbose=1)
        self.nodes = res.x.reshape(-1, 3)

def main():
    pg = PoseGraph()
    
    # 1. Simulate Ground Truth (Square 10x10)
    # 4 corners: (0,0), (10,0), (10,10), (0,10), (0,0)
    # 4 steps per side
    steps = 16
    side_len = 10.0
    step_len = side_len / 4.0
    
    # 2. Generate Noisy Odometry
    curr_pose = np.array([0.0, 0.0, 0.0])
    pg.add_node(curr_pose)
    
    true_poses = [curr_pose.copy()]
    
    for i in range(steps):
        # Determine move
        dx, dy, dtheta = 0, 0, 0
        if i < 4: dx = step_len # East
        elif i < 8: dy = step_len; dtheta = np.pi/2 if i==4 else 0 # North
        elif i < 12: dx = -step_len; dtheta = np.pi/2 if i==8 else 0 # West
        else: dy = -step_len; dtheta = np.pi/2 if i==12 else 0 # South
        
        # Add Noise
        noise = np.random.normal(0, 0.2, 3)
        meas = np.array([dx, dy, dtheta]) + noise
        
        # Update Estimate (Dead Reckoning)
        # T_new = T_curr * T_meas
        T_curr = get_transform(*curr_pose)
        T_meas = get_transform(*meas)
        T_new = T_curr @ T_meas
        curr_pose = get_pose_from_transform(T_new)
        
        pg.add_node(curr_pose)
        pg.add_edge(i, i+1, meas)
        
    # 3. Add Loop Closure
    # The last node (16) should be same as first node (0)
    # Measurement is Identity (0,0,0) with high confidence
    pg.add_edge(steps, 0, [0, 0, 0])
    
    # Plot Before
    initial_nodes = np.array(pg.nodes)
    
    # Optimize
    pg.optimize()
    
    final_nodes = np.array(pg.nodes)
    
    # Plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(initial_nodes[:, 0], initial_nodes[:, 1], 'r-o', label='Noisy Odom')
    plt.title("Before Optimization")
    plt.grid()
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.plot(final_nodes[:, 0], final_nodes[:, 1], 'b-o', label='Optimized')
    plt.title("After Optimization")
    plt.grid()
    plt.axis('equal')
    
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Snap

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   **Before:** The path looks like an open spiral. The end point is far from the start.
    -   **After:** The path snaps into a closed square. The error is distributed backwards along the path.
3.  **Experiment:**
    -   Remove the Loop Closure edge (`pg.add_edge(steps, 0, ...)`).
    -   **Result:** The optimization does nothing (or just shifts the whole graph). Without the loop closure constraint, the odometry is the "best guess".

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Bad Loop Closures
**Symptom:** The map becomes twisted and broken.
**Cause:** A False Positive loop closure (e.g., matching two similar looking trees that are actually 1km apart).
**Solution:** Robust Back-End (Switchable Constraints or Max-Mixture models) that can reject outliers during optimization.

#### 2. Information Matrix
**Symptom:** Loop closure is ignored.
**Cause:** The solver thinks Odometry is very accurate (High weight) and Loop Closure is noisy (Low weight).
**Solution:** Tune the weights ($\Omega$). Loop closures usually have high confidence (if correct).

---

## ⚡ Optimization & Best Practices

### 1. Sparse Solvers
The Jacobian matrix is huge ($3N \times 3N$) but sparse (mostly zeros).
-   Use solvers that exploit sparsity (Cholesky decomposition of sparse matrices).
-   **g2o** and **GTSAM** are optimized for this.

### 2. Incremental SLAM (iSAM)
Don't re-optimize the whole graph every step.
-   **iSAM2 (GTSAM):** Only updates the part of the graph affected by the new measurement (Bayes Tree).
-   Allows real-time SLAM.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between Filter SLAM (EKF) and Graph SLAM?
    *   **A:** EKF marginalizes out old poses (keeps only current state). Graph SLAM keeps all history. Graph SLAM is more accurate but grows in size.
2.  **Q:** What is a "Residual"?
    *   **A:** The difference between the measurement and the prediction given the current state estimate. $r = z - h(x)$.
3.  **Q:** Why do we anchor the first node?
    *   **A:** SLAM is relative. The graph has 3 degrees of freedom (Gauge Freedom). Anchoring $x_0 = (0,0,0)$ fixes the coordinate system.

### Challenge Task
**Task:** 3D Graph.
1.  Extend the code to 3D ($x, y, z, roll, pitch, yaw$).
2.  Use $4 \times 4$ transformation matrices.
3.  This is the basis of 6-DOF SLAM used in drones and cars.

---

## 📚 Further Reading & References
-   [A Tutorial on Graph-Based SLAM (Grisetti)](http://www2.informatik.uni-freiburg.de/~stachnis/pdf/grisetti10titsmag.pdf)
-   [GTSAM Library](https://gtsam.org/)

---

**Day 131 Complete** | Phase 4: ADAS & Robotics Systems | Week 19: Localization & SLAM
