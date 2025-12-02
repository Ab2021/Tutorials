# Day 13: SLAM Fundamentals (Simultaneous Localization and Mapping)
## Phase 4: ADAS & Robotics Systems | Week 2: Sensor Fusion & State Estimation

---

> **📝 Day 13 Focus:**
> Localization assumes a map exists. Mapping assumes the robot's pose is known. What if we have neither? This is the **SLAM (Simultaneous Localization and Mapping)** problem, considered the "Holy Grail" of mobile robotics. Today, we explore how to solve this chicken-and-egg problem using Graph Optimization.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Formulate** the SLAM problem as a probabilistic inference task.
2.  **Differentiate** between Filter-based SLAM (FastSLAM) and Graph-based SLAM (Pose Graph).
3.  **Explain** the concept of Loop Closure and why it is critical for correcting drift.
4.  **Implement** a simple 2D GraphSLAM backend using Non-Linear Least Squares optimization.
5.  **Analyze** the sparsity of the SLAM information matrix and its impact on performance.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 9:** Kalman Filter (Least Squares intuition).
-   **Optimization:** Gradient Descent / Newton-Gauss.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows/Mac with Python).

### Software Stack
-   **Python Libraries:** `numpy`, `scipy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The SLAM Problem

We want to estimate both the Trajectory ($X_{1:t}$) and the Map ($M$) given Controls ($U_{1:t}$) and Measurements ($Z_{1:t}$).

$$ p(X_{1:t}, M | Z_{1:t}, U_{1:t}) $$

#### 1.1 The Chicken and Egg
-   If we knew the Map, we could use MCL (Day 12) to find the Pose.
-   If we knew the Pose, we could simply paste sensor readings to build the Map.
-   **SLAM** estimates both jointly.

#### 1.2 Full SLAM vs. Online SLAM
-   **Full SLAM:** Estimate the *entire* path $x_1, \dots, x_t$. (GraphSLAM).
-   **Online SLAM:** Estimate only the *current* pose $x_t$. (EKF-SLAM, FastSLAM).

---

### 🔹 Part 2: GraphSLAM (Pose Graph Optimization)

Modern SLAM systems (Cartographer, ORB-SLAM) use GraphSLAM.

#### 2.1 The Graph Structure
-   **Nodes:** Robot Poses ($x_1, x_2, \dots$) and Landmarks ($m_1, m_2, \dots$).
-   **Edges:** Constraints.
    -   **Odometry Edge:** Constraint between $x_{t-1}$ and $x_t$ (from wheel encoders).
    -   **Measurement Edge:** Constraint between $x_t$ and $m_j$ (from Lidar/Camera).

#### 2.2 The Optimization
Each edge represents an error.
$$ e_{ij} = z_{ij} - h(x_i, x_j) $$

We want to find the configuration of nodes ($X, M$) that minimizes the total squared error (Mahalanobis distance):
$$ X^*, M^* = \text{argmin} \sum_{ij} e_{ij}^T \Omega_{ij} e_{ij} $$
Where $\Omega_{ij}$ is the Information Matrix (Inverse Covariance).

This is a **Non-Linear Least Squares** problem. We solve it using **Levenberg-Marquardt** or **Gauss-Newton**.

#### 2.3 Loop Closure
This is the magic moment.
1.  Robot drives in a large circle. Odometry drifts, so the end point doesn't match the start point.
2.  Robot recognizes the start point ("I've been here before!").
3.  A new constraint (edge) is added between $x_{current}$ and $x_{start}$.
4.  The optimization "snaps" the graph shut, distributing the error along the entire trajectory.

---

### 🔹 Part 3: FastSLAM (Particle Filter SLAM)

FastSLAM decomposes the problem using Rao-Blackwellization.
$$ p(X, M | Z) = p(M | X, Z) \cdot p(X | Z) $$

-   **Particle Filter:** Estimates the Trajectory ($X$).
-   **EKF:** Each particle maintains its *own* map ($M$).
    -   If we have $N$ particles and $K$ landmarks, we have $N \times K$ Kalman Filters!

**Pros:** Handles non-linearities well.
**Cons:** Computationally expensive for large maps.

---

## 💻 Implementation: 2D GraphSLAM (Backend)

We will implement a simple Pose Graph Optimizer.
-   **Scenario:** Robot moves in 1D (for simplicity of math, but code is generalizable).
-   **Constraints:** Odometry and Landmarks.

### 🛠️ Setup
Create `week2_day13` and `graph_slam.py`.

```bash
mkdir -p ~/ros2_ws/src/week2_day13
cd ~/ros2_ws/src/week2_day13
touch graph_slam.py
```

### 👨‍💻 Code: GraphSLAM with Scipy

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

class GraphSLAM:
    def __init__(self):
        # State Vector: [x0, x1, ..., xN, m0, m1, ...]
        # We will construct this dynamically
        pass

    def solve(self, measurements, odometry, initial_guess):
        """
        measurements: list of [pose_idx, landmark_idx, distance]
        odometry: list of [pose_idx_from, pose_idx_to, distance]
        initial_guess: numpy array of all states (poses + landmarks)
        """
        
        def error_function(state):
            residuals = []
            
            # 1. Odometry Constraints
            for odom in odometry:
                idx_from, idx_to, dist_obs = odom
                
                # State values
                x_from = state[idx_from]
                x_to = state[idx_to]
                
                # Expected distance
                dist_pred = x_to - x_from
                
                # Residual
                residuals.append(dist_pred - dist_obs)
                
            # 2. Measurement Constraints
            for meas in measurements:
                pose_idx, lm_idx, dist_obs = meas
                
                x_robot = state[pose_idx]
                x_lm = state[lm_idx]
                
                dist_pred = x_lm - x_robot
                
                residuals.append(dist_pred - dist_obs)
                
            # 3. Anchor Constraint (Fix first pose to 0)
            residuals.append(state[0] - 0.0)
            
            return np.array(residuals)

        # Optimize
        result = least_squares(error_function, initial_guess, verbose=2)
        return result.x

def run_simulation():
    # --- True World ---
    # Robot moves: 0 -> 5 -> 10 -> 15
    true_poses = [0, 5, 10, 15]
    # Landmark at 20
    true_landmarks = [20]
    
    # --- Noisy Observations ---
    # Odometry (Robot thinks it moved 4.5 instead of 5)
    odom_constraints = [
        [0, 1, 4.5], # 0->1
        [1, 2, 4.5], # 1->2
        [2, 3, 4.5]  # 2->3
    ]
    
    # Measurements (Robot sees landmark)
    # At x=0, sees LM at 20 (dist 20)
    # At x=5, sees LM at 20 (dist 15)
    # ...
    meas_constraints = [
        [0, 4, 19.5], # Pose 0 sees LM(idx 4) at dist 19.5 (True 20)
        [1, 4, 14.5], # Pose 1 sees LM(idx 4) at dist 14.5 (True 15)
        [2, 4, 9.5],  # Pose 2 sees LM(idx 4) at dist 9.5  (True 10)
        [3, 4, 4.5]   # Pose 3 sees LM(idx 4) at dist 4.5  (True 5)
    ]
    
    # Note: State vector indices:
    # 0: Pose 0
    # 1: Pose 1
    # 2: Pose 2
    # 3: Pose 3
    # 4: Landmark 0
    
    # --- Initial Guess (Dead Reckoning) ---
    # Based on bad odometry
    x0 = [0, 4.5, 9.0, 13.5, 0] # LM guess is 0 (unknown)
    # Let's initialize LM based on first measurement: 0 + 19.5 = 19.5
    x0[4] = 19.5
    
    print(f"Initial Guess: {x0}")
    
    # --- Solve ---
    slam = GraphSLAM()
    optimized_state = slam.solve(meas_constraints, odom_constraints, np.array(x0))
    
    print(f"Optimized State: {optimized_state}")
    print(f"True State:      {true_poses + true_landmarks}")
    
    # --- Visualization ---
    plt.figure(figsize=(10, 2))
    plt.plot(true_poses, np.zeros_like(true_poses), 'go', label='True Poses')
    plt.plot(true_landmarks, [0], 'g*', markersize=15, label='True LM')
    
    plt.plot(x0[:4], np.zeros(4)-0.1, 'rx', label='Initial Guess')
    plt.plot(x0[4], -0.1, 'r*', label='Initial LM')
    
    plt.plot(optimized_state[:4], np.zeros(4)+0.1, 'b.', label='Optimized')
    plt.plot(optimized_state[4], 0.1, 'b*', label='Optimized LM')
    
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.title("1D GraphSLAM")
    plt.show()

if __name__ == "__main__":
    run_simulation()
```

---

## 🔬 Lab Exercise: Loop Closure Effect

### Lab Objectives
1.  Modify the simulation to be circular (1D cyclic world).
2.  Robot moves $0 \to 5 \to 10 \to 15 \to 0$ (Back to start).
3.  Add a "Loop Closure" constraint:
    -   Odometry says $15 \to 20$ (Drift).
    -   Loop Closure says $15 \to 0$ (Distance 5, but connects to Pose 0).
4.  **Observe:** Without the loop closure, the map is a line $0..20$. With it, the optimizer bends the line into a circle (or overlaps 20 with 0).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Singular Matrix (Ill-posed problem)
**Symptom:** Optimization fails or returns garbage.
**Cause:** The system is floating. You defined relative constraints (A to B), but didn't anchor anything to the world.
**Solution:** Fix the first pose ($x_0 = 0$) by adding a constraint with very high weight (or removing it from optimization variables).

#### 2. Local Minima
**Symptom:** Graph looks tangled.
**Cause:** Initial guess was too far off. Non-linear optimization converged to a bad solution.
**Solution:** Good initialization is key. Usually, we initialize using Odometry (Dead Reckoning).

#### 3. Outliers
**Symptom:** One bad measurement ruins the whole map.
**Cause:** Least Squares is sensitive to outliers (quadratic penalty).
**Solution:** Use **Robust Kernels** (Huber, Cauchy) instead of squared error. They penalize large errors linearly, not quadratically.

---

## ⚡ Optimization & Best Practices

### 1. Sparsity
The Information Matrix (Hessian) is sparse.
-   Pose $i$ is only connected to Pose $i-1$, $i+1$, and visible landmarks.
-   We use sparse linear algebra solvers (Cholesky factorization of sparse matrices) to solve this efficiently (e.g., `g2o`, `ceres-solver`).

### 2. Marginalization
To keep the graph size bounded, we can remove old nodes.
-   However, simply deleting a node removes information.
-   We must **marginalize** it, which adds dense connections between its neighbors (filling in the sparsity pattern).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the "Graph" in GraphSLAM?
    *   **A:** A set of Nodes (Poses/Landmarks) and Edges (Constraints/Measurements).
2.  **Q:** Why is Loop Closure important?
    *   **A:** It eliminates accumulated drift by connecting the current pose to a past pose, allowing the optimizer to correct the entire trajectory.
3.  **Q:** What is the difference between Odometry and SLAM?
    *   **A:** Odometry integrates motion (errors accumulate). SLAM uses landmarks/constraints to correct errors globally.

### Challenge Task
**Task:** 2D Pose Graph.
1.  Implement constraints for 2D ($x, y, \theta$).
2.  Error function involves rotation matrices.
3.  Simulate a robot driving in a square.

---

## 📚 Further Reading & References
-   [A Tutorial on Graph-Based SLAM (Grisetti et al.)](http://www2.informatik.uni-freiburg.de/~stachnis/pdf/grisetti10titsmag.pdf)
-   [Ceres Solver](http://ceres-solver.org/) - Google's non-linear least squares solver (used in Cartographer).
-   [g2o](https://github.com/RainerKuemmerle/g2o) - General Graph Optimization library.

---

**Day 13 Complete** | Phase 4: ADAS & Robotics Systems | Week 2: Sensor Fusion & State Estimation
