# Day 11: Factor Graphs (GTSAM)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 2: Advanced SLAM & State Estimation

---

> **📝 Content Creator Instructions:**
> Filters (EKF) are Markovian: they forget the past. Smoothing (Factor Graphs) remembers everything.
> - **Focus:** Mathematical intuition of Least Squares on Graphs, iSAM2, and GTSAM library usage.
> - **Code:** Implementation of Pose Graph Optimization using `gtsam` Python bindings.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Filtering (EKF) and Smoothing (Factor Graphs).
2.  **Formulate** a SLAM problem as a Bipartite Graph of Variables and Factors.
3.  **Explain** how solving the graph is equivalent to Non-Linear Least Squares ($Ax=b$) via Gaussian Elimination.
4.  **Use** the GTSAM (Georgia Tech Smoothing and Mapping) library to solve a Pose Graph.
5.  **Apply** iSAM2 (Incremental Smoothing and Mapping) for real-time updates.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Simulation).

### Software Environment
```bash
pip install gtsam  # Python bindings
```
*Note: GTSAM is the backbone of LIO-SAM, LeGO-LOAM, and modern VIO.*

### Prior Knowledge
- Least Squares.
- Sparse Matrices (Hessian Matrix).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Filtering vs. Smoothing

*   **EKF (Filtering):** Only keeps the current state estimation $\hat{x}_k$ and covariance $P_k$. Once we marginalize out $x_{k-1}$, we can never linearize around it again.
    *   *Issue:* Linearization points are fixed. If the estimate improves later (Loop Closure), we can't correct the past drift.
*   **Factor Graph (Smoothing):** Keeps *all* previous states $X = \{x_1, ..., x_k\}$ and minimizes the error over the entire trajectory.
    *   *Benefit:* Global consistency. Allows re-linearization.

### 🔹 Part 2: The Factor Graph Model

A Factor Graph is a **Bipartite Graph**:
1.  **Variable Nodes ($x_i$):** Robot Poses, Landmarks.
2.  **Factor Nodes ($f_j$):** Measurements/Constraints (e.g., Odometry between $x_i, x_{i+1}$, GPS at $x_i$).

**Objective Function:**
We want to find $X^*$ that maximizes the joint probability:
$$ X^* = \arg \max_X \prod_i P(z_i | X) $$
Assuming Gaussian noise, this is equivalent to minimizing the sums of squared errors (Mahalanobis distance):
$$ X^* = \arg \min_X \sum_i || h_i(X) - z_i ||_{\Sigma_i}^2 $$

### 🔹 Part 3: Solving the Graph

To minimize the Non-Linear Least Squares error:
1.  **Linearize** measurement functions $h(X)$ around initial guess (Jacobians).
2.  Build the system $A\Delta x = b$ (Normal Equations $A^T A \Delta x = A^T b$).
    *   $A^T A$ is the **Hessian Information Matrix**. It is Sparse!
    *   Only connected variables have non-zero entries in Hessian.
3.  **Solve** for $\Delta x$ (using Cholesky Decomposition or QR).
4.  **Update** $X \leftarrow X + \Delta x$.
5.  **Iterate** until convergence (Gauss-Newton or Levenberg-Marquardt).

**iSAM2 (Incremental):**
Instead of re-solving the *whole* matrix every step, iSAM2 uses the **Bayes Tree** data structure to only update the parts of the matrix affected by the new measurement.
*   *Result:* Real-time performance even with thousands of nodes.

---

## 💻 Implementation: Pose Graph Optimization

We will simulate a robot driving in a square. Odometry is noisy. We act as "God" and provide a Loop Closure constraint when it returns to start.

### 🛠️ Project Structure
```text
day11_gtsam/
├── src/
│   ├── pose_graph.py
│   └── visualizer.py
└── run_optimization.py
```

### 👨‍💻 Code Implementation (`src/pose_graph.py`)

```python
import gtsam
import numpy as np
import matplotlib.pyplot as plt

def run_slam():
    # 1. Create a Factor Graph container
    graph = gtsam.NonlinearFactorGraph()
    
    # 2. Add a Prior Factor to the first pose (Fix the world frame)
    # "I am at (0,0,0) with high confidence"
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1])) # x, y, theta noise
    prior_factor = gtsam.PriorFactorPose2(1, gtsam.Pose2(0, 0, 0), prior_noise)
    graph.add(prior_factor)
    
    # 3. Simulate Odometry: Robot drives forward 1 unit
    # Odometry Meas: (dx=1, dy=0, dtheta=0)
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    odometry_factor = gtsam.BetweenFactorPose2(1, 2, gtsam.Pose2(2.0, 0, 0), odometry_noise)
    graph.add(odometry_factor)
    
    odometry_factor = gtsam.BetweenFactorPose2(2, 3, gtsam.Pose2(2.0, 0, np.pi/2), odometry_noise) # Turn
    graph.add(odometry_factor)
    
    # ... Add more steps ...
    
    # 4. Add Loop Closure (Constraint between Pose 5 and Pose 2)
    # Scanner says: Pose 5 is 0.0m away from Pose 2
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.01]))
    loop_factor = gtsam.BetweenFactorPose2(5, 2, gtsam.Pose2(0, 0, 0), loop_noise)
    graph.add(loop_factor)
    
    # 5. Initialize Estimates (Bad guess: Dead Reckoning)
    initial_estimate = gtsam.Values()
    initial_estimate.insert(1, gtsam.Pose2(0.5, 0.0, 0.2)) # Noisy prior
    initial_estimate.insert(2, gtsam.Pose2(2.3, 0.1, -0.2))
    initial_estimate.insert(3, gtsam.Pose2(4.1, 0.1, 1.0))
    initial_estimate.insert(5, gtsam.Pose2(1.0, 1.0, 0.0))
    
    # 6. Optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    
    # 7. Print Results
    result.print("Final Result:\n")
    
    # Get Covariance (Marginals)
    marginals = gtsam.Marginals(graph, result)
    print("Covariance at Pose 5:\n", marginals.marginalCovariance(5))

    return result

if __name__ == "__main__":
    run_slam()
```

---

## 🔬 Lab Exercise: The "G2O" File

Standard SLAM datasets come in `.g2o` format (text file listing nodes and edges).

### 1. Lab Objectives
- Parse a 2D City Dataset (Intel Research Lab dataset).
- Load into GTSAM.
- Optimize and Visualize "Before vs After".

### 2. Step-by-Step Guide

#### Phase A: Python Script

```python
import gtsam
from gtsam.utils import plot

# Load g2o
graph, initial = gtsam.readG2o('data/input_INTEL.g2o', is3D=False)

# Add Prior on first node (anchoring)
priorModel = gtsam.noiseModel.Diagonal.Variances(gtsam.Point3(1e-6, 1e-6, 1e-8))
graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(), priorModel))

# Optimize
params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
result = optimizer.optimize()

# Plot
plot.plot_trajectory(1, result, scale=1)
plt.show()
```

### 3. Expected Output
- **Before Optimization:** The map looks like a broken, twisted corridor. Loop closures are visible as red lines connecting far-away points.
- **After Optimization:** The corridors straighten out. The loop ends perfectly meet.

---

## 🚀 Project: GPS-Informed Graph SLAM

**Scenario:** You have Odometry (High Rate) and sporadic GPS readings (Low Rate).
**Goal:** Build a Graph where:
*   Nodes: Robot Poses at 1Hz.
*   Factors Type 1: Odometry between $x_t, x_{t+1}$.
*   Factors Type 2: GPS Unary Factor on $x_t$ (if available).

### 1. Implementation Logic
*   If we lose GPS for 10 nodes (Tunnel), the graph relies on Odometry factors (Graph chain stretches).
*   When GPS returns at Node 11, the "GPS Factor" pulls Node 11 to the correct location.
*   **The Magic:** Because Factor Graphs optimize globally, pulling Node 11 **back-propagates** the correction to Nodes 1-10 inside the tunnel! (Unlike EKF which cannot fix the past).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Underconstrained System"
*   **Symptom:** Optimization fails or explodes.
*   **Cause:** The graph is floating in space.
*   **Fix:** **Always** add a `PriorFactor` to the first node to anchor the map to a global coordinate frame (0,0,0).

#### 2. "Bad Initial Guess"
*   **Symptom:** Optimization converges to a local minimum (Twisted map).
*   **Cause:** Non-linear optimization needs a decent initialization.
*   **Fix:** Use Odometry Integration to generate the `InitialEstimate`. Do not initialize everything at (0,0,0).

---

## ⚡ Optimization: Robust Kernels

Outliers (Bad loop closures) ruin Least Squares (Quadratic penalty).
**Huber / Cauchy Kernel:** reduces the penalty for large errors (Linear instead of Quadratic).
```python
# In GTSAM
noise = gtsam.noiseModel.Robust(
    gtsam.noiseModel.mEstimator.Huber(1.345),
    gtsam.noiseModel.Diagonal.Sigmas(...)
)
```
*   **Task:** Add large noise to one Loop Closure in the lab above and see the distortion. Then apply Huber Kernel and watch it ignore the bad loop.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is the Hessian Matrix sparse?
    *   **A:** In SLAM, a robot pose $x_i$ is only connected to $x_{i-1}$, $x_{i+1}$, and maybe a few landmarks. It is not connected to *every* other pose. This sparsity allows solving systems with 10k+ variables.
2.  **Q:** What is the difference between Batch and Incremental optimization?
    *   **A:** Batch (Levenberg-Marquardt) solves the whole graph. Incremental (iSAM2) updates the factorization, only touching affected cliques in the Bayes Tree.
3.  **Q:** How does a Factor differ from a Constraint?
    *   **A:** A Factor is a "Soft" constraint (probabilistic penalty). A Hard constraint (Equality) is rarely used in noisy SLAM.

### Challenge Task
> **Task:** Landmark SLAM.
> 1. Add Point2 variables (Landmarks) to the graph.
> 2. Add Measurement Factors (Bearing + Range) from Pose nodes to Landmark nodes.
> 3. Optimize both Robot Trajectory and Map simultaneously.

---

## 📚 Further Reading
- **Factor Graphs for Robot Perception:** Frank Dellaert (Foundational Booklet).
- **iSAM2:** Kaess et al. (IJRR 2012).
- **GTSAM Library:** borglab.org

---

**Day 11 Complete**
