# Day 16: Optimization-Based Planning (TEB, MPPI)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 3: Advanced Navigation & Planning

---

> **📝 Content Creator Instructions:**
> RRT finds a path, but it's not smooth or dynamically feasible. To drive *fast*, we need optimization.
> - **Focus:** TEB (Timed Elastic Band) and MPPI (Model Predictive Path Integral).
> - **Code:** Implementation of a basic MPPI Controller for a simulated racecar.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Formulate** Motion Planning as an Optimization Problem (Minimize Time/Energy subject to Constraints).
2.  **Explain** how TEB (Timed Elastic Band) deforms a Global Path to avoid obstacles dynamically.
3.  **Implement** MPPI (Model Predictive Path Integral) Control to handle complex non-linear dynamics.
4.  **Tune** Cost Functions for aggressive vs. safe driving behaviors.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU (Recommended for MPPI).

### Software Environment
```bash
sudo apt install ros-humble-teb-local-planner
pip install torch  # MPPI uses GPU tensors
```

### Prior Knowledge
- Gradient Descent.
- Robot Dynamics ($F=ma$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Trajectory Optimization

Instead of searching for a path (A*), we assume an initial path exists and **Optimize** it.
$$ J(u) = \sum_{t=0}^T C_{state}(x_t) + C_{control}(u_t) + C_{obstacle}(x_t) $$

#### 1.1 TEB (Timed Elastic Band)
*   **Elastic Band:** Connects configurations like a rubber band.
*   **Forces:**
    *   **External:** Obstacles push the band away.
    *   **Internal:** Rubber tension pulls the band tight (shortest path).
*   **Timed:** Adds time intervals $\Delta T_i$ between poses as optimization variables. This allows optimizing for *Velocity* and *Acceleration* limits.
*   **Solver:** g2o (Hyper-Graph Optimization).

### 🔹 Part 2: Model Predictive Control (MPC)

*   **Plan:** Solve optimization for horizon $T$ (e.g., next 2 seconds).
*   **Act:** Execute only the first step $u_0$.
*   **Repeat:** Recalculate everything at next timestep (Receding Horizon).

#### 2.1 MPPI (Model Predictive Path Integral)
A sampling-based MPC variant that works for **Non-Convex, Non-Differentiable** cost functions.
1.  **Simulate:** Roll out 1000s of parallel trajectories using random noise in control input.
    $$ u_t^k = u_{nominal} + \delta \epsilon $$
2.  **Score:** Calculate cost $S_k$ for each trajectory.
3.  **Average:** The optimal control is the weighted average of the random samples. Low-cost samples get higher weights.
    $$ u^* = \frac{\sum e^{-S_k / \lambda} u^k}{\sum e^{-S_k / \lambda}} $$
*   *Pros:* Trivially parallelizable on GPU. Handles arbitrary cost maps (even Neural Networks).

---

## 💻 Implementation: MPPI from Scratch

We will implement MPPI in PyTorch for a simple Unicycle Model.

### 🛠️ Project Structure
```text
day16_mppi/
├── src/
│   ├── dynamics.py
│   ├── cost_function.py
│   └── mppi_solver.py
└── run_control.py
```

### 👨‍💻 Code Implementation (`src/mppi_solver.py`)

```python
import torch
import time

class MPPIController:
    def __init__(self, dynamics, cost_fn, horizon=20, num_samples=1000, noise_std=0.5):
        self.d = dynamics
        self.c = cost_fn
        self.H = horizon
        self.K = num_samples
        self.sigma = noise_std
        self.lambda_ = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Nominal Control Sequence (initially zeros)
        self.U = torch.zeros((self.H, 2)).to(self.device) # [v, w]
        
    def compute_control(self, state):
        # 1. Generate Noise
        # Shape: [K, H, 2]
        noise = torch.randn((self.K, self.H, 2), device=self.device) * self.sigma
        
        # 2. Perturb Controls
        # u_samples[k, t] = U[t] + noise[k, t]
        U_expanded = self.U.unsqueeze(0).repeat(self.K, 1, 1)
        u_samples = U_expanded + noise
        
        # Clamp controls (e.g., max velocity)
        u_samples[:, :, 0] = torch.clamp(u_samples[:, :, 0], 0, 2.0)
        u_samples[:, :, 1] = torch.clamp(u_samples[:, :, 1], -1.0, 1.0)
        
        # 3. Rollout Dynamics
        states = self.d.rollout(state, u_samples)
        
        # 4. Compute Costs
        costs = self.c.compute(states, u_samples) # Shape: [K]
        
        # 5. Model Predictive Path Integral (Weighted Average)
        weights = torch.exp(-costs / self.lambda_)
        weights = weights / torch.sum(weights) # Normalize
        
        # Weighted difference
        # sum (weights * noise)
        weighted_noise = torch.sum(weights.view(-1, 1, 1) * noise, dim=0)
        
        # Update Nominal Control
        self.U = self.U + weighted_noise
        
        # Return first control action
        action = self.U[0].clone()
        
        # Shift controls (Receding Horizon)
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = 0 # Append zero at end
        
        return action
```

### 👨‍💻 Helper: Dynamics (`src/dynamics.py`)

```python
class UnicycleDynamics:
    def __init__(self, dt=0.1):
        self.dt = dt
        
    def rollout(self, start_state, controls):
        """
        start_state: [x, y, theta]
        controls: [K, H, 2]
        """
        K, H, _ = controls.shape
        states = torch.zeros((K, H+1, 3), device=controls.device)
        states[:, 0] = torch.tensor(start_state, device=controls.device)
        
        for t in range(H):
            curr_x = states[:, t, 0]
            curr_y = states[:, t, 1]
            curr_th = states[:, t, 2]
            
            v = controls[:, t, 0]
            w = controls[:, t, 1]
            
            next_x = curr_x + v * torch.cos(curr_th) * self.dt
            next_y = curr_y + v * torch.sin(curr_th) * self.dt
            next_th = curr_th + w * self.dt
            
            states[:, t+1, 0] = next_x
            states[:, t+1, 1] = next_y
            states[:, t+1, 2] = next_th
            
        return states
```

---

## 🔬 Lab Exercise: The Obstacle Course

### 1. Lab Objectives
- Set up a Cost Function that penalizes: dist to goal, collision with circles, and high velocity sideslip.
- Run the MPPI controller.
- **Visualize:** The "Cloud" of trajectories. Watch how the cloud "bends" around obstacles like a flowing river.

### 2. Cost Function Setup (`src/cost_function.py`)

```python
class CostFunction:
    def __init__(self, goal, obstacles):
        self.goal = torch.tensor(goal).cuda()
        self.obs = torch.tensor(obstacles).cuda() # [N, 3] -> (x, y, radius)
        
    def compute(self, states, controls):
        # States: [K, H+1, 3]
        
        # 1. Dist to Goal Cost (Terminal State Only or Sum)
        diff = states[:, :, :2] - self.goal
        dist_sq = torch.sum(diff**2, dim=2)
        goal_cost = torch.sum(dist_sq, dim=1)
        
        # 2. Collision Cost
        # Dist from robot to all obstacles
        # Minimal dist
        coll_cost = torch.zeros_like(goal_cost)
        
        for ob in self.obs:
           ob_pos = ob[:2]
           ob_r = ob[2]
           dvec = states[:, :, :2] - ob_pos
           d = torch.norm(dvec, dim=2)
           # Smooth penalty
           penalty = 1000 * torch.exp(-0.5 * (d - ob_r)**2 / 0.1) # Gaussian spike at obstacle
           coll_cost += torch.sum(penalty, dim=1)
           
        return goal_cost + coll_cost
```

---

## 🚀 Project: "TEB in ROS 2"

**Goal:** Configure TEB Local Planner for a physical differential drive robot (or TurtleBot3 Sim).
1.  **Install:** `sudo apt install ros-humble-teb-local-planner`.
2.  **Config:** `nav2_params.yaml`.
3.  **Tuning:**
    *   `max_vel_x`: 0.26
    *   `acc_lim_x`: 2.5
    *   `min_turning_radius`: 0.0 (Diff Drive!)
    *   `min_obstacle_dist`: 0.2

*Observation:* Command a goal behind an obstacle.
*   **A* / DWA:** Might get stuck or take a wide turn.
*   **TEB:** Will find a tight, smooth S-curve around the obstacle (Parallel Parking maneuver).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Robot Oscillates at Goal"
*   **Symptom:** Robot reaches goal, then wiggles left/right endlessly.
*   **Cause:** Goal Tolerance (`yaw_goal_tolerance`) is too small for the robot's control authority / Odometry noise.
*   **Fix:** Increase tolerance (e.g., 0.1 rad). Or add `xy_goal_tolerance`.

#### 2. "MPPI Ignores Obstacles"
*   **Cause:** Lambda ($\lambda$) parameter is too large (Temperature). Result is a simple average of noise, ignoring the cost weights.
*   **Fix:** Lower $\lambda$ to make the selection "Greedier" (Focus more on low-cost paths).

---

## ⚡ Optimization: GPU Acceleration

MPPI is only viable because of GPUs.
*   Sample 100 paths on CPU: 50ms (Too slow for 20Hz control).
*   Sample 10,000 paths on GPU: 5ms.
*   **Tip:** Keep all tensors on GPU. Do not copy back to CPU until the final `action` is needed.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Difference between DWA (Dynamic Window Approach) and TEB?
    *   **A:** DWA samples *velocities* (v, w) for one step (Arc). TEB optimizes an *entire trajectory* (Band) over time. TEB handles "Parallel Parking", DWA cannot (it has no horizon).
2.  **Q:** Why does MPPI work with Non-Differentiable costs?
    *   **A:** It uses *sampling* to estimate gradients (Finite Differences style) rather than analytical derivatives. It just needs to *evaluate* the cost, not differentiate it.

### Challenge Task
> **Task:** Implement "Control Barrier Functions" (CBF).
> 1. Add a hard constraint to MPPI outcomes.
> 2. If $h(x) < 0$ (Collision), override control to push away from boundary.
> 3. Ensures safety even if MPC fails.

---

## 📚 Further Reading
- **TEB Paper:** Rösmann et al. (IEEE RA-L 2017).
- **MPPI Paper:** Williams et al. (ICRA 2016).
- **AutoRally:** Georgia Tech's 1/5 scale rally car using MPPI.

---

**Day 16 Complete**
