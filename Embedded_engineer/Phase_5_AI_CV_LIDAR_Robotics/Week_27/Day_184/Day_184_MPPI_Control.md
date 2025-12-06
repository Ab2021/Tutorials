# Day 184: Model Predictive Path Integral (MPPI)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 27: Advanced Control & Dynamics

---

> **📝 Content Creator Instructions:**
> Why solve one hard math problem when you can simulate 10,000 easy ones?
> - **Focus:** Sampling-based MPC, GPU Acceleration, Cost Maps, and Handling Non-Linear Dynamics without linearization (Jacobians).
> - **Code:** `mppi_torch.py`. A PyTorch-based MPPI controller for a differential drive robot navigating a clutter of obstacles.
> - **Concept:** Thermodynamic interpretation of control.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the intuition behind MPPI (Weighted average of random rollouts).
2.  **Implement** a large-batch simulation on GPU using PyTorch tensors.
3.  **Tune** MPPI hyperparameters (Temperature $\lambda$, Number of Samples $K$, Horizon $T$).
4.  **Compare** MPPI vs Linear MPC (iLQR).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU (Optional but recommended for speed).
- CPU works for small batch sizes ($K=100$).

### Software Environment
```bash
pip install torch matplotlib
```

### Prior Knowledge
- MPC (Day 153).
- Monte Carlo Sampling.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Curse of Linearization

Standard MPC (LQR/iLQR) requires derivatives ($\frac{\partial f}{\partial x}$).
*   **Problem:** Discontinuous dynamics (Collision, Sand, Mud). Jacobians explode or don't exist.
*   **Solution:** Sampling. Just run the sim forward.

### 🔹 Part 2: The Algorithm

1.  **Guess:** Keep a nominal control sequence $U = \{u_0, \dots, u_T\}$.
2.  **Perturb:** Generate $K$ rollouts. $u_{k,t} = u_t + \delta_{k,t}$, where $\delta \sim \mathcal{N}(0, \Sigma)$.
3.  **Evaluate:** Simulate forward. Calculate Cost $S_k = \sum c(x_t, u_t)$.
4.  **Weigh:** Calculate weight $w_k = e^{-\frac{1}{\lambda} (S_k - \min S)}$.
    *   Low cost $\to$ High weight.
    *   $\lambda$ (Temperature): Controls selection pressure.
5.  **Audit:** Update $u_t = u_t + \sum_{k} w_k \delta_{k,t}$.

### 🔹 Part 3: GPU Acceleration

Simulating 1 robot is fast. Simulating 2000 is... also fast, if parallel.
*   **PyTorch/CUDA:** Treats the batch dimension $K$ as strictly parallel threads.
*   **Result:** 50Hz control loops with 4000 samples.

---

## 💻 Implementation: GPU-Accelerated MPPI

We navigate a 2D robot through circular obstacles.

### 🛠️ Project Structure
```text
day184_mppi/
├── src/
│   ├── mppi_torch.py
│   └── diff_drive_model.py
└── output/
    └── mppi_viz.gif
```

### 👨‍💻 MPPI Controller (`src/mppi_torch.py`)

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Check GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {DEVICE}")

class MPPIController:
    def __init__(self, horizon=30, num_samples=1000):
        self.H = horizon
        self.K = num_samples
        
        # Dimensions (Diff Drive: x, y, theta)
        self.nx = 3
        # Controls (v, w)
        self.nu = 2
        
        # Hyperparams
        self.lambda_ = 0.1 # Temperature
        self.noise_sigma = torch.tensor([[0.5, 0.0], [0.0, 1.0]], device=DEVICE) 
        
        # Nominal Control Sequence (initialized to 0)
        self.U = torch.zeros((self.H, self.nu), device=DEVICE)
        
        # Obstacles (x, y, radius)
        self.obs = torch.tensor([
            [2.0, 0.0, 0.5],
            [3.5, 1.0, 0.5],
            [3.5, -1.0, 0.5],
            [5.0, 0.0, 0.5]
        ], device=DEVICE)
        
        self.goal = torch.tensor([6.0, 0.0, 0.0], device=DEVICE)
        
    def dynamics(self, state, action, dt=0.1):
        """
        Batch Dynamics for Diff Drive.
        state: (K, 3) [x, y, theta]
        action: (K, 2) [v, w]
        Returns: next_state (K, 3)
        """
        next_state = state.clone()
        theta = state[:, 2]
        v = action[:, 0]
        w = action[:, 1]
        
        # Kinematics
        next_state[:, 0] += v * torch.cos(theta) * dt
        next_state[:, 1] += v * torch.sin(theta) * dt
        next_state[:, 2] += w * dt
        
        return next_state
        
    def compute_cost(self, states, actions):
        """
        Batch Cost Calculation.
        states: (K, H, 3)
        actions: (K, H, 2)
        Returns: costs (K,)
        """
        # 1. Goal Cost (Terminal + Running)
        dist_to_goal = torch.norm(states[:, :, :2] - self.goal[:2], dim=2)
        goal_cost = dist_to_goal.sum(dim=1) * 1.0
        
        # 2. Collision Cost
        # Dist to each obstacle
        # Expand states to (K, H, 1, 2) and obs to (1, 1, N_obs, 2)
        pos = states[:, :, :2].unsqueeze(2) # (K,H,1,2)
        obs_pos = self.obs[:, :2].view(1, 1, -1, 2)
        obs_r = self.obs[:, 2].view(1, 1, -1)
        
        dists = torch.norm(pos - obs_pos, dim=3) # (K,H,N_obs)
        collision_mask = dists < (obs_r + 0.3) # + Robot Radius
        
        coll_cost = collision_mask.any(dim=2).float().sum(dim=1) * 1000.0
        
        # 3. Control Effort
        ctrl_cost = (actions ** 2).sum(dim=(1,2)) * 0.01
        
        return goal_cost + coll_cost + ctrl_cost
        
    def optimization_step(self, current_state):
        # 1. Generate Noise
        noise = torch.randn((self.K, self.H, self.nu), device=DEVICE) @ self.noise_sigma
        
        # 2. Perturb controls: u_k = U_nom + noise
        U_batch = self.U.unsqueeze(0) + noise
        
        # 3. Rollout Dynamics
        # Current state shape (3,) -> (K, 3)
        x_curr = current_state.unsqueeze(0).repeat(self.K, 1)
        
        states = torch.zeros((self.K, self.H, self.nx), device=DEVICE)
        
        for t in range(self.H):
            x_curr = self.dynamics(x_curr, U_batch[:, t, :])
            states[:, t, :] = x_curr
            
        # 4. Evaluate Costs
        costs = self.compute_cost(states, U_batch)
        
        # 5. Weighting
        min_cost = torch.min(costs)
        exp_costs = torch.exp(- (costs - min_cost) / self.lambda_)
        weights = exp_costs / torch.sum(exp_costs) # Normalize
        
        # 6. Update Nominal Prediction
        # u_new = Sum(w_k * u_k)
        # Reshape weights to (K, 1, 1) for broadcast
        w_expanded = weights.view(-1, 1, 1)
        
        self.U = (w_expanded * U_batch).sum(dim=0)
        
        # Shift Horizon (Receding Horizon)
        # Discard first step, append zeros at end
        action_to_apply = self.U[0].clone()
        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1] = torch.zeros(self.nu, device=DEVICE)
        
        return action_to_apply.cpu().numpy(), states.cpu().numpy() # Return all rollouts for viz

def main():
    mppi = MPPIController()
    
    # Initial State
    state = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
    
    # Sim Log
    traj = []
    
    plt.ion()
    fig, ax = plt.subplots()
    
    for t in range(100):
        action, rollouts = mppi.optimization_step(state)
        
        # Viz
        ax.clear()
        # Draw Obstacles
        for o in mppi.obs:
            circle = plt.Circle((o[0].cpu(), o[1].cpu()), o[2].cpu(), color='red')
            ax.add_patch(circle)
            
        # Draw Rollouts (Thin lines)
        # Plot every 10th rollout to be faster
        for k in range(0, mppi.K, 50):
            path = rollouts[k]
            ax.plot(path[:,0], path[:,1], color='green', alpha=0.1)
            
        # Draw Robot
        ax.plot(state[0].item(), state[1].item(), 'bo', markersize=10)
        
        # Draw Goal
        ax.plot(mppi.goal[0].item(), mppi.goal[1].item(), 'rx', markersize=12)
        
        ax.set_xlim(-1, 7)
        ax.set_ylim(-3, 3)
        plt.pause(0.01)
        
        # Sim Step
        state_tens = state.unsqueeze(0)
        act_tens = torch.tensor(action, device=DEVICE).unsqueeze(0)
        state = mppi.dynamics(state_tens, act_tens).squeeze(0)
        
        traj.append(state.cpu().numpy())
        
        dist = torch.norm(state[:2] - mppi.goal[:2])
        if dist < 0.2:
            print("Goal Reached!")
            break
            
    plt.ioff()
    print("Done.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Sand Trap"

### 1. Lab Objectives
- **Modify:** The `dynamics` function.
- **Scenario:** Add a region $x \in [3, 4]$ where specific friction is high (Sand).
- **Rule:** Inside Sand, $v_{effective} = 0.5 \times v_{cmd}$.
- **Observe:** Linear MPC fails (assumes constant model). MPPI adapts because the rollouts that go through sand get delayed (High Cost), so weights favor paths *around* the sand.

---

## 🚀 Project: "Drifting Parking"

**Goal:** Park a car by drifting.
1.  **Model:** Bicycle Model with nonlinear tire friction (Pacejka).
2.  **State:** $(x, y, \theta, v_x, v_y, \omega)$.
3.  **Cost:** Minimal time to box $x \in [0,1], y \in [0,1]$, Angle $= 0$.
4.  **Result:** MPPI discovers "Scandinavian Flick" maneuver automatically.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Mode Collapse"
*   **Cause:** Temperature $\lambda$ too small. Algorithm puts 100% weight on the single best sample. No smoothing.
*   **Fix:** Increase $\lambda$.

#### 2. "Local Minima"
*   **Cause:** All samples crash into the wall.
*   **Fix:** Increase Noise Variance $\Sigma$. You need samples that accidentally *jump* over the wall (conceptually) or turn hard enough to find the opening.

---

## ⚡ Optimization: Spline Parameterization

Instead of sampling 30 inputs ($u_0 \dots u_{29}$), sample 3 Control Points for a Spline.
*   **Advantage:** Smoother Trajs. Dimensionality reduction ($30 \to 3$).
*   **Disadvantage:** Less reactive to sudden changes.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Jacobian vs Sampling?
    *   **A:** Jacobian = Gradient Descent (Fast, Local). Sampling = Genetic Algorithm (Slow/Parallel, Global-ish).
2.  **Q:** Does MPPI guarantee collision avoidance?
    *   **A:** Probabilistically yes. Strictly no. If 0/1000 samples avoid collision, you crash. (Need fallback safety).

### Challenge Task
> **Task:** "Tunnel Following".
> 1. Cost Map = Image (White=Free, Black=Wall).
> 2. Cost filtering directly lookups pixels.
> 3. Navigate a complex maze image.

---

## 📚 Further Reading
- **Williams et al.:** "Information Theoretic MPC for Model-Based Reinforcement Learning". (The original MPPI paper).
- **NVIDIA Isaac Gym:** Uses MPPI for huge swarms.

---

**Day 184 Complete**
