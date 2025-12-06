# Day 183: Contact-Implicit Control
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 27: Advanced Control & Dynamics

---

> **📝 Content Creator Instructions:**
> Making contact is hard. Making *good* contact is harder.
> - **Focus:** Hybrid Dynamics, Linear Complementarity Problems (LCP), and Trajectory Optimization through Contact.
> - **Code:** `contact_sim.py`. A simulation of a block sliding and hitting a wall, modeling the impact forces without heuristic "if-else" switches.
> - **Concept:** $0 \le \lambda \perp \phi(q) \ge 0$.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Formulate** contact constraints as Complementarity Conditions (the $\perp$ operator).
2.  **Solve** a simple LCP (Linear Complementarity Problem) to find contact forces.
3.  **Differentiate** between "Hybrid Systems" (Mode switching) and "Contact-Implicit" (Unified math).
4.  **Simulate** a legged robot foot touchdown without explicit state machines.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Physics Sim).

### Software Environment
```bash
pip install numpy cvxopt
```

### Prior Knowledge
- Lagrangian Dynamics (Day 120).
- Optimization (Convex stuff).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Hybrid Nightmare

Robots that interact with the world (Walking, Grasping) switch modes.
*   **Mode A (Free Motion):** $M \ddot{q} + C \dot{q} + G = \tau$.
*   **Mode B (Contact):** $M \ddot{q} + \dots = \tau + J^T \lambda$.
*   **Problem:** If you have 4 feet, you have $2^4 = 16$ modes. Hard to code `if-else` for all.

### 🔹 Part 2: Complementarity Constraints

Instead of modes, we use constraints that hold *always*.
For a distance to ground $\phi(q)$ and contact force $\lambda$:
1.  **No Penetration:** $\phi(q) \ge 0$.
2.  **Push Only (No suction):** $\lambda \ge 0$.
3.  **Complementarity:** $\phi(q) \cdot \lambda = 0$.
    *   If $\phi > 0$ (Air), then $\lambda = 0$.
    *   If $\lambda > 0$ (Pushing), then $\phi = 0$.

### 🔹 Part 3: Contact-Implicit Trajectory Optimization

We ask the optimizer (e.g., DIRCON) to find $q(t), u(t), \lambda(t)$ that satisfy dynamics AND complementarity.
Result: The optimizer *discovers* walking or jumping. We don't tell it "Lift foot now".

---

## 💻 Implementation: The Bouncing Cube

We simulate a 1D mass falling under gravity and hitting the floor using LCP time-stepping.
Equation: $M(v_{t+1} - v_t) = h(F_{ext} + \lambda)$.
Constraint: $q_{t+1} \approx q_t + h v_{t+1} \ge 0$.

### 🛠️ Project Structure
```text
day183_control/
├── src/
│   ├── lcp_solver.py
│   └── bouncing_sim.py
└── output/
    └── trajectory_plot.png
```

### 👨‍💻 LCP Solver (`src/lcp_solver.py`)

A simplified Lemke's algorithm solver (or projected Gauss-Seidel).
For $w = Mz + q, w \ge 0, z \ge 0, w^T z = 0$.

```python
import numpy as np

def solve_lcp_projected_gs(M, q, max_iter=100):
    """
    Solves LCP (Linear Complementarity Problem) using Projected Gauss-Seidel.
    Find z >= 0 such that w = Mz + q >= 0 and z.T @ w = 0.
    """
    n = len(q)
    z = np.zeros(n)
    
    for _ in range(max_iter):
        z_prev = z.copy()
        for i in range(n):
            # Update z[i]
            # w[i] = Sum(M[i,j]*z[j]) + q[i]
            # We want w[i] >= 0, z[i] >= 0, z[i]*w[i] = 0
            
            # PGS update:
            # z_new = z_old - r/M[i,i] * w_i
            # Actually, standard formula:
            sigma = 0
            for j in range(n):
                if i != j:
                    sigma += M[i,j] * z[j]
            
            # M[i,i]*z[i] + sigma + q[i] >= 0
            if M[i,i] > 1e-9:
                val = -(sigma + q[i]) / M[i,i]
                z[i] = max(0.0, val)
        
        if np.linalg.norm(z - z_prev) < 1e-6:
            break
            
    return z
```

### 👨‍💻 Simulation Loop (`src/bouncing_sim.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
from lcp_solver import solve_lcp_projected_gs

class ContactSim:
    def __init__(self):
        self.mass = 1.0
        self.g = -9.81
        self.dt = 0.01
        
        self.q = 1.0 # Initial height
        self.v = 0.0
        
    def step(self):
        # Time Stepping Formulation
        # v_next = v + dt/m * (F_g + lambda)
        # q_next = q + dt * v_next (Semi-implicit Euler)
        # Constraint: q_next >= 0 -> q + dt*v_next >= 0
        
        # Substitute v_next:
        # q + dt*(v + dt/m*Fg + dt/m*lambda) >= 0
        # q + dt*v + dt^2/m*Fg + dt^2/m*lambda >= 0
        # (dt^2/m)*lambda >= - (q + dt*v + dt^2/m*Fg)
        
        # This is LCP form: A*lambda + b >= 0
        # w = A*lambda + b
        
        A = np.array([[ (self.dt**2)/self.mass ]])
        b_val = self.q + self.dt*self.v + (self.dt**2/self.mass)*(self.mass*self.g)
        b = np.array([ b_val ]) # Note: q constraint is >= 0. LCP is w >= 0.
        
        # Solve for lambda (Contact Force / Impulse)
        lam = solve_lcp_projected_gs(A, b)
        contact_force = lam[0]
        
        # Integrate
        v_next = self.v + self.dt * (self.g + contact_force/self.mass)
        q_next = self.q + self.dt * v_next
        
        # Restitution (Bounce)?
        # The simple LCP implies inelastic collision (stick or slide). 
        # To bounce, we need velocity level constraints (Newton restitution).
        # For now, let's observe the "Sticky" landing (Inelastic).
        
        self.v = v_next
        self.q = q_next
        
        return self.q, self.v, contact_force

def main():
    sim = ContactSim()
    t_hist, q_hist, f_hist = [], [], []
    
    for t in np.arange(0, 2.0, 0.01):
        q, v, f = sim.step()
        t_hist.append(t)
        q_hist.append(q)
        f_hist.append(f)
        
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t_hist, q_hist, label='Pos')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.plot(t_hist, f_hist, label='Force', color='red')
    plt.ylabel('Contact Force (N)')
    plt.grid()
    
    plt.savefig('output/bounce_trajectory.png')
    print("Simulated. Check output plot.")
    # Expected: Parabola falling, then q=0, Force ~ 9.81 (Weight support)

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Floor is Lava"

### 1. Lab Objectives
- **Run:** `bouncing_sim.py`. Observe the mass stops exactly at $q=0$.
- **Modify:** Add "Restitution".
    *   Change constraint: $v_{next} \ge -e \cdot v_{current}$.
    *   This is a velocity-level LCP.
- **Visualize:** The ball bouncing with decaying height.

---

## 🚀 Project: "Sliding Block"

**Goal:** Friction.
1.  **State:** $(x, y)$. 2D sliding.
2.  **Constraint:** $\lambda_N \ge 0$ (Normal force).
3.  **Friction:** $|\lambda_T| \le \mu \lambda_N$.
    *   This is a "Pyramid" constraint, harder to put in LCP.
    *   Approximation: Polyhedral friction cone.
4.  **Simulate:** Block sliding to a halt.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Penetration"
*   **Cause:** Large time step $dt$. In explicit integration ($q = q + v dt$), the object teleports into the floor before force calculation.
*   **Fix:** Use CCD (Continuous Collision Detection) or Implicit Time Stepping (as implemented above).

#### 2. "Zeno's Paradox"
*   **Cause:** With restitution, bounces get infinitely small/fast. Sim stuck at $t=1.0001$.
*   **Fix:** If velocity < epsilon, force inelastic collision (settle).

---

## ⚡ Optimization: Warm Starting

LCP solvers iterate.
*   **Cold Start:** Assume $\lambda = 0$. Converges in 50 iters.
*   **Warm Start:** Assume $\lambda_t = \lambda_{t-1}$. Converges in 2 iters.
*   **Significance:** Critical for Real-Time MPC (1000Hz).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What does $\perp$ mean?
    *   **A:** Complementarity. One variable can be positive, or the other, but not both. (Strictly, their product is zero).
2.  **Q:** Why not just use specific Mode Logic (If $z < 0$, force = K*z)?
    *   **A:** That's "Spring-Damper" contact (Soft contact). It's stiff (requires small dt) and oscillatory. Constraint-based (Hard contact) is more stable for rigid body robotics.

### Challenge Task
> **Task:** "Walk".
> 1. Use the Contact-Implicit formulation to optimize a trajectory for a 2-link leg (Hip, Knee).
> 2. Objective: Move Hip +1m.
> 3. Initial Guess: Flying.
> 4. Result: Solver *invents* "Push off ground".

---

## 📚 Further Reading
- **Posa et al.:** "A Direct Method for Trajectory Optimization of Rigid Bodies Through Contact". (The DIRCON paper).
- **MuJoCo:** Uses a convex relaxation of contact math.

---

## 🔗 External Resources
### 📜 Open Source Libraries
- [ToyotaResearchInstitute/idto](https://github.com/ToyotaResearchInstitute/idto) - Inverse Dynamics Trajectory Optimization for Contact-Implicit Planning.
- [dojo-sim/ContactImplicitMPC.jl](https://github.com/dojo-sim/ContactImplicitMPC.jl) - Fast contact-implicit MPC implementation in Julia.

### 📺 Video Tutorials
- [Contact-Implicit MPC (IJRR)](https://www.youtube.com/results?search_query=Contact-Implicit+MPC) - Controlling Diverse Quadruped Motions Without Pre-Planned Contact Modes.
- [Staged Contact Optimization](https://www.youtube.com/results?search_query=Staged+Contact+Optimization) - Combining Contact-Implicit and Multi-Phase Hybrid Trajectory Optimization.

---

**Day 183 Complete**
