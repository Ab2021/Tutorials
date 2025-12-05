# Day 22: Optimal Control (LQR)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 4: Advanced Control & Dynamics

---

> **📝 Content Creator Instructions:**
> Planning tells us where to go. Control tells us how to fire the motors.
> - **Focus:** Linear Quadratic Regulator (LQR). The most robust linear controller in existence.
> - **Code:** Stabilization of a 2D Drone (Cart-Pole) using LQR.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Formulate** a Dynamic System in State-Space form ($\dot{x} = Ax + Bu$).
2.  **Define** the Cost Function $J$ for optimal control (Performance vs Effort).
3.  **Solve** the Algebraic Riccati Equation (ARE) to find the optimal gain $K$.
4.  **Implement** an LQR controller to balance an unstable system (Inverted Pendulum).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install control numpy matplotlib
```

### Prior Knowledge
- Linear Algebra (Eigenvalues).
- Differential Equations.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: State-Space Representation

PID controllers are SISO (Single Input Single Output). Robotics is MIMO.
We describe systems as:
$$ \dot{x} = Ax + Bu $$
$$ y = Cx + Du $$

*   $x$: State Vector (Position, Velocity, Angle, Angular Vel).
*   $u$: Control Input (Motor Thrust / Torque).
*   $A$: System Matrix (Internal Physics).
*   $B$: Input Matrix (How motors affect state).

### 🔹 Part 2: The Optimal Control Optimization

We want to drive $x \to 0$ (Stabilize) while minimizing energy.
$$ J = \int_0^\infty (x^T Q x + u^T R u) dt $$
*   $Q$ Matrix: Penalty on State Error (e.g., "Accuracy is critical").
*   $R$ Matrix: Penalty on Control Effort (e.g., "Save Battery").

### 🔹 Part 3: LQR Solution

The optimal control law is a simple **Linear State Feedback**:
$$ u = -Kx $$
Where $K = R^{-1} B^T P$, and $P$ is the solution to the **Result Algebraic Riccati Equation (CARE)**:
$$ A^T P + P A - P B R^{-1} B^T P + Q = 0 $$

*   **Robustness:** LQR guarantees infinite Gain Margin and $60^\circ$ Phase Margin (i.e., very stable).

---

## 💻 Implementation: Inverted Pendulum

We will balance a pole purely by math, no PID tuning!

### 🛠️ Project Structure
```text
day22_lqr/
├── src/
│   ├── dynamics.py
│   ├── lqr_solver.py
└── run_sim.py
```

### 👨‍💻 Code Implementation (`src/lqr_solver.py`)

```python
import numpy as np
import scipy.linalg

def solve_lqr(A, B, Q, R):
    """
    Solves Continuous Algebraic Riccati Equation
    """
    # Scipy solves P for: A.T P + P A - P B R^-1 B.T P + Q = 0
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    
    # K = R^-1 B.T P
    K = np.linalg.inv(R) @ B.T @ P
    
    return K
```

### 👨‍💻 Simulation (`run_sim.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
from src.lqr_solver import solve_lqr

# System: Inverted Pendulum on Cart
# x = [p, v, theta, omega]
M = 0.5  # Mass of Cart
m = 0.2  # Mass of Pole
b = 0.1  # Friction
L = 0.3  # Length
I = 0.006 # Inertia
g = 9.8

# Linearized Dynamics around Upright (theta=0)
denom = I*(M+m) + M*m*L**2

A = np.array([
    [0, 1, 0, 0],
    [0, -(I+m*L**2)*b/denom, (m**2*g*L**2)/denom, 0],
    [0, 0, 0, 1],
    [0, -(m*L*b)/denom, m*g*L*(M+m)/denom, 0]
])

B = np.array([
    [0],
    [(I+m*L**2)/denom],
    [0],
    [m*L/denom]
])

# Design Costs
# Q: High penalty on Theta (Angle), Low on Position
Q = np.diag([1.0, 1.0, 10.0, 1.0]) 

# R: Low penalty on Force (Use as much power as needed)
R = np.array([[0.01]])

# Solve
K = solve_lqr(A, B, Q, R)
print("Optimal Gain K:", K)

# Simulate Closed Loop: dx = (A - BK)x
dt = 0.01
x = np.array([0.0, 0.0, 0.1, 0.0]) # Start tipped 0.1 rad
history = [x]

for t in range(500):
    u = -K @ x
    dx = A @ x + B @ u
    x = x + dx * dt
    history.append(x)

# Plot
hist_arr = np.array(history)
plt.plot(hist_arr[:, 2]) # Theta
plt.title("Pendulum Angle Stabilization")
plt.show()
```

---

## 🔬 Lab Exercise: Drone Altitude Control

### 1. Lab Objectives
- Model the altitude dynamics of a quadcopter: $\ddot{z} = \frac{T}{m} - g$.
- Linearize around Hover ($T_{hover} = mg$).
- Design LQR for $z \to 10m$.
- **Experiment:** Compare aggressive $Q$ vs lazy $Q$.

### 2. Step-by-Step Guide
1.  State $x = [z, \dot{z}]$.
2.  Input $u = \Delta Thrust$.
3.  $A = [[0, 1], [0, 0]]$.
4.  $B = [[0], [1/m]]$.
5.  Set $Q = diag(100, 1)$ (Focus on height accuracy).
6.  Simulate.

---

## 🚀 Project: "LQR Path Tracking"

**Goal:** Make a car follow a trajectory path.
1.  **Error Dynamics:** Calculate Cross-Track Error ($e$) and Heading Error ($\psi$).
2.  **State:** $x = [e, \dot{e}, \psi, \dot{\psi}]$.
3.  **LQR:** Design $K$ to minimize these errors.
4.  **Feedforward:** Add Feedforward term curvature $\kappa$ to handle corners.
    $$ \delta = -Kx + \delta_{ff} $$

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Uncontrollable System"
*   **Symptom:** `solve_continuous_are` fails.
*   **Cause:** Your system physics are flawed (Rank(Ctrb) < n). E.g., trying to control horizontal position of a drone using only vertical thrust. Impossible.
*   **Fix:** Check Controllability Matrix `ctrb(A, B)`.

#### 2. "Saturation"
*   **Symptom:** Sim works, Real Robot flips.
*   **Cause:** LQR commanded 1000N force. Motor max is 10N.
*   **Fix:** Tune $R$ higher to penalize large inputs. Or use MPC (Day 23).

---

## ⚡ Optimization: Metric Units

LQR depends heavily on units.
*   If $x$ is in meters (0-10) and $\theta$ is in radians (0-0.1), giving them equal weight in $Q$ effectively ignores $\theta$.
*   **Normalization:** Always scale your Q/R matrices based on maximum expected values.
    $$ Q_{ii} \approx \frac{1}{x_{max}^2} $$

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What happens if $R$ is very large?
    *   **A:** The controller becomes "Lazy" (uses very little control effort). Response is slow.
2.  **Q:** Does LQR handle limits (e.g., Max Voltage)?
    *   **A:** No. LQR is unconstrained. It assumes infinite power is available.
3.  **Q:** Why not use PID?
    *   **A:** PID doesn't account for coupling. In LQR, pushing the cart *right* affects the pole *angle*. LQR considers this interaction automatically via the $A$ matrix.

### Challenge Task
> **Task:** Pole Placement.
> 1. Use `scipy.signal.place_poles`.
> 2. Force eigenvalues to be at `-1, -2, -3, -4`.
> 3. Compare the generated $K$ with LQR's $K$.

---

## 📚 Further Reading
- **Underactuated Robotics:** Russ Tedrake (MIT).
- **Control Systems:** Ogata.

---

**Day 22 Complete**
