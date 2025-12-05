# Day 62: System Identification (Physics ID)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 9: Simulation & Sim-to-Real

---

> **📝 Content Creator Instructions:**
> The simulation thinks the robot weighs 5kg. The scale says 5.2kg. The Data says 5.15kg. Trust the Data.
> - **Focus:** Estimating Inertial Parameters (Mass, CoM, Inertia Matrix), Friction, and Motor Constants ($K_t$).
> - **Code:** A Least Squares solver to find robot parameters from recorded Torque/Position data.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Formulate** the Dynamics Equation in regressor form: $Y(q, \dot{q}, \ddot{q}) \cdot \pi = \tau$.
2.  **Generate** Excitation Trajectories (Fourier/Sweeps) to maximize data richness.
3.  **Solve** the Ordinary Least Squares (OLS) problem to find $\pi$ (Parameters).
4.  **Validate** the identified model by comparing predicted torque vs actual torque.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Robot with Torque sensing (or current sensing).

### Software Environment
```bash
pip install numpy scipy matplotlib
```

### Prior Knowledge
- Rigid Body Dynamics (Euler-Lagrange).
- Matrix Algebra ($A x = b$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Equation of Motion

$$ \tau = M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) + F_{fric}(\dot{q}) $$
*   **Problem:** We know $q, \dot{q}, \ddot{q}$ (Encoders) and $\tau$ (Sensors). We don't know $M, C, G$ (Model).
*   **Linearity of Parameters:** The dynamics are *linear* with respect to the parameters $\pi$ (Mass, Inertia, Friction).
    $$ \tau = Y(q, \dot{q}, \ddot{q}) \cdot \pi $$
    *   $Y$: Regressor Matrix (Known Data).
    *   $\pi$: Parameter Vector (Unknowns: $m_1, m_1 c_{x1}, I_{xx1}, ...$).

### 🔹 Part 2: Excitation Trajectories

To find mass, we must accelerate. To find friction, we must move at constant velocity. To find CoM, we must hold poses against gravity.
*   **Bad Trajectory:** Sitting still. (Only G identified).
*   **Good Trajectory:** Sinusoidal sweep covering full workspace and velocity range. "Persistent Excitation".

### 🔹 Part 3: Least Squares Estimation

Collect $N$ samples. Stack them.
$$
\begin{bmatrix}
\tau_1 \\ \vdots \\ \tau_N
\end{bmatrix}
=
\begin{bmatrix}
Y_1 \\ \vdots \\ Y_N
\end{bmatrix}
\cdot \pi
$$
$$ \mathbf{\tau} = \mathbf{Y} \cdot \pi $$
$$ \pi_{est} = (\mathbf{Y}^T \mathbf{Y})^{-1} \mathbf{Y}^T \mathbf{\tau} $$

---

## 💻 Implementation: 1-DOF Identification

We will identify Mass and Friction of a simple Pendulum.
Model: $\tau = ml^2 \ddot{q} + mgl \sin(q) + b \dot{q}$.
Parameters $\pi = [ml^2, mgl, b]^T$. (Inertia, Gravity Moment, Friction).

### 🛠️ Project Structure
```text
day62_sysid/
├── data/
│   ├── experiment_data.csv (t, q, dq, ddq, tau)
├── src/
│   ├── regressor.py
│   └── ols_solver.py
└── run_id.py
```

### 👨‍💻 Data Generation (Simulation for now)

```python
import numpy as np
import pandas as pd

def generate_data():
    # True Parameters
    m = 2.0
    l = 0.5
    g = 9.81
    b = 0.1
    
    # Derived True Params
    pi_true = [m*l**2, m*g*l, b] # [0.5, 9.81, 0.1]
    
    # Trajectory
    t = np.linspace(0, 10, 1000)
    q = np.sin(t) + np.sin(3*t) # Exciting traj
    dq = np.gradient(q, t)
    ddq = np.gradient(dq, t)
    
    # Torque with noise
    tau = (m*l**2)*ddq + (m*g*l)*np.sin(q) + b*dq
    tau += np.random.normal(0, 0.05, len(t)) # Sensor noise
    
    df = pd.DataFrame({'q': q, 'dq': dq, 'ddq': ddq, 'tau': tau})
    return df, pi_true
```

### 👨‍💻 Regressor & Solver (`src/ols_solver.py`)

```python
import numpy as np

def build_regressor(df):
    # Row: Y_i = [ddq, sin(q), dq]
    # Tau_i = tau
    
    N = len(df)
    Y = np.zeros((N, 3))
    tau = np.zeros(N)
    
    for i in range(N):
        row = df.iloc[i]
        Y[i, 0] = row['ddq']       # Coeff for Inertia (ml^2)
        Y[i, 1] = np.sin(row['q']) # Coeff for Gravity (mgl)
        Y[i, 2] = row['dq']        # Coeff for Friction (b)
        
        tau[i] = row['tau']
        
    return Y, tau

def solve_ols(Y, tau):
    # pi = pinv(Y) * tau
    pi_est, residuals, rank, s = np.linalg.lstsq(Y, tau, rcond=None)
    return pi_est
```

### 👨‍💻 Main Loop (`run_id.py`)

```python
from src.ols_solver import build_regressor, solve_ols
# from data_gen import generate_data (defined above)

# 1. Get Data
df, pi_true = generate_data()

# 2. Build Regressor
Y, tau = build_regressor(df)

# 3. Solve
pi_est = solve_ols(Y, tau)

print("--- Results ---")
print(f"True Params: {pi_true}")
print(f"Est Params:  {pi_est}")

# Calculate Errors
err = np.abs(np.array(pi_true) - pi_est)
print(f"Errors: {err}")

# Extract Physical Params (Need assumed l=0.5)
l_assumed = 0.5
m_est_inertia = pi_est[0] / (l_assumed**2)
m_est_gravity = pi_est[1] / (9.81 * l_assumed)
b_est = pi_est[2]

print(f"Est Mass (from Inertia): {m_est_inertia:.3f}")
print(f"Est Mass (from Gravity): {m_est_gravity:.3f}")
```

### 3. Expected Output
*   Should be very close to True values (e.g., Mass $\approx 2.001$).
*   If noise is high, error increases.

---

## 🔬 Lab Exercise: The "Black Box" Payload

### 1. Lab Objectives
- Attach unknown weight to robot.
- Run the identification script.
- **Find:** Mass of the weight.
- **Check:** Move robot slowly (Quasi-static).
    - $\tau \approx G(q)$.
    - $\tau \approx m g l \sin(q)$.
    - Plot $\tau$ vs $\sin(q)$. Slope determines Mass.

---

## 🚀 Project: "Sim Calibration"

**Goal:** Automatic URDF Correction.
1.  **Real:** Run ID trajectory. Estimate $M_{real}$.
2.  **Sim:** Parse URDF. Read `<mass>`.
3.  **Update:** Overwrite URDF with identified values.
4.  **Verify:** Run same trajectory in Sim. Check if $\tau_{sim}$ matches $\tau_{real}$.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Physical Inconsistency"
*   **Symptom:** Estimated Mass is negative ($-5.0$ kg).
*   **Cause:** Noise, or poor excitation (e.g., constant velocity). OLS mathematically finds best fit, even if physically impossible.
*   **Fix:** **Constrained Least Squares**. Enforce $m > 0, I > 0$. Needs Convex Optimization (CVXPY).

#### 2. "Coulomb Friction"
*   **Symptom:** Bad fit at low velocities.
*   **Cause:** We only modeled Viscous Friction ($b \dot{q}$). Real motors have Coulomb regmies ($sign(\dot{q})$) and Stiction.
*   **Fix:** Add $sign(\dot{q})$ term to regressor.

---

## ⚡ Optimization: Recursive Newton-Euler (RNEA)

Constructing the $Y$ matrix for 6-DOF is huge ($6 \times 50$ parameters).
*   **Solution:** Numerical differentiation of the Inverse Dynamics algorithm.
*   Instead of symbolic formulas, call `RNEA` with Basis Vectors.
*   Efficient for high-DOF chains.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can we identify Link Length ($l$)?
    *   **A:** No, not easily from Torque dynamics alone. Usually assumes Kinematics are known/calibrated separately.
2.  **Q:** Why not just weigh the parts?
    *   **A:** Cables, grease, and electronics add mass/friction not listed in CAD. Physics ID captures the *effective* dynamics.
3.  **Q:** What is "Base Parameters"?
    *   **A:** Some parameters are redundant (e.g., Mass of link 1 + Mass of link 2 might perform identically if joint 2 is locked). We identify the minimal set.

### Challenge Task
> **Task:** Friction Map.
> 1. Move joint at constant velocities: $0.1, 0.2, ... 1.0$ rad/s.
> 2. Record constant torque required.
> 3. Plot $\tau$ vs $\dot{q}$.
> 4. You will see the Stribeck Curve (Dip at low velocity).

---

## 📚 Further Reading
- **Robot Dynamics and Control:** Spong & Hutchinson.
- **System Identification:** Ljung.

---

**Day 62 Complete**
