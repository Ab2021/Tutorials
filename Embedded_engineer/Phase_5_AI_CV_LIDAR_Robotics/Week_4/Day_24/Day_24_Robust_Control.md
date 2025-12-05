# Day 24: Robust Control (Sliding Mode)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 4: Advanced Control & Dynamics

---

> **📝 Content Creator Instructions:**
> LQR and MPC assume the model $A, B$ is perfect. It never is. Wind gusts, friction changes, and payload shifts occur.
> - **Focus:** Sliding Mode Control (SMC). The brute-force way to handle uncertainty.
> - **Code:** Implementation of SMC to control a Mass-Spring-Damper system with unknown parameters.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** Robustness: The ability to maintain stability despite "Matched Uncertainty".
2.  **Derive** the Sliding Surface $s(x)$ and the Reachability Condition $s \dot{s} < 0$.
3.  **Implement** a Sliding Mode Controller with Discontinuous Control Law ($u = -k \cdot \text{sgn}(s)$).
4.  **Mitigate** the "Chattering" phenomenon using boundary layers ($\text{tanh}(s)$).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Phase Plane Analysis.
- Lyapunov Stability ($V > 0, \dot{V} < 0$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Sliding Surface

Consider a system: $\ddot{x} = f(x) + u + d(t)$
*   $d(t)$: Unknown disturbance (Wind, Noise) where $|d(t)| < D_{max}$.

We define a **Sliding Surface**:
$$ s = \dot{e} + \lambda e $$
where $e = x - x_{des}$.
*   If we can force $s \to 0$, then $\dot{e} = -\lambda e$.
*   This is a first-order ODE. The solution is $e(t) = e_0 \exp(-\lambda t)$.
*   **Magic:** If $s=0$, the error *must* decay to zero, regardless of the system dynamics! The system "Slides" along the surface to the origin.

### 🔹 Part 2: The Reachability Condition (Lyapunov)

Choose Lyapunov candidate $V = \frac{1}{2} s^2$.
To guarantee stability, we need $\dot{V} < 0 \implies s \dot{s} < 0$.

Let control law be:
$$ u = u_{eq} - k \cdot \text{sgn}(s) $$
*   $u_{eq}$: Equivalent control (Terms to cancel known dynamics).
*   $k$: Switching gain. If $k > D_{max}$, we overpower the disturbance.

### 🔹 Part 3: Chattering

A purely discontinuous sign function ($\text{sgn}(s)$) switches infinitely fast at $s=0$.
*   **Physical Reality:** Motors cannot switch voltage at Infinite Hz. Gearboxes break. High frequency vibration.
*   **Solution:** Replace $\text{sgn}(s)$ with a smooth approximation:
    *   Saturation: $\text{sat}(s/\phi)$
    *   Sigmoid: $\tanh(s / \phi)$
    *   This creates a "Boundary Layer" where we tolerate small error but avoid vibration.

---

## 💻 Implementation: Robust Position Control

System: Mass with unknown friction.
$$ m \ddot{x} + b \dot{x} + kx = u + d(t) $$
We assume we *don't* strictly know $b, k, m$.

### 🛠️ Project Structure
```text
day24_smc/
├── src/
│   ├── smc_controller.py
│   └── plant.py
└── run_robust.py
```

### 👨‍💻 Code Implementation (`src/smc_controller.py`)

```python
import numpy as np

class SMC:
    def __init__(self, lambda_gain, k_gain, layer_width=0.1):
        self.lam = lambda_gain
        self.k = k_gain
        self.phi = layer_width
        
    def compute(self, x_curr, dot_x_curr, x_des, dot_x_des, ddot_x_des):
        # 1. Error terms
        e = x_curr - x_des
        dot_e = dot_x_curr - dot_x_des
        
        # 2. Sliding Surface
        s = dot_e + self.lam * e
        
        # 3. Equivalent Control (Ideally cancels dynamics, here we assume Mass=1)
        # s_dot = ddot_e + lam * dot_e
        # We want s_dot = -k * sgn(s)
        # u_eq approx = ddot_x_des - lam * dot_e ... (Assuming partial knowledge)
        u_eq = 0 # Assume we know nothing! Pure robust control.
        
        # 4. Switching Control
        # Soft switching
        u_switch = -self.k * np.tanh(s / self.phi)
        
        return u_eq + u_switch
```

### 👨‍💻 Simulation (`run_robust.py`)

```python
import matplotlib.pyplot as plt
from src.smc_controller import SMC

# Plant
def plant_dynamics(state, u, t):
    x, v = state
    
    # Unknown Disturbance
    wind = 5.0 * np.sin(2 * t) 
    
    # True Physics (Unknown to Controller)
    # Mass=2 (Controller thinks Mass=1)
    force_total = u + wind - 0.5 * v 
    acc = force_total / 2.0
    
    return [v, acc]

# Sim Loop
smc = SMC(lambda_gain=2.0, k_gain=10.0) # k=10 is enough to beat wind=5
state = [0.0, 0.0]
ref = 10.0
dt = 0.01

history = []
for t in np.arange(0, 10, dt):
    u = smc.compute(state[0], state[1], ref, 0, 0)
    
    # Integrate
    dx = plant_dynamics(state, u, t)
    state[0] += dx[0] * dt
    state[1] += dx[1] * dt
    
    history.append([t, state[0], u])

# Plot
hist_arr = np.array(history)
plt.subplot(2,1,1)
plt.plot(hist_arr[:, 0], hist_arr[:, 1], label='Position')
plt.plot(hist_arr[:, 0], [ref]*len(hist_arr), '--', label='Target')
plt.legend()

plt.subplot(2,1,2)
plt.plot(hist_arr[:, 0], hist_arr[:, 2], label='Control Input')
plt.show()
```

### 3. Expected Output
- **Position:** Converges to 10.0 perfectly, despite the sinusoidal wind.
- **Control:** Oscillates (chatters) to counteract the wind exactly. The `tanh` smoothing reduces the sharpness.

---

## 🔬 Lab Exercise: The Tuning Game

### 1. Lab Objectives
- Increase disturbance magnitude to $> K$.
- **Observation:** The system drifts away. The sliding mode is "broken" (Reachability condition $s\dot{s} < 0$ violated).
- **Fix:** Increase $K$ (Switching Gain).
- **Trade-off:** High $K$ causes more chattering and uses more energy/voltage. The art is setting $K$ *just above* the maximum uncertainty.

---

## 🚀 Project: "Robust Drone Landing"

**Scenario:** Land a quadcopter while subjected to "Ground Effect" (Turbulence near ground).
**Model:**
$$ m \ddot{z} = T - mg - F_{ground}(z) $$
where $F_{ground} = C / z^2$ (Unknown/Unmodeled).

**Solution:**
1.  Design SMC for altitude $z$.
2.  Set $K$ high enough to overcome the extra lift from ground effect.
3.  **Result:** The drone descends at a constant velocity ($\dot{z}_{des} = -0.5$) smoothly until touchdown, ignoring the air cushion.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "High Frequency Vibration"
*   **Symptom:** Motors hum loudly, get hot.
*   **Cause:** Chattering. Boundary layer $\phi$ is too small.
*   **Fix:** Increase $\phi$. Error precision will drop slightly ($e \to \epsilon$ instead of $e \to 0$), but system life increases.

#### 2. "Windup"
*   **Symptom:** Integral terms (if using Integral SMC) accumulate error during saturation.
*   **Fix:** Clamping / Anti-windup on the sliding surface integration.

---

## ⚡ Optimization: High-Order SMC

**Super-Twisting Algorithm:**
A variation of SMC that hides the discontinuity in the integral term.
$$ u = -k_1 |s|^{0.5} \text{sgn}(s) + v $$
$$ \dot{v} = -k_2 \text{sgn}(s) $$
*   **Result:** Exact convergence in finite time witout chattering! (Ideal for actual robotics).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the "Sliding Phase"?
    *   **A:** The phase after the system reaches the surface $s=0$. The dynamics are governed solely by the surface definition ($\dot{e} + \lambda e = 0$), independent of the plant.
2.  **Q:** Why not just use high-gain PID?
    *   **A:** High-gain PID is effectively a linear approximation of SMC. But SMC provides a rigorous framework for stability bounds and finite-time convergence.
3.  **Q:** Can SMC handle unmatched uncertainty?
    *   **A:** No. If disturbance enters a channel we cannot control (e.g., side-wind on a car with no side-thrusters), we cannot cancel it directly. We can only fight its effects on the controllable states.

### Challenge Task
> **Task:** Implement Super-Twisting Control.
> 1. Use the equations above.
> 2. Compare the control smoothness vs tanh-SMC.

---

## 📚 Further Reading
- **Sliding Mode Control:** Utkin (The Father of SMC).
- **Applied Nonlinear Control:** Slotine and Li.

---

**Day 24 Complete**
