# Day 25: Adaptive Control (MRAC)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 4: Advanced Control & Dynamics

---

> **📝 Content Creator Instructions:**
> Robust Control (SMC) fights uncertainty with high gain. Adaptive Control "learns" the uncertainty and cancels it.
> - **Focus:** Model Reference Adaptive Control (MRAC) and Online Parameter Estimation.
> - **Code:** Implementation of an MRAC controller for an arm lifting unknown weights.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Robust Control (Fixed gains) and Adaptive Control (Time-varying gains).
2.  **Derive** the Parameter Adaption Law using Lyapunov Stability (MIT Rule).
3.  **Implement** MRAC to force a plant with unknown parameters to behave like a Reference Model.
4.  **Simulate** a manipulator picking up a heavy object and adapting its torque instantly.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Lyapunov Stability.
- Linear Algebra.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Concept

We have a Plant: $\dot{x} = -a x + b u$ ($a, b$ are unknown).
We want it to behave like a Reference Model: $\dot{x}_m = -a_m x_m + b_m r$ ($a_m, b_m$ are chosen by us).

If we knew $a, b$, we could choose control law:
$$ u = k_x x + k_r r $$
By matching terms (Algebra), we find ideal gains $k_x^*, k_r^*$.

**Adaptive Idea:**
Since we don't know ideal gains, we estimate them as $\hat{k}_x(t), \hat{k}_r(t)$ and update them based on the tracking error $e = x - x_m$.

### 🔹 Part 2: The Adaption Law (Lyapunov)

Define Lyapunov Function $V(e, \tilde{K}) = \frac{1}{2} e^2 + \frac{1}{2\gamma} \tilde{K}^2$.
To ensure $\dot{V} < 0$, the update law must be:
$$ \dot{\hat{k}}_x = -\gamma x e \cdot \text{sgn}(b) $$
$$ \dot{\hat{k}}_r = -\gamma r e \cdot \text{sgn}(b) $$

*   $\gamma$: Learning Rate (Adaptation Gain).
*   $e$: Tracking error.
*   $x, r$: Regressor vector.

### 🔹 Part 3: Robust vs Adaptive

| Feature | Robust (SMC) | Adaptive (MRAC) |
| :--- | :--- | :--- |
| **Uncertainty** | Bounded $|d| < D$ | Constant/Slowly Varying |
| **Action** | High frequency switching | Parameter convergence |
| **Noise** | Robust to noise | Sensitive to noise (Drift) |
| **Convergence** | Asymptotic | Asymptotic (Output), Maybe (Params) |

---

## 💻 Implementation: Adaptive Cruise Control

System: $m \dot{v} = u - f v$.
*   $m$: Mass (Unknown, e.g., hauling a trailer).
*   $f$: Friction (Unknown).
Reference: $\dot{v}_m = -1 v_m + 1 r$ (First order response with $\tau=1s$).

### 🛠️ Project Structure
```text
day25_mrac/
├── src/
│   ├── mrac_controller.py
│   └── plant.py
└── run_adaptive.py
```

### 👨‍💻 Code Implementation (`src/mrac_controller.py`)

```python
import numpy as np

class MRAC:
    def __init__(self, gamma_x, gamma_r, am, bm):
        self.gam_x = gamma_x
        self.gam_r = gamma_r
        
        # Reference Model Params
        self.am = am
        self.bm = bm
        
        # Estimated Gains (Start with 0 or random)
        self.kx = 0.0
        self.kr = 0.0
        
        # Reference State
        self.xm = 0.0
        
    def update(self, x, r, dt):
        # 1. Update Reference Model
        dx_m = -self.am * self.xm + self.bm * r
        self.xm += dx_m * dt
        
        # 2. Compute Error
        e = x - self.xm
        
        # 3. Adaptation Law (Lyapunov Rule)
        # Sign assumption: we assume we know the sign of 'b' (engine pushes forward)
        sign_b = 1.0 
        
        d_kx = -self.gam_x * x * e * sign_b
        d_kr = -self.gam_r * r * e * sign_b
        
        self.kx += d_kx * dt
        self.kr += d_kr * dt
        
        # 4. Control Law
        u = self.kx * x + self.kr * r
        
        return u, self.xm, e, self.kx
```

### 👨‍💻 Simulation (`run_adaptive.py`)

```python
import matplotlib.pyplot as plt
from src.mrac_controller import MRAC

# Plant with unknown params
# True system: dot_x = -2*x + 3*u
# Reference: dot_x = -x + r (We want it to be slower/stable)

def plant_dynamics(x, u):
    return -2.0 * x + 3.0 * u # a=2, b=3

mrac = MRAC(gamma_x=2.0, gamma_r=2.0, am=1.0, bm=1.0)

x = 0.0
dt = 0.01
history = []

for t in np.arange(0, 20, dt):
    # Step Reference (Speed limit changes)
    r = 10.0 if t < 10 else 5.0
    
    u, xm, e, kx = mrac.update(x, r, dt)
    
    # Simulate Plant
    dx = plant_dynamics(x, u)
    x += dx * dt
    
    history.append([t, x, xm, u, kx])

# Visualization
hist = np.array(history)
plt.subplot(3,1,1)
plt.plot(hist[:,0], hist[:,1], label='Actual')
plt.plot(hist[:,0], hist[:,2], '--', label='Reference')
plt.title("Tracking Performance")
plt.legend()

plt.subplot(3,1,2)
plt.plot(hist[:,0], hist[:,4], label='Estimated Gain Kx')
plt.title("Parameter Adaptation")
plt.legend()

plt.show()
```

### 3. Expected Output
- **Start:** Actual state oscillates or lags behind Reference.
- **Middle:** Gain $K_x$ evolves. Oscillation dampens.
- **End:** Actual state tracks Reference perfectly ($e \to 0$). The controller has "learned" the plant physics.

---

## 🔬 Lab Exercise: The Mystery Payload

### 1. Lab Objectives
- Simulate a Drone Lifting task via MRAC.
- At $t=5s$, double the mass (simulate picking up a box).
- **Observe:** The drone drops momentarily ($z$ decreases). The Adapter senses the error, ramps up the Thrust Gain. The drone recovers and tracks the reference ascent.

### 2. Tuning Guide
- **Low Gamma:** Slow adaptation. The drone drops significantly before recovering.
- **High Gamma:** Fast adaptation, but risks oscillation (instability).

---

## 🚀 Project: "Adaptive Friction Compensation"

**Goal:** Control a robotic joint that has sticky friction (Stiction).
1.  **Model:** $\tau = I \ddot{\theta} + F_c \text{sgn}(\dot{\theta})$.
2.  **Adaptive Term:** $\hat{F}_c \text{sgn}(\dot{\theta})$.
3.  **Adaptation:** $\dot{\hat{F}}_c = -\gamma |\dot{\theta}| e$.
4.  **Result:** The robot learns exactly how much friction exists and cancels it out, creating zero-error tracking.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Parameter Drift" (bursting)
*   **Symptom:** Parameters drift to infinity when error is zero (Noise).
*   **Cause:** $e \approx 0$ but measurement noise exists. Integral action accumulates noise.
*   **Fix:** **Deadzone**. Turn off adaptation if $|e| < \epsilon$. Or **Sigma-Modification** (Leaky Integrator).

#### 2. "High Frequency Oscillation"
*   **Cause:** Gamma/Learning Rate is too high.
*   **Fix:** Reduce $\gamma$.

---

## ⚡ Optimization: Composite Adaptation

Use both **Tracking Error** ($e$) and **Prediction Error** ($\epsilon$) to drive adaptation.
*   Prediction Error: Compare predicted state $\hat{x}$ with actual $x$.
*   Ideally, $x$ contains less noise than $\dot{x}$, allowing smoother adaptation.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Does the parameter estimate $\hat{K}$ converge to the true optimal $K^*$?
    *   **A:** Not necessarily! It only guarantees $e \to 0$. To guarantee parameter convergence, the input signal must be "Persistently Exciting" (Rich in frequencies).
2.  **Q:** What is the "MIT Rule"?
    *   **A:** An early heuristic $\dot{\theta} = -\gamma e \frac{\partial e}{\partial \theta}$. It is the gradient descent of the squared error.
3.  **Q:** Can MRAC handle sudden changes?
    *   **A:** Yes, that is its main purpose (picking up payloads, fuel burning off).

### Challenge Task
> **Task:** Implement 'Personeity' Adaptation.
> 1. In exoskeleton control, different humans have different stiffness.
> 2. Use MRAC to adapt the assistance torque to the wearer's stiffness.

---

## 📚 Further Reading
- **Adaptive Control:** Astrom and Wittenmark.
- **Stable Adaptive Systems:** Narendra and Annaswamy.

---

**Day 25 Complete**
