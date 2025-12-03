# Day 114: Kalman Filter Basics (1D)
## Phase 4: ADAS & Robotics Systems | Week 17: Sensor Fusion

---

> **📝 Day 114 Focus:**
> Sensors are noisy. The world is uncertain. The **Kalman Filter (KF)** is the optimal estimator for linear systems with Gaussian noise. It combines what we *think* happened (Prediction) with what we *saw* happened (Measurement).

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the Predict-Update cycle.
2.  **Define** the State ($x$), Covariance ($P$), Process Noise ($Q$), and Measurement Noise ($R$).
3.  **Derive** the Kalman Gain ($K$).
4.  **Implement** a 1D Kalman Filter in Python.
5.  **Visualize** how the filter converges over time.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Probability:** Mean ($\mu$) and Variance ($\sigma^2$).
-   **Day 113:** Weighted Average.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Philosophy

The KF asks two questions:
1.  **Prediction:** "Based on physics (velocity), where should the car be now?"
    -   Result: A guess with some uncertainty ($P_{pred} = P + Q$).
2.  **Update:** "The sensor says the car is at $z$. How much do I trust the sensor vs. my physics?"
    -   Result: A weighted average ($x_{new} = x_{pred} + K(z - x_{pred})$).

### 🔹 Part 2: The Equations (1D)

**Prediction:**
1.  $x_k = x_{k-1} + u$ (Motion Model)
2.  $P_k = P_{k-1} + Q$ (Uncertainty grows)

**Update:**
1.  $K = P_k / (P_k + R)$ (Kalman Gain)
2.  $x_k = x_k + K(z_k - x_k)$ (Correction)
3.  $P_k = (1 - K)P_k$ (Uncertainty shrinks)

-   $Q$: Process Noise (Wind, bumps).
-   $R$: Measurement Noise (Sensor error).
-   $K$: If $R$ is huge, $K \to 0$ (Trust Prediction). If $R$ is tiny, $K \to 1$ (Trust Measurement).

---

## 💻 Implementation: 1D Tracker

**Scenario:**
-   Car moving at constant velocity.
-   Noisy GPS measurements.
-   Task: Estimate true position.

### 🛠️ Setup
Create `week17_day114` and `kalman_1d.py`.

```bash
mkdir -p ~/ros2_ws/src/week17_day114
cd ~/ros2_ws/src/week17_day114
touch kalman_1d.py
```

### 👨‍💻 Code: The 1D Kalman Filter

```python
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter1D:
    def __init__(self, initial_x, initial_p, q, r):
        self.x = initial_x # State Estimate
        self.p = initial_p # Covariance Estimate
        self.q = q         # Process Noise
        self.r = r         # Measurement Noise
        
    def predict(self, u=0):
        # x = x + u
        self.x = self.x + u
        # P = P + Q
        self.p = self.p + self.q
        return self.x
        
    def update(self, z):
        # K = P / (P + R)
        k = self.p / (self.p + self.r)
        
        # x = x + K(z - x)
        self.x = self.x + k * (z - self.x)
        
        # P = (1 - K)P
        self.p = (1 - k) * self.p
        
        return self.x, self.p

def main():
    # --- Config ---
    n_steps = 50
    true_x = 0.0
    velocity = 1.0 # m/s
    
    # Filter Setup
    # Initial guess: x=0, P=1000 (Don't know where I am)
    kf = KalmanFilter1D(initial_x=0.0, initial_p=1000.0, q=0.1, r=5.0)
    
    # Storage
    history = {
        'true': [],
        'meas': [],
        'est': [],
        'cov': []
    }
    
    for i in range(n_steps):
        # 1. Simulate Reality
        true_x += velocity
        
        # 2. Simulate Measurement (GPS Noise sigma=sqrt(5) ~= 2.2)
        z = true_x + np.random.normal(0, np.sqrt(kf.r))
        
        # 3. KF Predict
        # We assume we know the velocity (control input u)
        kf.predict(u=velocity)
        
        # 4. KF Update
        est_x, est_p = kf.update(z)
        
        # Log
        history['true'].append(true_x)
        history['meas'].append(z)
        history['est'].append(est_x)
        history['cov'].append(est_p)
        
    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(history['true'], 'k--', label='Ground Truth')
    plt.plot(history['meas'], 'r.', label='Measurements', alpha=0.5)
    plt.plot(history['est'], 'b-', label='KF Estimate', linewidth=2)
    plt.title("1D Kalman Filter Tracking")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(history['cov'], 'g-', label='Uncertainty (P)')
    plt.title("Covariance Convergence")
    plt.ylabel("Variance (m^2)")
    plt.xlabel("Time Step")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: Tuning Q and R

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   The Blue Line (Estimate) is smoother than the Red Dots (Measurements).
    -   The Green Line (P) drops rapidly from 1000 to a stable value (~1.5).
3.  **Experiment A (Trust Sensor):**
    -   Set `r = 0.1` (Very accurate sensor).
    -   **Result:** The Blue Line follows the Red Dots closely. It's jittery but responsive.
4.  **Experiment B (Trust Physics):**
    -   Set `r = 100.0` (Terrible sensor).
    -   **Result:** The Blue Line is very smooth (almost a straight line). It ignores the noise.
    -   **Risk:** If the car changes speed, the filter will lag behind (Lag Error).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Filter Lag
**Symptom:** The estimate tracks the shape but is delayed by 5 steps.
**Cause:** `Q` is too small (Model is too rigid) or `R` is too large (Ignoring sensor).
**Solution:** Increase `Q` to allow the filter to adapt to changes faster.

#### 2. Divergence
**Symptom:** $P$ grows to infinity.
**Cause:** Prediction step adds uncertainty ($+Q$), but Update step never happens (Sensor loss).
**Solution:** This is expected behavior during sensor dropout. The uncertainty *should* grow.

---

## ⚡ Optimization & Best Practices

### 1. Initialization
How to pick $x_0$ and $P_0$?
-   **First Measurement:** Set $x_0 = z_0$.
-   **Uncertainty:** Set $P_0 = R$ (Trust the first measurement variance).
-   This avoids the initial "convergence period" where the filter flies in from 0 to 50.

### 2. Constant Velocity vs Constant Acceleration
-   Our model: $x_{k} = x_{k-1} + u$. This is a **Constant Velocity** model (if $u$ is velocity).
-   If the car accelerates, this model fails.
-   We need a **State Vector** $[x, v]$ to estimate velocity too. (Day 115).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What happens if $K=1$?
    *   **A:** $x_{new} = x_{pred} + 1(z - x_{pred}) = z$. The filter ignores the prediction and fully trusts the measurement.
2.  **Q:** Why does $P$ decrease after update?
    *   **A:** Because we gained information. Two sources of info (Prediction + Measurement) are always better than one.
3.  **Q:** What is "Process Noise" ($Q$)?
    *   **A:** The uncertainty in our motion model. "I commanded 10 m/s, but wind/friction might make it 9.9 or 10.1".

### Challenge Task
**Task:** Stop and Go.
1.  Modify simulation: Velocity changes from 1.0 to 0.0 at step 25.
2.  Keep `predict(u=1.0)` (Model assumes constant speed).
3.  Observe the "Overshoot". The filter thinks the car is still moving!
4.  This shows why we need to estimate velocity, not just assume it.

---

## 📚 Further Reading & References
-   [Kalman Filter Visualization](https://www.cs.utexas.edu/~teammco/misc/kalman_filter/)
-   [Interactive KF Tutorial](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

---

**Day 114 Complete** | Phase 4: ADAS & Robotics Systems | Week 17: Sensor Fusion
