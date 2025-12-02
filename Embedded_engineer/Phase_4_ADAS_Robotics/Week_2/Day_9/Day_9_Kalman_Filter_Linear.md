# Day 9: The Linear Kalman Filter
## Phase 4: ADAS & Robotics Systems | Week 2: Sensor Fusion & State Estimation

---

> **📝 Day 9 Focus:**
> The **Kalman Filter (KF)** is arguably the most important algorithm in control theory and robotics. It is the optimal estimator for linear systems with Gaussian noise. Today, we derive the KF equations, understand the role of each matrix (F, H, Q, R, P), and implement a tracker for an autonomous vehicle.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Derive** the Kalman Filter equations from the properties of Gaussian multiplication.
2.  **Construct** the State Transition Matrix ($F$) and Measurement Matrix ($H$) for a Constant Velocity model.
3.  **Tune** the Process Noise ($Q$) and Measurement Noise ($R$) covariance matrices.
4.  **Implement** a Linear Kalman Filter in Python to track a vehicle in 1D and 2D.
5.  **Analyze** the convergence of the Error Covariance Matrix ($P$).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 8:** Gaussian distributions, Covariance matrices.
-   **Physics:** Kinematics equations ($x = x_0 + vt + 0.5at^2$).
-   **Linear Algebra:** Matrix multiplication, Transpose, Inverse.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows/Mac with Python).

### Software Stack
-   **Python Libraries:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The System Model

The Kalman Filter assumes the world is **Linear** and **Gaussian**.

#### 1.1 State Vector ($x$)
The state is what we want to know. For a car moving in 1D:
$$ \mathbf{x} = \begin{bmatrix} p \\ v \end{bmatrix} $$
Where $p$ is position, $v$ is velocity.

#### 1.2 Process Model (Prediction)
How does the state evolve over time?
$$ \mathbf{x}_{k} = F \mathbf{x}_{k-1} + B \mathbf{u}_{k} + \mathbf{w}_{k} $$

-   $F$: **State Transition Matrix**.
-   $B$: **Control Input Matrix**.
-   $u$: **Control Vector** (e.g., acceleration).
-   $w$: **Process Noise** (Gaussian, $\mathcal{N}(0, Q)$).

**Example (Constant Velocity):**
$$ p_k = p_{k-1} + v_{k-1} \Delta t $$
$$ v_k = v_{k-1} $$

Matrix form:
$$ \begin{bmatrix} p_k \\ v_k \end{bmatrix} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} p_{k-1} \\ v_{k-1} \end{bmatrix} $$
So, $F = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}$.

#### 1.3 Measurement Model (Correction)
How do sensors relate to the state?
$$ \mathbf{z}_{k} = H \mathbf{x}_{k} + \mathbf{v}_{k} $$

-   $z$: **Measurement Vector** (e.g., GPS position).
-   $H$: **Measurement Matrix**.
-   $v$: **Measurement Noise** (Gaussian, $\mathcal{N}(0, R)$).

**Example:**
We only measure position ($p$), not velocity.
$$ z_k = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} p_k \\ v_k \end{bmatrix} $$
So, $H = \begin{bmatrix} 1 & 0 \end{bmatrix}$.

---

### 🔹 Part 2: The Kalman Filter Algorithm

The KF is a recursive two-step process: **Predict** -> **Update**.

#### 2.1 Prediction Step (Time Update)
We project the state and uncertainty forward in time.

1.  **Project State:**
    $$ \hat{\mathbf{x}}_{k|k-1} = F \hat{\mathbf{x}}_{k-1|k-1} + B \mathbf{u}_k $$
2.  **Project Covariance:**
    $$ P_{k|k-1} = F P_{k-1|k-1} F^T + Q $$
    *Note: Uncertainty increases here because we add Process Noise $Q$.*

#### 2.2 Update Step (Measurement Update)
We correct the prediction using the new measurement $z_k$.

1.  **Innovation (Residual):**
    $$ \mathbf{y}_k = \mathbf{z}_k - H \hat{\mathbf{x}}_{k|k-1} $$
    *(Difference between what we measured and what we expected)*
2.  **Innovation Covariance:**
    $$ S_k = H P_{k|k-1} H^T + R $$
3.  **Kalman Gain:**
    $$ K_k = P_{k|k-1} H^T S_k^{-1} $$
    *(Weighting factor: High if sensor is precise, Low if prediction is precise)*
4.  **Update State:**
    $$ \hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + K_k \mathbf{y}_k $$
5.  **Update Covariance:**
    $$ P_{k|k} = (I - K_k H) P_{k|k-1} $$
    *Note: Uncertainty decreases here because we gained information.*

---

### 🔹 Part 3: Tuning Q and R

This is the "Dark Art" of Kalman Filtering.

#### 3.1 Measurement Noise ($R$)
-   Represents sensor error.
-   **Source:** Datasheets (e.g., GPS accuracy $\pm 2m$).
-   **Effect:**
    -   High $R$: Filter trusts prediction more (slow response, smooth).
    -   Low $R$: Filter trusts measurement more (fast response, jittery).

#### 3.2 Process Noise ($Q$)
-   Represents model uncertainty (e.g., wind, bumps, driver intent).
-   **Source:** Tuned experimentally.
-   **Effect:**
    -   High $Q$: Filter assumes system is volatile (trusts measurement more).
    -   Low $Q$: Filter assumes system follows model strictly (trusts prediction more).

**Continuous White Noise Model:**
For Constant Velocity, noise enters via acceleration ($a$).
$$ Q = \sigma_a^2 \begin{bmatrix} \frac{\Delta t^4}{4} & \frac{\Delta t^3}{2} \\ \frac{\Delta t^3}{2} & \Delta t^2 \end{bmatrix} $$

---

## 💻 Implementation: 1D Vehicle Tracker

We will simulate a car moving at constant velocity, measure its position with noise, and estimate its true position and velocity.

### 🛠️ Setup
Create `week2_day9` and `kalman_1d.py`.

```bash
mkdir -p ~/ros2_ws/src/week2_day9
cd ~/ros2_ws/src/week2_day9
touch kalman_1d.py
```

### 👨‍💻 Code: Linear Kalman Filter Class

```python
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F  # State Transition
        self.H = H  # Measurement Matrix
        self.Q = Q  # Process Noise
        self.R = R  # Measurement Noise
        self.P = P  # Covariance
        self.x = x  # State Vector

    def predict(self):
        # x = Fx
        self.x = self.F @ self.x
        # P = FPF' + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        # y = z - Hx
        y = z - self.H @ self.x
        # S = HPH' + R
        S = self.H @ self.P @ self.H.T + self.R
        # K = PH'S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # x = x + Ky
        self.x = self.x + K @ y
        # P = (I - KH)P
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x

def run_simulation():
    # --- Configuration ---
    dt = 0.1
    n_steps = 100
    
    # True System
    true_x = np.array([[0], [5]]) # Pos=0, Vel=5 m/s
    
    # Kalman Filter Initialization
    F = np.array([[1, dt],
                  [0, 1]])
    
    H = np.array([[1, 0]]) # Measure position only
    
    # Process Noise (Assume acceleration variance = 1.0)
    sigma_a = 1.0
    Q = np.array([[0.25*dt**4, 0.5*dt**3],
                  [0.5*dt**3, dt**2]]) * sigma_a**2
    
    # Measurement Noise (GPS accuracy = 3m)
    sigma_z = 3.0
    R = np.array([[sigma_z**2]])
    
    # Initial Guess (Wrong!)
    x_init = np.array([[0], [0]]) # Assume stationary
    P_init = np.array([[100, 0],
                       [0, 100]]) # High uncertainty
    
    kf = KalmanFilter(F, H, Q, R, P_init, x_init)
    
    # --- Storage for Plotting ---
    history_true = []
    history_meas = []
    history_est = []
    history_cov = [] # Store P[0,0] (Pos Variance)
    
    # --- Loop ---
    for i in range(n_steps):
        # 1. Simulate Reality
        # Add process noise to truth (random acceleration)
        noise_process = np.array([[0.5*dt**2], [dt]]) * np.random.normal(0, sigma_a)
        true_x = F @ true_x + noise_process
        
        # 2. Simulate Measurement
        # z = Hx + v
        z = H @ true_x + np.random.normal(0, sigma_z)
        
        # 3. Filter Predict
        kf.predict()
        
        # 4. Filter Update
        kf.update(z)
        
        # Store
        history_true.append(true_x[0,0])
        history_meas.append(z[0,0])
        history_est.append(kf.x[0,0])
        history_cov.append(kf.P[0,0])

    # --- Visualization ---
    t = np.arange(n_steps) * dt
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Position
    plt.subplot(2, 1, 1)
    plt.plot(t, history_true, 'g-', label='True Position')
    plt.plot(t, history_meas, 'r.', alpha=0.5, label='Measurements (GPS)')
    plt.plot(t, history_est, 'b-', linewidth=2, label='KF Estimate')
    plt.title("Kalman Filter Tracking (Constant Velocity)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()
    
    # Plot 2: Uncertainty
    plt.subplot(2, 1, 2)
    plt.plot(t, np.sqrt(history_cov), 'k-', label='Std Dev (Position)')
    plt.title("Estimated Uncertainty (P matrix)")
    plt.xlabel("Time (s)")
    plt.ylabel("Sigma (m)")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()
```

---

## 🔬 Lab Exercise: Tuning Impact

### Lab Objectives
1.  Run the simulation. Observe how quickly the estimate converges to the truth.
2.  **Experiment 1:** Set $R = 1000$ (Very noisy sensor).
    -   *Expected:* The estimate will be very smooth but lag behind changes.
3.  **Experiment 2:** Set $Q = 0.0001$ (Very rigid model).
    -   *Expected:* The filter will refuse to believe the car changed speed.
4.  **Experiment 3:** Set $P_{init} = 0$ (Overconfidence).
    -   *Expected:* The filter will take a long time to correct the initial wrong guess because it thinks it's already right.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Filter Divergence
**Symptom:** Estimate flies away to infinity.
**Cause:**
-   Wrong signs in F or H.
-   dt is too large for the dynamics.
-   Numerical instability (P becomes non-positive-definite).

#### 2. Laggy Estimate
**Symptom:** Estimate follows the trend but is always delayed.
**Cause:** $R$ is too high relative to $Q$. The filter trusts the "old" prediction too much.

#### 3. Noisy Estimate
**Symptom:** Estimate jumps around almost as much as the raw sensor.
**Cause:** $Q$ is too high relative to $R$. The filter thinks the system is erratic.

---

## ⚡ Optimization & Best Practices

### 1. Matrix Inversion
The Kalman Gain requires inverting $S$.
$$ K = P H^T S^{-1} $$
In Python/C++, use `solve` instead of `inv` for stability:
`K = np.linalg.solve(S.T, (P @ H.T).T).T` (or similar optimized solvers).

### 2. Joseph Form
To ensure $P$ remains symmetric and positive definite, use the **Joseph Form** for the covariance update:
$$ P_{new} = (I - KH) P (I - KH)^T + K R K^T $$
This is numerically more stable than the standard form.

### 3. Steady State
If $F, H, Q, R$ are constant, $P$ and $K$ will converge to a constant value (Steady State Kalman Filter). You can pre-calculate $K$ and save CPU cycles!

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is the Kalman Filter called "Optimal"?
    *   **A:** It minimizes the Mean Squared Error (MSE) for linear systems with Gaussian noise.
2.  **Q:** What matrix determines how much we trust the sensor?
    *   **A:** $R$ (Measurement Noise Covariance).
3.  **Q:** If we measure velocity directly (Speedometer), how does $H$ change?
    *   **A:** $H = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ (Identity), assuming we measure both pos and vel. Or $\begin{bmatrix} 0 & 1 \end{bmatrix}$ if only velocity.

### Challenge Task
**Task:** 2D Tracker.
1.  State: $[x, y, v_x, v_y]$.
2.  Measurement: $[x, y]$ (GPS).
3.  Implement the 4x4 F matrix and 2x4 H matrix.
4.  Simulate a car driving in a circle.

---

## 📚 Further Reading & References
-   [Kalman Filter Explained Simply](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
-   [Greg Welch & Gary Bishop's Intro](https://www.cs.unc.edu/~welch/kalman/)

---

**Day 9 Complete** | Phase 4: ADAS & Robotics Systems | Week 2: Sensor Fusion & State Estimation
