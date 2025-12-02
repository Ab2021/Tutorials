# Day 11: The Unscented Kalman Filter (UKF)
## Phase 4: ADAS & Robotics Systems | Week 2: Sensor Fusion & State Estimation

---

> **📝 Day 11 Focus:**
> While the EKF is powerful, calculating Jacobians is tedious and error-prone. Worse, for highly non-linear systems, the first-order approximation can diverge. The **Unscented Kalman Filter (UKF)** solves this by using a deterministic sampling approach called the **Unscented Transform** to propagate probability distributions through non-linear functions without linearization.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Contrast** the EKF (Linearization) with the UKF (Unscented Transform) and understand when to use which.
2.  **Generate Sigma Points** and Weights using the Merwe Scaled Sigma Point algorithm.
3.  **Implement** the Unscented Transform to predict mean and covariance after a non-linear transformation.
4.  **Develop** a full UKF in Python to track a vehicle using the CTRV (Constant Turn Rate and Velocity) model.
5.  **Compare** EKF and UKF performance on a highly non-linear tracking problem.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 10:** EKF and its limitations.
-   **Statistics:** Mean and Covariance calculation from samples.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows/Mac with Python).

### Software Stack
-   **Python Libraries:** `numpy`, `scipy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Linearization Problem

**EKF Approach:**
1.  Linearize the function $f(x)$ at the mean $\mu$ (Tangent line).
2.  Transform the Gaussian using this linear function.
3.  **Issue:** If the function curves significantly within the uncertainty region ($\sigma$), the transformed mean is *biased* and the covariance is *underestimated*.

**UKF Approach:**
"It is easier to approximate a probability distribution than it is to approximate an arbitrary non-linear function." — *Julier & Uhlmann*

1.  Select a minimal set of sample points (**Sigma Points**) that capture the mean and covariance of the distribution.
2.  Pass each point through the *actual* non-linear function $f(x)$.
3.  Calculate the new mean and covariance from the transformed points.

### 🔹 Part 2: The Unscented Transform

#### 2.1 Sigma Point Selection
For an $n$-dimensional state, we choose $2n+1$ points.

1.  **Mean:** $\mathcal{X}_0 = \mu$
2.  **Positive Direction:** $\mathcal{X}_i = \mu + (\sqrt{(n+\lambda)\Sigma})_i$ for $i=1..n$
3.  **Negative Direction:** $\mathcal{X}_{i+n} = \mu - (\sqrt{(n+\lambda)\Sigma})_i$ for $i=1..n$

Where:
-   $\lambda = \alpha^2(n+\kappa) - n$ is a scaling parameter.
-   $\alpha$ determines the spread (usually small, e.g., $10^{-3}$).
-   $\kappa$ is a secondary scaling parameter (usually $0$ or $3-n$).
-   $\sqrt{\Sigma}$ is the Cholesky decomposition of the covariance matrix.

#### 2.2 Weights
Each Sigma Point has a weight for the Mean ($W^m$) and Covariance ($W^c$).

-   $W_0^m = \frac{\lambda}{n+\lambda}$
-   $W_0^c = \frac{\lambda}{n+\lambda} + (1 - \alpha^2 + \beta)$
-   $W_i^m = W_i^c = \frac{1}{2(n+\lambda)}$ for $i=1..2n$

$\beta$ incorporates prior knowledge of the distribution (for Gaussian, $\beta=2$ is optimal).

### 🔹 Part 3: The UKF Algorithm

#### 3.1 Prediction Step
1.  **Generate Sigma Points:** $\mathcal{X}_{k-1}$ based on $\hat{x}_{k-1}$ and $P_{k-1}$.
2.  **Predict Sigma Points:** Pass each through the motion model.
    $$ \mathcal{Y}_i = f(\mathcal{X}_i, u_k) $$
3.  **Predict Mean:** Weighted sum of transformed points.
    $$ \hat{x}_{k|k-1} = \sum W_i^m \mathcal{Y}_i $$
4.  **Predict Covariance:** Weighted sum of squared differences + Process Noise $Q$.
    $$ P_{k|k-1} = \sum W_i^c (\mathcal{Y}_i - \hat{x}_{k|k-1})(\mathcal{Y}_i - \hat{x}_{k|k-1})^T + Q $$

#### 3.2 Update Step
1.  **Project to Measurement Space:** Pass predicted sigma points through measurement model.
    $$ \mathcal{Z}_i = h(\mathcal{Y}_i) $$
2.  **Predicted Measurement Mean:**
    $$ \hat{z}_k = \sum W_i^m \mathcal{Z}_i $$
3.  **Innovation Covariance ($S$):**
    $$ S_k = \sum W_i^c (\mathcal{Z}_i - \hat{z}_k)(\mathcal{Z}_i - \hat{z}_k)^T + R $$
4.  **Cross Covariance ($T$):** Correlation between State and Measurement.
    $$ T_k = \sum W_i^c (\mathcal{Y}_i - \hat{x}_{k|k-1})(\mathcal{Z}_i - \hat{z}_k)^T $$
5.  **Kalman Gain:**
    $$ K_k = T_k S_k^{-1} $$
6.  **Update:**
    $$ \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - \hat{z}_k) $$
    $$ P_{k|k} = P_{k|k-1} - K_k S_k K_k^T $$

---

## 💻 Implementation: UKF for CTRV Model

We will track a car using the **Constant Turn Rate and Velocity (CTRV)** model, which is highly non-linear.

**State:** $x = [p_x, p_y, v, \psi, \dot{\psi}]$
(Position X, Y, Velocity, Yaw, Yaw Rate)

### 🛠️ Setup
Create `week2_day11` and `ukf_ctrv.py`.

```bash
mkdir -p ~/ros2_ws/src/week2_day11
cd ~/ros2_ws/src/week2_day11
touch ukf_ctrv.py
```

### 👨‍💻 Code: UKF Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

class UKF:
    def __init__(self, dim_x, dim_z, dt):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        
        # Sigma Point Parameters
        self.alpha = 0.001
        self.beta = 2.0
        self.kappa = 0.0
        self.n = dim_x
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        
        # Weights
        self.Wm = np.zeros(2*self.n + 1)
        self.Wc = np.zeros(2*self.n + 1)
        
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2*self.n + 1):
            self.Wm[i] = 1.0 / (2 * (self.n + self.lambda_))
            self.Wc[i] = self.Wm[i]
            
        # State
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        
        self.sigmas_f = np.zeros((2*self.n + 1, self.dim_x))

    def generate_sigma_points(self, x, P):
        sigmas = np.zeros((2*self.n + 1, self.n))
        sigmas[0] = x
        
        # Numerical stability check for Cholesky
        try:
            L = cholesky((self.n + self.lambda_) * P, lower=True)
        except np.linalg.LinAlgError:
            # If P is not positive definite (due to numerical errors), fix it
            print("Warning: P not PD, fixing...")
            P = (P + P.T) / 2 + np.eye(self.n) * 1e-6
            L = cholesky((self.n + self.lambda_) * P, lower=True)
            
        for i in range(self.n):
            sigmas[i+1] = x + L[:, i]
            sigmas[self.n+i+1] = x - L[:, i]
            
        return sigmas

    def predict(self, fx):
        # 1. Generate Sigma Points
        sigmas = self.generate_sigma_points(self.x, self.P)
        
        # 2. Pass through process model
        self.sigmas_f = np.zeros_like(sigmas)
        for i in range(2*self.n + 1):
            self.sigmas_f[i] = fx(sigmas[i], self.dt)
            
        # 3. Predict Mean
        self.x = np.dot(self.Wm, self.sigmas_f)
        
        # 4. Predict Covariance
        self.P = np.zeros((self.n, self.n))
        for i in range(2*self.n + 1):
            y = self.sigmas_f[i] - self.x
            # Normalize angle (yaw)
            y[3] = (y[3] + np.pi) % (2 * np.pi) - np.pi
            self.P += self.Wc[i] * np.outer(y, y)
        self.P += self.Q

    def update(self, z, hx):
        # 1. Pass predicted sigma points through measurement model
        sigmas_h = np.zeros((2*self.n + 1, self.dim_z))
        for i in range(2*self.n + 1):
            sigmas_h[i] = hx(self.sigmas_f[i])
            
        # 2. Predict Measurement Mean
        zp = np.dot(self.Wm, sigmas_h)
        
        # 3. Calculate S and Cross Covariance T
        S = np.zeros((self.dim_z, self.dim_z))
        T = np.zeros((self.dim_x, self.dim_z))
        
        for i in range(2*self.n + 1):
            z_diff = sigmas_h[i] - zp
            # Normalize angle if measurement includes angle (e.g. Radar)
            if self.dim_z == 3: # Radar [rho, phi, rho_dot]
                z_diff[1] = (z_diff[1] + np.pi) % (2 * np.pi) - np.pi
                
            S += self.Wc[i] * np.outer(z_diff, z_diff)
            
            x_diff = self.sigmas_f[i] - self.x
            x_diff[3] = (x_diff[3] + np.pi) % (2 * np.pi) - np.pi
            
            T += self.Wc[i] * np.outer(x_diff, z_diff)
            
        S += self.R
        
        # 4. Kalman Gain
        K = np.dot(T, np.linalg.inv(S))
        
        # 5. Update
        y = z - zp
        if self.dim_z == 3:
            y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
            
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(S, K.T))

# --- Models ---

def ctrv_model(x, dt):
    # x = [px, py, v, yaw, yaw_rate]
    px, py, v, yaw, yawd = x
    
    if abs(yawd) > 0.001:
        px_p = px + v/yawd * (np.sin(yaw + yawd*dt) - np.sin(yaw))
        py_p = py + v/yawd * (np.cos(yaw) - np.cos(yaw + yawd*dt))
    else:
        px_p = px + v*dt*np.cos(yaw)
        py_p = py + v*dt*np.sin(yaw)
        
    v_p = v
    yaw_p = yaw + yawd*dt
    yawd_p = yawd
    
    return np.array([px_p, py_p, v_p, yaw_p, yawd_p])

def radar_measurement(x):
    # x = [px, py, v, yaw, yawd]
    px, py, v, yaw, yawd = x
    
    rho = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    rho_dot = (px*v*np.cos(yaw) + py*v*np.sin(yaw)) / rho if rho > 0.0001 else 0
    
    return np.array([rho, phi, rho_dot])

def run_simulation():
    # Setup
    ukf = UKF(dim_x=5, dim_z=3, dt=0.1)
    
    # Process Noise (Tuned)
    std_a = 1.5 # Acceleration noise
    std_yawdd = 0.5 # Yaw acceleration noise
    ukf.Q = np.diag([0, 0, 1, 0, 1]) # Simplified Q
    
    # Measurement Noise (Radar)
    ukf.R = np.diag([0.09, 0.0009, 0.09])
    
    # Initial State
    ukf.x = np.array([0, 0, 5, 0, 0.1])
    ukf.P *= 1.0
    
    # Simulation Loop
    history_est = []
    
    # Ground Truth: Car turning
    # ... (Similar to Day 10 simulation code) ...
    # For brevity, let's just run a few steps
    
    print("Running UKF Simulation...")
    for i in range(50):
        # Fake Measurement
        z = np.array([10 + i*0.1, 0.1 + i*0.01, 5]) 
        
        ukf.predict(ctrv_model)
        ukf.update(z, radar_measurement)
        
        history_est.append(ukf.x)
        
    history_est = np.array(history_est)
    
    plt.figure()
    plt.plot(history_est[:,0], history_est[:,1], label='UKF Path')
    plt.title("UKF Tracking (CTRV Model)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_simulation()
```

---

## 🔬 Lab Exercise: EKF vs UKF

### Lab Objectives
1.  Run the UKF simulation.
2.  Compare the code complexity with Day 10 (EKF).
    -   *Observation:* UKF requires NO Jacobian derivation!
3.  **Experiment:** Increase the non-linearity.
    -   Simulate a car doing aggressive S-turns.
    -   Compare how well EKF and UKF track the velocity vector.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Cholesky Decomposition Failure
**Symptom:** `np.linalg.LinAlgError: Matrix is not positive definite`.
**Cause:**
-   Numerical instability makes $P$ slightly asymmetric or negative eigenvalues.
-   $Q$ or $R$ are too small.
**Solution:**
-   Enforce symmetry: `P = (P + P.T) / 2`.
-   Add small epsilon to diagonal: `P += eye(n) * 1e-9`.

#### 2. Angle Wrapping
**Symptom:** Filter jumps when crossing $\pm \pi$.
**Solution:** Ensure angle normalization is applied in:
-   Covariance calculation (State difference).
-   Innovation calculation (Measurement difference).
-   Cross-covariance calculation.

#### 3. Sigma Point Collapse
**Symptom:** All sigma points converge to a single point.
**Cause:** $\lambda$ is negative and too large magnitude.
**Solution:** Check Merwe scaling parameters ($\alpha, \beta, \kappa$).

---

## ⚡ Optimization & Best Practices

### 1. Square Root UKF (SR-UKF)
Instead of propagating the Covariance Matrix $P$, propagate its square root $S$ (where $P = SS^T$).
-   Avoids taking Cholesky at every step (expensive).
-   Numerically more stable (guarantees positive semi-definiteness).
-   Uses QR decomposition for updates.

### 2. Tuning Alpha
-   $\alpha$ controls the spread of sigma points.
-   Small $\alpha$ ($10^{-3}$) keeps points close to mean (good for local non-linearities).
-   Large $\alpha$ ($1$) spreads them out (captures global structure but risks sampling invalid regions).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the main advantage of UKF over EKF?
    *   **A:** No Jacobians required, and better accuracy (3rd order) for non-linear systems.
2.  **Q:** What is the computational cost difference?
    *   **A:** UKF is generally slower ($2n+1$ function evaluations) compared to EKF (1 evaluation + Jacobian calculation).
3.  **Q:** What are Sigma Points?
    *   **A:** Deterministic samples chosen to capture the mean and covariance of the distribution.

### Challenge Task
**Task:** Implement the "Augmented State" UKF.
1.  Include Process Noise in the state vector: $x_{aug} = [x, \nu_a, \nu_{\psi}]$.
2.  Generate Sigma Points for the augmented state (Dimension $n+2$).
3.  This captures the non-linear effect of noise (e.g., noise entering through the `cos/sin` terms).

---

## 📚 Further Reading & References
-   [The Unscented Kalman Filter for Nonlinear Estimation (Wan & Van der Merwe)](https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf) - The original paper.
-   [FilterPy Library](https://github.com/rlabbe/filterpy) - Production-grade Python filters.

---

**Day 11 Complete** | Phase 4: ADAS & Robotics Systems | Week 2: Sensor Fusion & State Estimation
