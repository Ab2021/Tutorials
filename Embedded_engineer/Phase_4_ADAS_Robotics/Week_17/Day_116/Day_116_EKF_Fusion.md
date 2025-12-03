# Day 116: Extended Kalman Filter (EKF)
## Phase 4: ADAS & Robotics Systems | Week 17: Sensor Fusion

---

> **📝 Day 116 Focus:**
> The Linear Kalman Filter (KF) is great, but the world is curved. Radar measures in Polar coordinates $(\rho, \phi, \dot{\rho})$. Converting this to Cartesian is non-linear. The **Extended Kalman Filter (EKF)** uses **Jacobians** to linearize the world around the current estimate.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** why Linear KF fails for Radar.
2.  **Calculate** the Jacobian Matrix ($H_j$) for Radar measurements.
3.  **Implement** the EKF Predict-Update cycle.
4.  **Fuse** Lidar (Linear) and Radar (Non-Linear) data.
5.  **Visualize** the RMSE (Root Mean Square Error) reduction.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 115:** Matrix KF.
-   **Calculus:** Partial Derivatives ($\frac{\partial f}{\partial x}$).

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Non-Linear Problem

-   **State:** $x = [p_x, p_y, v_x, v_y]^T$.
-   **Lidar:** $z = [p_x, p_y]^T$. Linear ($z = Hx$). Easy.
-   **Radar:** $z = [\rho, \phi, \dot{\rho}]^T$.
    -   $\rho = \sqrt{p_x^2 + p_y^2}$
    -   $\phi = \arctan(p_y / p_x)$
    -   $\dot{\rho} = \frac{p_x v_x + p_y v_y}{\sqrt{p_x^2 + p_y^2}}$
-   This is **Non-Linear** ($z = h(x)$). We cannot write $z = Hx$.

### 🔹 Part 2: Taylor Series Linearization

We approximate the curve as a straight line (tangent) at the current point.
$$ h(x) \approx h(\mu) + \frac{\partial h}{\partial x}(x - \mu) $$
The slope $\frac{\partial h}{\partial x}$ is the **Jacobian Matrix** ($H_j$).

### 🔹 Part 3: The Radar Jacobian

We need derivatives of $\rho, \phi, \dot{\rho}$ with respect to $p_x, p_y, v_x, v_y$.

$$ H_j = \begin{bmatrix} 
\frac{\partial \rho}{\partial p_x} & \frac{\partial \rho}{\partial p_y} & 0 & 0 \\
\frac{\partial \phi}{\partial p_x} & \frac{\partial \phi}{\partial p_y} & 0 & 0 \\
\frac{\partial \dot{\rho}}{\partial p_x} & \frac{\partial \dot{\rho}}{\partial p_y} & \frac{\partial \dot{\rho}}{\partial v_x} & \frac{\partial \dot{\rho}}{\partial v_y}
\end{bmatrix} $$

(Derivation is messy, we will implement the result).

---

## 💻 Implementation: Sensor Fusion (Lidar + Radar)

**Scenario:**
-   Car moves in a curve.
-   Lidar updates: Linear Update (Standard KF).
-   Radar updates: Non-Linear Update (EKF).

### 🛠️ Setup
Create `week17_day116` and `ekf_fusion.py`.

```bash
mkdir -p ~/ros2_ws/src/week17_day116
cd ~/ros2_ws/src/week17_day116
touch ekf_fusion.py
```

### 👨‍💻 Code: EKF Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def calculate_jacobian(x_state):
    px, py, vx, vy = x_state.flatten()
    
    c1 = px**2 + py**2
    c2 = np.sqrt(c1)
    c3 = (c1 * c2)
    
    if c1 < 1e-4:
        return np.zeros((3, 4)) # Avoid division by zero
        
    Hj = np.array([
        [px/c2, py/c2, 0, 0],
        [-py/c1, px/c1, 0, 0],
        [py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2]
    ])
    return Hj

def cartesian_to_polar(x_state):
    px, py, vx, vy = x_state.flatten()
    rho = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    rho_dot = (px*vx + py*vy) / rho if rho > 1e-4 else 0
    return np.array([[rho], [phi], [rho_dot]])

class EKF:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 1000
        
        # F (Linear Motion Model - Constant Velocity)
        self.F = np.eye(4)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        
        # Q (Process Noise)
        noise_ax = 9.0
        noise_ay = 9.0
        dt2 = dt**2; dt3 = dt**3/2; dt4 = dt**4/4
        self.Q = np.array([
            [dt4*noise_ax, 0, dt3*noise_ax, 0],
            [0, dt4*noise_ay, 0, dt3*noise_ay],
            [dt3*noise_ax, 0, dt2*noise_ax, 0],
            [0, dt3*noise_ay, 0, dt2*noise_ay]
        ])
        
        # Measurement Matrices
        self.H_lidar = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R_lidar = np.array([[0.0225, 0], [0, 0.0225]])
        self.R_radar = np.array([[0.09, 0, 0], [0, 0.0009, 0], [0, 0, 0.09]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_lidar(self, z):
        y = z - self.H_lidar @ self.x
        S = self.H_lidar @ self.P @ self.H_lidar.T + self.R_lidar
        K = self.P @ self.H_lidar.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H_lidar) @ self.P

    def update_radar(self, z):
        # 1. Linearize Measurement Function
        Hj = calculate_jacobian(self.x)
        
        # 2. Calculate Residual (Innovation)
        # z is Polar, x is Cartesian. Convert x to Polar to compare.
        z_pred = cartesian_to_polar(self.x)
        y = z - z_pred
        
        # Normalize Angle (-pi to pi)
        while y[1] > np.pi: y[1] -= 2*np.pi
        while y[1] < -np.pi: y[1] += 2*np.pi
        
        # 3. Standard KF Update using Hj
        S = Hj @ self.P @ Hj.T + self.R_radar
        K = self.P @ Hj.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ Hj) @ self.P

def main():
    # Simulate Data
    dt = 0.1
    t = np.arange(0, 20, dt)
    
    # Ground Truth (Figure 8)
    gt_x = 10 * np.sin(t)
    gt_y = 10 * np.sin(t) * np.cos(t)
    # Velocities (approx)
    gt_vx = np.gradient(gt_x, dt)
    gt_vy = np.gradient(gt_y, dt)
    
    ekf = EKF(dt)
    est_x = []
    est_y = []
    
    for i in range(len(t)):
        ekf.predict()
        
        # Alternate Lidar/Radar
        if i % 2 == 0:
            # Lidar Measurement
            z = np.array([[gt_x[i]], [gt_y[i]]]) + np.random.normal(0, 0.15, (2, 1))
            ekf.update_lidar(z)
        else:
            # Radar Measurement
            px, py = gt_x[i], gt_y[i]
            vx, vy = gt_vx[i], gt_vy[i]
            rho = np.sqrt(px**2 + py**2)
            phi = np.arctan2(py, px)
            rho_dot = (px*vx + py*vy) / rho if rho > 0 else 0
            
            z = np.array([[rho], [phi], [rho_dot]]) + np.random.normal(0, 0.1, (3, 1))
            ekf.update_radar(z)
            
        est_x.append(ekf.x[0, 0])
        est_y.append(ekf.x[1, 0])
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(gt_x, gt_y, 'k--', label='Ground Truth')
    plt.plot(est_x, est_y, 'b-', label='EKF Fusion')
    plt.title("EKF Sensor Fusion (Lidar + Radar)")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Radar Advantage

### Lab Objectives
1.  Run the script.
2.  **Observation:** The EKF tracks the Figure-8 path smoothly.
3.  **Experiment:**
    -   Disable Lidar updates (Comment out `ekf.update_lidar`).
    -   **Result:** The track becomes much noisier (Radar angle resolution is poor).
    -   Disable Radar updates.
    -   **Result:** Position is good, but Velocity estimation lags (because Lidar doesn't measure velocity directly).
    -   **Conclusion:** Fusion gives the best of both worlds: Lidar's Position accuracy + Radar's Velocity accuracy.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Angle Normalization
**Symptom:** Filter goes crazy when crossing the X-axis (Angle jumps from $\pi$ to $-\pi$).
**Cause:** The residual $y = z - h(x)$ might be $3.1 - (-3.1) = 6.2$.
**Solution:** Always wrap angles to $[-\pi, \pi]$. `y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi`.

#### 2. Jacobian Division by Zero
**Symptom:** `RuntimeWarning: divide by zero`.
**Cause:** Target is at $(0,0)$. $\rho = 0$.
**Solution:** Check `if rho < 0.0001` and skip update or use a small epsilon.

---

## ⚡ Optimization & Best Practices

### 1. Unscented Kalman Filter (UKF)
EKF Linearization is an approximation. If the curve is sharp, it fails.
-   **UKF:** Instead of calculating Jacobians (Calculus), it picks "Sigma Points", passes them through the non-linear function, and calculates the mean/variance of the result.
-   **Pros:** No Jacobians! Better accuracy.
-   **Cons:** Slightly slower.

### 2. Coordinate Choice
Why not convert Radar to Cartesian first ($x = \rho \cos \phi$) and use Linear KF?
-   **Problem:** The noise becomes non-Gaussian (Banana shaped).
-   KF assumes Gaussian noise.
-   EKF handles the conversion inside the filter correctly.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the Jacobian?
    *   **A:** The matrix of first-order partial derivatives. It represents the local slope of a multi-dimensional function.
2.  **Q:** Why do we need `cartesian_to_polar` in `update_radar`?
    *   **A:** To calculate the residual $y = z - z_{pred}$. We must convert our state estimate (Cartesian) to the measurement space (Polar) to compare them.
3.  **Q:** Can EKF handle non-linear motion models too?
    *   **A:** Yes. If $x_{k} = f(x_{k-1})$, we calculate the Jacobian of $f$ ($F_j$) for the prediction step.

### Challenge Task
**Task:** UKF Research.
1.  Look up "Sigma Points".
2.  Understand how UKF avoids calculating derivatives.
3.  (Optional) Try implementing UKF using `filterpy` library.

---

## 📚 Further Reading & References
-   [Udacity CarND EKF Project](https://github.com/udacity/CarND-Extended-Kalman-Filter-Project)
-   [Kalman Filter Book (Roger Labbe)](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

---

**Day 116 Complete** | Phase 4: ADAS & Robotics Systems | Week 17: Sensor Fusion
