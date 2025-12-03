# Day 115: Multi-Dimensional Kalman Filter
## Phase 4: ADAS & Robotics Systems | Week 17: Sensor Fusion

---

> **📝 Day 115 Focus:**
> Cars don't move in 1D. They move in 2D (X, Y) and have velocity ($v_x, v_y$). Today, we upgrade our Kalman Filter to handle **Vectors and Matrices**. We will implement a **Constant Velocity (CV)** model to track a vehicle.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the State Vector ($x$) and State Transition Matrix ($F$).
2.  **Construct** the Measurement Matrix ($H$) for Lidar (Position only).
3.  **Implement** the Matrix Kalman Filter equations.
4.  **Simulate** a 2D tracking scenario (Car driving in a circle).
5.  **Compare** Constant Velocity (CV) vs. Constant Acceleration (CA) models.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 114:** 1D Kalman Filter.
-   **Linear Algebra:** Matrix Multiplication, Inverse, Transpose.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `numpy` (Heavy usage).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The State Vector

Instead of a single number, our state is a vector:
$$ \mathbf{x} = \begin{bmatrix} p_x \\ p_y \\ v_x \\ v_y \end{bmatrix} $$

### 🔹 Part 2: The Motion Model (Prediction)

Physics: $p_{new} = p_{old} + v \cdot \Delta t$.
$$ \mathbf{x}_{k} = F \mathbf{x}_{k-1} + \mathbf{w} $$
$$ F = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} $$
-   This matrix says: "New Position = Old Position + Velocity * dt". "New Velocity = Old Velocity". (Constant Velocity).

### 🔹 Part 3: The Measurement Model (Update)

Lidar gives us Position $(p_x, p_y)$, but not Velocity.
$$ \mathbf{z} = H \mathbf{x} + \mathbf{v} $$
$$ H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} $$
-   This matrix extracts the first two rows (Position) from the state vector.

---

## 💻 Implementation: 2D Tracker

**Scenario:**
-   Car driving in a circle.
-   Lidar provides noisy $(x, y)$.
-   KF estimates $(x, y, v_x, v_y)$.

### 🛠️ Setup
Create `week17_day115` and `kalman_2d.py`.

```bash
mkdir -p ~/ros2_ws/src/week17_day115
cd ~/ros2_ws/src/week17_day115
touch kalman_2d.py
```

### 👨‍💻 Code: Matrix Kalman Filter

```python
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter2D:
    def __init__(self, dt, initial_x, initial_P, process_noise, meas_noise):
        self.dt = dt
        self.x = initial_x # State [x, y, vx, vy] (4x1)
        self.P = initial_P # Covariance (4x4)
        
        # F: State Transition
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # H: Measurement Function (Lidar measures x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Q: Process Noise Covariance
        # Assume acceleration is the noise source
        # Discrete White Noise Acceleration Model
        # q_var is variance of acceleration
        q_var = process_noise
        dt2 = dt**2
        dt3 = dt**3 / 2
        dt4 = dt**4 / 4
        
        self.Q = np.array([
            [dt4, 0, dt3, 0],
            [0, dt4, 0, dt3],
            [dt3, 0, dt2, 0],
            [0, dt3, 0, dt2]
        ]) * q_var
        
        # R: Measurement Noise Covariance
        self.R = np.eye(2) * meas_noise

    def predict(self):
        # x = Fx
        self.x = self.F @ self.x
        # P = FPF' + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        # z is [x, y] (2x1)
        
        # y = z - Hx (Residual)
        y = z - self.H @ self.x
        
        # S = HPH' + R (Residual Covariance)
        S = self.H @ self.P @ self.H.T + self.R
        
        # K = P H' S^-1 (Kalman Gain)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # x = x + Ky
        self.x = self.x + K @ y
        
        # P = (I - KH)P
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x

def generate_ground_truth(n_steps, dt):
    # Circle Path
    t = np.linspace(0, n_steps*dt, n_steps)
    radius = 50.0
    omega = 0.1 # rad/s
    
    x = radius * np.cos(omega * t)
    y = radius * np.sin(omega * t)
    
    # Velocity (Derivative)
    vx = -radius * omega * np.sin(omega * t)
    vy = radius * omega * np.cos(omega * t)
    
    return np.vstack((x, y, vx, vy)).T

def main():
    dt = 0.1
    n_steps = 200
    
    # Ground Truth
    gt_data = generate_ground_truth(n_steps, dt)
    
    # Measurements (Add Noise)
    meas_noise_std = 2.0
    measurements = gt_data[:, 0:2] + np.random.normal(0, meas_noise_std, (n_steps, 2))
    
    # Initialize KF
    # Start at (0,0) with 0 velocity (Bad guess)
    x0 = np.zeros((4, 1))
    P0 = np.eye(4) * 1000.0 # High uncertainty
    
    kf = KalmanFilter2D(dt, x0, P0, process_noise=0.1, meas_noise=meas_noise_std**2)
    
    est_path = []
    
    for i in range(n_steps):
        # 1. Predict
        kf.predict()
        
        # 2. Update
        z = measurements[i].reshape(2, 1)
        kf.update(z)
        
        est_path.append(kf.x.flatten())
        
    est_path = np.array(est_path)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(gt_data[:, 0], gt_data[:, 1], 'k--', label='Ground Truth')
    plt.scatter(measurements[:, 0], measurements[:, 1], c='r', s=10, alpha=0.3, label='Lidar Meas')
    plt.plot(est_path[:, 0], est_path[:, 1], 'b-', linewidth=2, label='KF Estimate')
    
    # Start/End
    plt.plot(gt_data[0,0], gt_data[0,1], 'go', label='Start')
    plt.plot(gt_data[-1,0], gt_data[-1,1], 'rx', label='End')
    
    plt.title("2D Kalman Filter Tracking (Constant Velocity)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Turn

### Lab Objectives
1.  Run the script.
2.  **Observation:** The KF tracks the circle well, but "cuts the corner" slightly.
    -   This is because we used a **Constant Velocity (CV)** model ($F$ assumes straight line).
    -   A circle is constant *acceleration* (centripetal).
3.  **Experiment:**
    -   Increase `process_noise` (Q).
    -   **Result:** The filter trusts the measurements more, reducing the corner-cutting, but becoming noisier.
    -   **Lesson:** $Q$ tells the filter "My motion model is imperfect".

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Dimension Mismatch
**Symptom:** `ValueError: shapes (4,4) and (2,1) not aligned`.
**Cause:** Matrix multiplication rules.
**Solution:** Always check shapes. $H$ is $2 \times 4$. $x$ is $4 \times 1$. $Hx$ is $2 \times 1$.

#### 2. Numerical Instability
**Symptom:** $P$ becomes non-symmetric or negative.
**Cause:** Floating point errors in repeated subtractions.
**Solution:** Enforce symmetry: `P = (P + P.T) / 2`. Use Joseph Form for update (more stable).

---

## ⚡ Optimization & Best Practices

### 1. Constant Turn Rate and Velocity (CTRV)
For cars, CV is bad. CA is okay. **CTRV** is best.
-   State: $[x, y, v, \psi, \dot{\psi}]$. (Yaw and Yaw Rate).
-   Problem: The motion model becomes **Non-Linear** ($x_{new} = x + v \cos(\psi) dt$).
-   Solution: **Extended Kalman Filter (EKF)**. (Day 116).

### 2. Matrix Libraries
In Python, `numpy` is fast.
In C++, use `Eigen`. It uses SIMD instructions for blazing fast matrix math.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is $H$ a matrix?
    *   **A:** Because we need to map the 4D state space to the 2D measurement space. It acts as a "Selector".
2.  **Q:** What happens to Velocity if we only measure Position?
    *   **A:** The KF infers velocity! $v \approx (p_k - p_{k-1}) / dt$. The correlation in the $P$ matrix allows this information to flow from Position to Velocity.
3.  **Q:** What is the size of $K$?
    *   **A:** $4 \times 2$. It maps the 2D residual (Position error) to a 4D correction (Pos + Vel correction).

### Challenge Task
**Task:** Radar Update.
1.  Radar measures $[x, y, v_x, v_y]$ (Doppler gives velocity!).
2.  Change $H$ to $4 \times 4$ Identity matrix.
3.  Update `R` to be $4 \times 4$.
4.  Observe how much faster the velocity converges.

---

## 📚 Further Reading & References
-   [Kalman Filter Matrix Cheatsheet](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/08-Designing-Kalman-Filters.ipynb)
-   [Eigen Library (C++)](https://eigen.tuxfamily.org/)

---

**Day 115 Complete** | Phase 4: ADAS & Robotics Systems | Week 17: Sensor Fusion
