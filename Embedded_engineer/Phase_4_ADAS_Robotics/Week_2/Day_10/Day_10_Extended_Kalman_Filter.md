# Day 10: The Extended Kalman Filter (EKF)
## Phase 4: ADAS & Robotics Systems | Week 2: Sensor Fusion & State Estimation

---

> **📝 Day 10 Focus:**
> Real-world systems are rarely linear. Cars track curves, not lines. Radars measure range and bearing, not X and Y. The Linear Kalman Filter breaks down here. Today, we master the **Extended Kalman Filter (EKF)**, which uses **Jacobians** to linearize the world around the current estimate.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Diagnose** why the Linear KF fails for non-linear systems (e.g., Ackermann steering, Radar).
2.  **Calculate Jacobians** (First-order partial derivatives) to linearize non-linear functions.
3.  **Implement** the EKF Prediction and Update steps using Jacobians.
4.  **Develop** a full EKF in Python to fuse Lidar (Linear) and Radar (Non-linear) data.
5.  **Evaluate** the limitations of EKF (linearization error) and when to use UKF.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Calculus:** Partial Derivatives ($\frac{\partial f}{\partial x}$).
-   **Linear Algebra:** Jacobian Matrices.
-   **Day 9:** Linear KF equations.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows/Mac with Python).

### Software Stack
-   **Python Libraries:** `numpy`, `sympy` (for symbolic math), `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Non-Linear Problem

#### 1.1 Why Linear KF Fails
The Linear KF assumes:
$$ x_{k} = F x_{k-1} $$
$$ z_{k} = H x_{k} $$

**Real Car (Bicycle Model):**
$$ x_{k} = x_{k-1} + v \cos(\theta) \Delta t $$
$$ y_{k} = y_{k-1} + v \sin(\theta) \Delta t $$
$$ \theta_{k} = \theta_{k-1} + \frac{v}{L} \tan(\delta) \Delta t $$

This contains $\cos, \sin, \tan$. There is no matrix $F$ such that $x_k = F x_{k-1}$.
If we just plug these into a Linear KF, the Gaussian distribution gets "warped" into a banana shape, and the mean/variance calculation becomes wrong.

#### 1.2 Taylor Series Linearization
We can approximate a non-linear function $f(x)$ around a point $a$ using a Taylor Series:
$$ f(x) \approx f(a) + f'(a)(x-a) $$

In high dimensions, the derivative $f'(a)$ becomes the **Jacobian Matrix ($J$)**.

---

### 🔹 Part 2: The Jacobian Matrix

The Jacobian is the matrix of all first-order partial derivatives.

For a function $h(x)$ mapping state vector $x$ to measurement vector $z$:
$$ H_j = \frac{\partial h}{\partial x} = \begin{bmatrix} \frac{\partial h_1}{\partial x_1} & \cdots & \frac{\partial h_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial h_m}{\partial x_1} & \cdots & \frac{\partial h_m}{\partial x_n} \end{bmatrix} $$

**Example: Radar Measurement**
State: $x = [p_x, p_y, v_x, v_y]$
Measurement: $z = [\rho, \phi, \dot{\rho}]$ (Range, Bearing, Range Rate)

$$ \rho = \sqrt{p_x^2 + p_y^2} $$
$$ \phi = \text{atan2}(p_y, p_x) $$

The Jacobian $H_j$ is:
$$ H_j = \begin{bmatrix} \frac{p_x}{\sqrt{p_x^2+p_y^2}} & \frac{p_y}{\sqrt{p_x^2+p_y^2}} & 0 & 0 \\ \frac{-p_y}{p_x^2+p_y^2} & \frac{p_x}{p_x^2+p_y^2} & 0 & 0 \\ \dots & \dots & \dots & \dots \end{bmatrix} $$

---

### 🔹 Part 3: The EKF Algorithm

The EKF equations are almost identical to KF, but we use the non-linear functions for the state/measurement, and Jacobians for the covariance.

#### 3.1 Prediction Step
1.  **Project State:**
    $$ \hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k) $$
    *(Use the actual non-linear physics equations)*
2.  **Project Covariance:**
    $$ P_{k|k-1} = F_j P_{k-1|k-1} F_j^T + Q $$
    *(Use the Jacobian $F_j$ of the motion model)*

#### 3.2 Update Step
1.  **Innovation:**
    $$ \mathbf{y}_k = \mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1}) $$
    *(Use the actual non-linear measurement function)*
    *Critical: Normalize angles in $y$ (e.g., $\phi$) to be between $-\pi$ and $\pi$.*
2.  **Innovation Covariance:**
    $$ S_k = H_j P_{k|k-1} H_j^T + R $$
    *(Use the Jacobian $H_j$ of the measurement model)*
3.  **Kalman Gain:**
    $$ K_k = P_{k|k-1} H_j^T S_k^{-1} $$
4.  **Update State:**
    $$ \hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + K_k \mathbf{y}_k $$
5.  **Update Covariance:**
    $$ P_{k|k} = (I - K_k H_j) P_{k|k-1} $$

---

## 💻 Implementation: Sensor Fusion (Lidar + Radar)

We will track a car using:
1.  **Lidar:** Measures $[x, y]$ (Linear).
2.  **Radar:** Measures $[\rho, \phi, \dot{\rho}]$ (Non-linear).
3.  **Motion Model:** Constant Velocity (Linear for simplicity, to focus on Radar non-linearity).

### 🛠️ Setup
Create `week2_day10` and `ekf_fusion.py`.

```bash
mkdir -p ~/ros2_ws/src/week2_day10
cd ~/ros2_ws/src/week2_day10
touch ekf_fusion.py
```

### 👨‍💻 Code: EKF Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import math

class ExtendedKalmanFilter:
    def __init__(self):
        # State: [px, py, vx, vy]
        self.x = np.zeros((4, 1))
        
        # Covariance
        self.P = np.eye(4) * 1000
        
        # Process Noise
        self.Q = np.eye(4) * 0.1
        
        # Measurement Noise
        self.R_lidar = np.array([[0.0225, 0],
                                 [0, 0.0225]])
        self.R_radar = np.array([[0.09, 0, 0],
                                 [0, 0.0009, 0],
                                 [0, 0, 0.09]])
        
        # Lidar Matrix (Linear)
        self.H_lidar = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]])
                                 
        self.I = np.eye(4)

    def predict(self, dt):
        # Constant Velocity Model (Linear F)
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        
        # Update x
        self.x = F @ self.x
        
        # Update P
        self.P = F @ self.P @ F.T + self.Q

    def update_lidar(self, z):
        # Standard Linear Update
        y = z - self.H_lidar @ self.x
        S = self.H_lidar @ self.P @ self.H_lidar.T + self.R_lidar
        K = self.P @ self.H_lidar.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H_lidar) @ self.P

    def calculate_jacobian(self, x_state):
        px, py, vx, vy = x_state.flatten()
        
        c1 = px**2 + py**2
        c2 = math.sqrt(c1)
        c3 = (c1 * c2)
        
        if abs(c1) < 0.0001:
            return np.zeros((3, 4)) # Avoid division by zero
            
        Hj = np.array([
            [px/c2, py/c2, 0, 0],
            [-py/c1, px/c1, 0, 0],
            [py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2]
        ])
        return Hj

    def update_radar(self, z):
        # Non-Linear Update
        px, py, vx, vy = self.x.flatten()
        
        # h(x) - Convert state to polar
        rho = math.sqrt(px**2 + py**2)
        phi = math.atan2(py, px)
        rho_dot = (px*vx + py*vy) / rho if rho > 0.0001 else 0
        
        z_pred = np.array([[rho], [phi], [rho_dot]])
        
        y = z - z_pred
        
        # Normalize angle phi to [-pi, pi]
        while y[1] > math.pi: y[1] -= 2*math.pi
        while y[1] < -math.pi: y[1] += 2*math.pi
        
        # Calculate Jacobian
        Hj = self.calculate_jacobian(self.x)
        
        S = Hj @ self.P @ Hj.T + self.R_radar
        K = self.P @ Hj.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (self.I - K @ Hj) @ self.P

def run_simulation():
    # Generate Ground Truth (Figure 8)
    t = np.linspace(0, 20, 200)
    gt_x = 10 * np.sin(t)
    gt_y = 10 * np.sin(t/2)
    gt_vx = 10 * np.cos(t)
    gt_vy = 5 * np.cos(t/2)
    
    ekf = ExtendedKalmanFilter()
    # Initialize near truth
    ekf.x = np.array([[0], [0], [10], [5]])
    
    est_x = []
    est_y = []
    meas_x = []
    meas_y = []
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        
        # 1. Predict
        ekf.predict(dt)
        
        # 2. Update (Alternate Lidar/Radar)
        if i % 2 == 0:
            # Lidar Measurement
            z = np.array([[gt_x[i]], [gt_y[i]]]) + np.random.normal(0, 0.15, (2,1))
            ekf.update_lidar(z)
            meas_x.append(z[0,0])
            meas_y.append(z[1,0])
        else:
            # Radar Measurement
            px, py = gt_x[i], gt_y[i]
            vx, vy = gt_vx[i], gt_vy[i]
            rho = math.sqrt(px**2 + py**2)
            phi = math.atan2(py, px)
            rho_dot = (px*vx + py*vy) / rho
            
            z = np.array([[rho], [phi], [rho_dot]]) + \
                np.random.normal(0, [0.3, 0.03, 0.3]).reshape(3,1)
            
            ekf.update_radar(z)
            # Convert radar to cartesian for plotting
            meas_x.append(z[0]*math.cos(z[1]))
            meas_y.append(z[0]*math.sin(z[1]))
            
        est_x.append(ekf.x[0,0])
        est_y.append(ekf.x[1,0])
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(gt_x, gt_y, 'k-', label='Ground Truth')
    plt.plot(meas_x, meas_y, 'g.', alpha=0.3, label='Measurements')
    plt.plot(est_x, est_y, 'b-', linewidth=2, label='EKF Estimate')
    plt.title("EKF Sensor Fusion (Lidar + Radar)")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    run_simulation()
```

---

## 🔬 Lab Exercise: Jacobian Calculation

### Lab Objectives
1.  Derive the Jacobian for a different sensor model.
2.  **Scenario:** A Bearing-Only sensor (e.g., Camera detecting a landmark).
    $$ z = \text{atan2}(p_y, p_x) $$
3.  **Task:** Calculate $H_j = \frac{\partial z}{\partial x}$.
    *   Hint: $\frac{d}{dx} \arctan(u) = \frac{1}{1+u^2} \frac{du}{dx}$.

### Solution
$$ H_j = \begin{bmatrix} \frac{-p_y}{p_x^2 + p_y^2} & \frac{p_x}{p_x^2 + p_y^2} & 0 & 0 \end{bmatrix} $$

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Angle Normalization
**Symptom:** EKF goes crazy when the object crosses the X-axis (angle jumps from $\pi$ to $-\pi$).
**Cause:** The innovation $y = z - h(x)$ might be $3.1 - (-3.1) = 6.2$.
**Solution:** Always normalize angles in $y$ to $[-\pi, \pi]$.

#### 2. Division by Zero
**Symptom:** Crash when object is at $(0,0)$.
**Cause:** Jacobian calculation involves dividing by $\rho$ or $\rho^2$.
**Solution:** Check if $\rho < \epsilon$ and return zeros or skip update.

#### 3. Jacobian Errors
**Symptom:** Filter diverges slowly.
**Cause:** Math error in Jacobian derivation.
**Solution:** Use `sympy` to verify derivatives or use Numerical Differentiation (Finite Differences) to check your analytical Jacobian.

---

## ⚡ Optimization & Best Practices

### 1. Symbolic Math
Don't derive Jacobians by hand if you can avoid it. Use Python's `sympy`:
```python
from sympy import symbols, Matrix, sqrt, atan2
px, py, vx, vy = symbols('px py vx vy')
state = Matrix([px, py, vx, vy])
h = Matrix([sqrt(px**2 + py**2), atan2(py, px)])
H_j = h.jacobian(state)
print(H_j)
```

### 2. Unscented Kalman Filter (UKF)
If the non-linearity is very strong (e.g., highly dynamic drone maneuvers), the first-order approximation of EKF fails.
**UKF** uses "Sigma Points" to sample the distribution and pass them through the non-linear function, avoiding Jacobians entirely.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the Jacobian matrix?
    *   **A:** A matrix of first-order partial derivatives describing how the output changes w.r.t. the input.
2.  **Q:** Why do we need to normalize angles in the EKF update step?
    *   **A:** Because angles are cyclic. An error of $2\pi$ is actually 0 error, but the linear algebra treats it as a huge error.
3.  **Q:** Can EKF handle non-Gaussian noise?
    *   **A:** No. It still assumes Gaussian noise, just propagated through linearized functions. For non-Gaussian, use Particle Filters.

### Challenge Task
**Task:** Implement the Prediction Step for a Bicycle Model.
1.  State: $[x, y, v, \theta]$.
2.  Input: $[a, \delta]$ (Acceleration, Steering Angle).
3.  Derive $F_j$ for this model.

---

## 📚 Further Reading & References
-   [Kalman and Bayesian Filters in Python (Roger Labbe)](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) - Excellent free book.
-   [SymPy Documentation](https://docs.sympy.org/latest/index.html)

---

**Day 10 Complete** | Phase 4: ADAS & Robotics Systems | Week 2: Sensor Fusion & State Estimation
