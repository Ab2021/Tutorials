# Day 51: Kalman Filter (KF) & Extended Kalman Filter (EKF)
## Phase 4: ADAS & Robotics Systems | Week 8: State Estimation & Sensor Fusion

---

> **📝 Day 51 Focus:**
> The Bayes Filter is the theory. The **Kalman Filter** is the practice. It is the workhorse of guidance, navigation, and control (GNC). From Apollo 11 to Tesla Autopilot, the Kalman Filter (and its non-linear cousin, the **EKF**) is what keeps systems on track.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Derive** the Kalman Filter equations: Predict (Prior) and Update (Posterior).
2.  **Explain** the role of Covariance Matrices ($P, Q, R$) in sensor fusion.
3.  **Differentiate** between Linear (KF) and Non-Linear (EKF) systems.
4.  **Calculate** Jacobians for a non-linear motion model (CTRV).
5.  **Implement** an EKF in Python to fuse Odometry and GPS data.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Linear Algebra:** Matrix Multiplication, Inverse, Transpose.
-   **Calculus:** Partial Derivatives (Jacobians).

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Linear Kalman Filter (KF)

Assumptions:
1.  **Gaussian Noise:** Everything is a Bell Curve.
2.  **Linear System:** $x_{new} = A \cdot x_{old} + B \cdot u$.

**The State:**
-   $x$: Mean (Best estimate).
-   $P$: Covariance (Uncertainty matrix).

**The Algorithm:**
1.  **Predict:**
    -   $x' = Fx + Bu$ (Physics).
    -   $P' = FPF^T + Q$ (Uncertainty grows).
2.  **Update:**
    -   $y = z - Hx'$ (Error / Innovation).
    -   $S = H P' H^T + R$ (System Uncertainty).
    -   $K = P' H^T S^{-1}$ (Kalman Gain).
    -   $x = x' + Ky$ (Corrected Mean).
    -   $P = (I - KH)P'$ (Corrected Uncertainty).

### 🔹 Part 2: The Extended Kalman Filter (EKF)

Real robots are **Non-Linear**.
-   Example: $x = x + v \cdot \cos(\theta) \cdot dt$. (Cosine is non-linear).
-   KF fails here because transforming a Gaussian through a non-linear function results in a non-Gaussian.

**The Solution:** Linearize!
-   We approximate the curve as a straight line (Tangent) at the current point.
-   **Jacobian Matrix ($F_j, H_j$):** The matrix of partial derivatives.
-   $F_j = \frac{\partial f}{\partial x}$.

---

## 💻 Implementation: EKF for Fusion

**Scenario:**
-   **Robot:** Moves in a 2D plane.
-   **State:** $[x, y, \theta, v]$ (4D).
-   **Sensors:**
    -   **Odometry:** Measures $[v, \omega]$ (Control Input).
    -   **GPS:** Measures $[x, y]$ (Measurement).

### 🛠️ Setup
Create `week8_day51` and `ekf_fusion.py`.

```bash
mkdir -p ~/ros2_ws/src/week8_day51
cd ~/ros2_ws/src/week8_day51
touch ekf_fusion.py
```

### 👨‍💻 Code: EKF Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Configuration ---
DT = 0.1  # Time step
SIM_TIME = 50.0

# Noise Covariances
Q = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2  # Process Noise (Motion)
R = np.diag([1.0, 1.0]) ** 2  # Measurement Noise (GPS)

# --- Motion Model (Non-Linear) ---
def motion_model(x, u):
    # x: [x, y, theta, v]
    # u: [v, omega]
    
    F = np.eye(4)
    B = np.zeros((4, 2))
    
    theta = x[2, 0]
    v = x[3, 0]
    
    # x_{t+1} = x_t + v * cos(theta) * dt
    # y_{t+1} = y_t + v * sin(theta) * dt
    # theta_{t+1} = theta_t + omega * dt
    # v_{t+1} = v (Constant velocity model)
    
    x[0, 0] += v * math.cos(theta) * DT
    x[1, 0] += v * math.sin(theta) * DT
    x[2, 0] += u[1, 0] * DT
    x[3, 0] = u[0, 0]
    
    return x

def jacobian_F(x, u):
    # Jacobian of Motion Model w.r.t State x
    theta = x[2, 0]
    v = x[3, 0]
    
    jF = np.eye(4)
    jF[0, 2] = -v * math.sin(theta) * DT
    jF[0, 3] = math.cos(theta) * DT
    jF[1, 2] = v * math.cos(theta) * DT
    jF[1, 3] = math.sin(theta) * DT
    
    return jF

# --- Measurement Model (Linear) ---
def measurement_model(x):
    # GPS measures [x, y]
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    z = H @ x
    return z, H

def jacobian_H(x):
    # Linear measurement, Jacobian is just H
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

# --- EKF Algorithm ---
def ekf_predict(xEst, PEst, u):
    # 1. Predict State
    xPred = motion_model(xEst, u)
    
    # 2. Predict Covariance
    jF = jacobian_F(xEst, u)
    PPred = jF @ PEst @ jF.T + Q
    
    return xPred, PPred

def ekf_update(xPred, PPred, z):
    # 1. Innovation
    zPred, H = measurement_model(xPred)
    y = z - zPred
    
    # 2. Kalman Gain
    jH = jacobian_H(xPred)
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    
    # 3. Update State
    xEst = xPred + K @ y
    
    # 4. Update Covariance
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    
    return xEst, PEst

# --- Simulation ---
def main():
    time = 0.0
    
    # State Vector [x, y, theta, v]
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)
    
    # Dead Reckoning (No GPS)
    xDR = np.zeros((4, 1))
    
    # History
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xDR
    hz = np.zeros((2, 1))
    
    while time <= SIM_TIME:
        time += DT
        
        # 1. Control Input (v=1.0 m/s, omega=0.1 rad/s)
        u = np.array([[1.0], [0.1]])
        
        # 2. Ground Truth (Perfect Motion)
        xTrue = motion_model(xTrue, u)
        
        # 3. Measurement (Noisy GPS)
        z = np.array([
            [xTrue[0, 0] + np.random.randn() * 1.0], # GPS Noise x
            [xTrue[1, 0] + np.random.randn() * 1.0]  # GPS Noise y
        ])
        
        # 4. Dead Reckoning (Just Prediction)
        xDR = motion_model(xDR, u)
        
        # 5. EKF
        xPred, PPred = ekf_predict(xEst, PEst, u)
        xEst, PEst = ekf_update(xPred, PPred, z)
        
        # Store History
        hxEst = np.hstack((hxEst, xEst))
        hxTrue = np.hstack((hxTrue, xTrue))
        hxDR = np.hstack((hxDR, xDR))
        hz = np.hstack((hz, z))
        
        # Visualization
        if int(time * 10) % 10 == 0:
            plt.cla()
            # Plot GPS Measurements
            plt.plot(hz[0, :], hz[1, :], ".g", label="GPS")
            # Plot Ground Truth
            plt.plot(hxTrue[0, :], hxTrue[1, :], "-b", label="True Path")
            # Plot Dead Reckoning
            plt.plot(hxDR[0, :], hxDR[1, :], "-k", label="Dead Reckoning")
            # Plot EKF Estimate
            plt.plot(hxEst[0, :], hxEst[1, :], "-r", label="EKF")
            
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.pause(0.001)
            
    plt.show()

if __name__ == '__main__':
    main()
```

---

## 🔬 Lab Exercise: The Drift

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   **Blue Line (Truth):** A perfect circle.
    -   **Green Dots (GPS):** Scattered around the circle (Noisy).
    -   **Black Line (Dead Reckoning):** Starts okay, but drifts away (Integration error).
    -   **Red Line (EKF):** Smoothly follows the Blue line, filtering out the GPS noise and correcting the Dead Reckoning drift.
3.  **Experiment:**
    -   Increase GPS noise ($R$). The Red line will trust the Black line (Model) more.
    -   Increase Motion noise ($Q$). The Red line will trust the Green dots (GPS) more.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Matrix Singularity
**Symptom:** `LinAlgError: Singular matrix`.
**Cause:** Matrix inversion ($S^{-1}$) failed.
**Solution:** Ensure $R$ is not zero. Add a small epsilon to the diagonal if needed.

#### 2. Divergence
**Symptom:** Estimate flies away to infinity.
**Cause:**
-   Wrong Jacobian derivation.
-   $dt$ is too large (Linearization error).
-   Initial covariance $P$ is too small (Overconfident).

---

## ⚡ Optimization & Best Practices

### 1. Numerical Stability
The standard Kalman Filter equation $P = (I - KH)P$ is numerically unstable (can result in non-symmetric or negative-definite matrix).
**Joseph Form:** $P = (I - KH)P(I - KH)^T + KRK^T$.
-   Guarantees symmetry and positive-definiteness.

### 2. Unscented Kalman Filter (UKF)
Calculating Jacobians is hard and error-prone.
**UKF:** Uses "Sigma Points" (deterministic sampling) to approximate the non-linear transformation. No Jacobians needed! often more accurate than EKF.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the "Kalman Gain" ($K$)?
    *   **A:** A weighting factor (0 to 1). High $K$ means "Trust Measurement". Low $K$ means "Trust Prediction".
2.  **Q:** Why do we need Jacobians in EKF?
    *   **A:** To linearize the non-linear motion/measurement models so we can propagate the Gaussian covariance matrix.
3.  **Q:** What happens if $Q$ (Process Noise) is zero?
    *   **A:** The filter assumes the model is perfect. It will eventually ignore all measurements (Covariance collapse).

### Challenge Task
**Task:** Radar Fusion.
1.  Add a Radar sensor.
2.  Radar measures $[r, \phi]$ (Range, Bearing).
3.  Measurement Model: $r = \sqrt{x^2 + y^2}$, $\phi = \arctan2(y, x)$.
4.  Derive Jacobian $H_j$ for Radar.
5.  Fuse Radar data in the EKF.

---

## 📚 Further Reading & References
-   [Kalman Filter for Beginners](https://www.kalmanfilter.net/)
-   [PythonRobotics (Atsushi Sakai)](https://github.com/AtsushiSakai/PythonRobotics)

---

**Day 51 Complete** | Phase 4: ADAS & Robotics Systems | Week 8: State Estimation & Sensor Fusion
