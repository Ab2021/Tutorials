# Day 10: Sensor Fusion (EKF/UKF)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 2: Advanced SLAM & State Estimation

---

> **📝 Content Creator Instructions:**
> No single sensor is perfect. GPS fails in tunnels. IMU drifts. Odometry slips.
> - **Focus:** Mathematical derivation of EKF, handling non-linearity, and production ROS 2 integration.
> - **Code:** Implementation of EKF from scratch and configuration of `robot_localization`.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Derive** the Predict (Motion Model) and Update (Measurement Model) equations for the Kalman Filter.
2.  **Explain** why Linear KF fails on robotic systems (e.g., differential drive motion is non-linear trigonometric).
3.  **Implement** an Extended Kalman Filter (EKF) processing IMU and GPS data.
4.  **Configure** the `robot_localization` ROS 2 package for sensor fusion.
5.  **Analyze** Covariance Matrices to understand uncertainty propagation.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- IMU (MPU6050) and GPS (Neo-6m) modules connected to a microcontroller (optional).
- Or use Gazebo simulation.

### Software Environment
```bash
sudo apt install ros-humble-robot-localization
pip install numpy matplotlib filterpy
```

### Prior Knowledge
- Gaussian Distributions (Mean $\mu$ and Covariance $\Sigma$).
- Jacobians (Partial Derivatives).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Recursive Bayes Filter

We want to estimate the state $x_k$ (e.g., Position, Velocity) given measurements $z_k$.
$$ P(x_k | z_{1:k}) \propto P(z_k | x_k) \cdot P(x_k | z_{1:k-1}) $$
*   **Posterior:** Probability after seeing measurement.
*   **Likelihood:** Probability of measurement given state (Sensor Model).
*   **Prior:** Probability predicted from previous state (Motion Model).

#### 1.1 The Kalman Filter (Linear)
If Motion $F$ and Measurement $H$ are **Linear** and noise is **Gaussian**:
1.  **Predict:**
    $$ \hat{x}_{k}^- = F \hat{x}_{k-1} + B u_k $$
    $$ P_k^- = F P_{k-1} F^T + Q $$
2.  **Update:**
    $$ K_k = P_k^- H^T (H P_k^- H^T + R)^{-1} $$
    $$ \hat{x}_k = \hat{x}_k^- + K_k (z_k - H \hat{x}_k^-) $$
    $$ P_k = (I - K_k H) P_k^- $$

*   $Q$: Process Noise Covariance (How perfect is my physics model?).
*   $R$: Measurement Noise Covariance (How perfect is my sensor?).
*   $K$: Kalman Gain (Trust sensor vs. trust physics?).

### 🔹 Part 2: Extended Kalman Filter (EKF)

Robots move in arcs ($x = v \cos \theta$). This is **Non-Linear**.
$$ x_{k} = f(x_{k-1}, u_k) $$
$$ z_{k} = h(x_{k}) $$

**The EKF Trick:** Linearize $f$ and $h$ using **Jacobians** (First-order Taylor expansion) around the current estimate.

*   $F_k = \frac{\partial f}{\partial x} |_{\hat{x}_{k-1}}$
*   $H_k = \frac{\partial h}{\partial x} |_{\hat{x}_{k}^-}$

**Why Jacobians?**
A Gaussian passed through a non-linear function is no longer Gaussian. We approximate the function as a straight line (slope = Jacobian) locally so the output remains Gaussian.

### 🔹 Part 3: Unscented Kalman Filter (UKF)

Linearization causes error (bias). UKF uses **Sigma Points**.
1.  Sample specific points around the mean $\mu$.
2.  Pass *points* through the non-linear function $f(x)$.
3.  Recover new Gaussian $\mu', \Sigma'$ from the transformed points.
*   *Pros:* No Jacobians needed! Better accuracy for highly non-linear systems.
*   *Cons:* Computationally slightly heavier.

---

## 💻 Implementation: EKF 2D Localization

We will fusion Odometer ($v, \omega$) and GPS ($x, y$).

### 🛠️ Project Structure
```text
day10_fusion/
├── data/
│   └── sensor_log.csv
├── src/
│   ├── ekf.py
│   └── plot_results.py
└── run_fusion.py
```

### 👨‍💻 Code Implementation (`src/ekf.py`)

```python
import numpy as np

class EKF:
    def __init__(self):
        # State: [x, y, theta]
        self.x = np.zeros((3, 1)) 
        
        # Covariance: High uncertainty initially
        self.P = np.eye(3) * 1000 
        
        # Process Noise
        self.Q = np.diag([0.1, 0.1, 0.05])
        
        # Measurement Noise (GPS accuracy ~1.0m)
        self.R = np.diag([1.0, 1.0])
        
    def predict(self, v, w, dt):
        """
        Motion Model: Differential Drive
        x' = x + v*cos(theta)*dt
        y' = y + v*sin(theta)*dt
        theta' = theta + w*dt
        """
        theta = self.x[2, 0]
        
        # 1. State Prediction (Non-linear)
        self.x[0, 0] += v * np.cos(theta) * dt
        self.x[1, 0] += v * np.sin(theta) * dt
        self.x[2, 0] += w * dt
        
        # 2. Jacobian F (Partial f / Partial x)
        # d(x')/d(theta) = -v*sin(theta)*dt
        # d(y')/d(theta) = v*cos(theta)*dt
        F = np.eye(3)
        F[0, 2] = -v * np.sin(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt
        
        # 3. Covariance Prediction
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, z):
        """
        Measurement Model: GPS Direct Observation
        z = [x_gps, y_gps]
        """
        # H is linear (Observation Matrix)
        H = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        
        # Innovation
        y = z - H @ self.x
        
        # Innovation Covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State Update
        self.x = self.x + K @ y
        
        # Covariance Update
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P
        
        return self.x, self.P
```

---

## 🔬 Lab Exercise: Robot Localization Package

### 1. Lab Objectives
- Use the standard `robot_localization` (RL) package to fuse Noisy Odometry + Noisy IMU.
- Observe how fusion reduces drift compared to raw odometry.

### 2. Step-by-Step Guide

#### Phase A: Configuration (`ekf.yaml`)
This is the most critical skill for a ROS engineer.

```yaml
ekf_filter_node:
  ros__parameters:
    frequency: 30.0
    two_d_mode: true
    publish_tf: true
    
    # Frame Definition
    map_frame: map
    odom_frame: odom
    base_link_frame: base_link
    world_frame: odom # Local Fusion
    
    # Sensors
    # Odom0: Wheel Encoders (x, y, yaw, vx, vy, vyaw)
    odom0: /odom
    odom0_config: [false, false, false, # XYZ
                   false, false, false, # RPY
                   true, true, false,  # Vx, Vy, Vz
                   false, false, true, # Vroll, Vpitch, Vyaw
                   false, false, false] # Ax, Ay, Az
    odom0_differential: false

    # Imu0: Gyroscope (yaw velocity) + Accelerometer (acceleration)
    imu0: /imu/data
    imu0_config: [false, false, false,
                  false, false, true, # Yaw (Absolute orientation? Maybe drifting)
                  false, false, false,
                  false, false, true, # Vyaw
                  true, false, false] # Ax (Linear Accel)
    imu0_differential: false
    imu0_remove_gravitational_acceleration: true
```

#### Phase B: Launch
Launch the node with the config.
```bash
ros2 launch robot_localization ekf.launch.py ekf_config:=path/to/ekf.yaml
```

#### Phase C: Analysis (PlotJuggler)
1.  Run `ros2 run plotjuggler plotjuggler`.
2.  Subscribe to `/odom` (raw) and `/odometry/filtered` (result).
3.  **Observation:** The `/odometry/filtered` line is smoother and, if ground truth is available, closer to reality.

---

## 🚀 Project: "Inertial Odometry Drifting"

**Goal:** Simulate an IMU-only odometry system and fix it with periodic GPS updates.
1.  **Pure Integration:** Integrate $a \to v \to x$. Watch it drift exponentially ($t^2$ error).
2.  **Fusion:** Add a "Fake GPS" update every 5 seconds.
3.  **Result:** Sawtooth error pattern. Uncertainty grows ($\Sigma$ expands) between GPS hits, then collapses ($\Sigma$ shrinks) on update.

### 1. Visualization
Draw uncertainty ellipses (Covariance) on the trajectory map.
*   **Large Ellipse:** "I don't know where I am."
*   **Small Ellipse:** "I am confident."
*   **Correction:** When GPS updates, the ellipse snaps small.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Covariance Explosion"
*   **Symptom:** Filter output becomes NaN or oscillates.
*   **Cause:** $P$ matrix became non-positive-definite or $Q$/$R$ are tuned completely wrong (e.g., assuming sensor implies it's 1000x more accurate than it is).
*   **Fix:** Check `config` booleans. Ensure units match (rad/s vs deg/s). Tune $Q$ and $R$.

#### 2. "TF Tree Loop"
*   **Symptom:** TF Errors in Rviz.
*   **Cause:** `robot_localization` publishes `odom -> base_link`. If your robot driver also publishes `odom -> base_link`, they fight.
*   **Fix:** Disable TF publication in the robot driver, let EKF handle it.

---

## ⚡ Optimization: Square Root Filters

Numerical stability issues (floating point errors) can make $P$ non-symmetric.
**Square Root UKF (SR-UKF):** Propagates the square root of covariance $S$ (where $P = SS^T$).
*   Guarantees positive semi-definiteness.
*   Implemented in `filterpy`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What does the Kalman Gain $K \approx 1$ mean?
    *   **A:** Trust the **Measurement** completely. (Sensor is accurate, Model is noisy).
2.  **Q:** Why not just use GPS?
    *   **A:** Low frequency (1Hz) and blocked by obstacles. EKF fills the gaps with Odometry/IMU (High frequency).
3.  **Q:** What happens if I fuse Yaw from Odom and Yaw from IMU?
    *   **A:** You double count information unless you correctly set process noise. Often better to fuse Yaw Velocity from IMU and integrate it.

### Challenge Task
> **Task:** Implement "Outlier Rejection"
> 1. Calculate Mahalanobis Distance of measurement $z$.
> 2. If distance > Threshold (e.g., 3 sigma), ignore the update.
> 3. Prevents "GPS Jumps" (Multipath reflection) from teleporting the robot.

---

## 📚 Further Reading
- **Probabilistic Robotics:** Thrun, Burgard, Fox (The Bible of SLAM).
- **Robot Localization Package:** Documentation (ros.org).
- **Kalman and Bayesian Filters in Python:** Roger Labbe (GitHub Book).

---

**Day 10 Complete**
