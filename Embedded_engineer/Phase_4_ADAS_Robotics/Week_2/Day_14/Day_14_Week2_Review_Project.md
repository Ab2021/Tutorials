# Day 14: Week 2 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 2: Sensor Fusion & State Estimation

---

> **📝 Day 14 Focus:**
> We have traversed the landscape of State Estimation: from the humble Gaussian to the mighty SLAM. We've built Linear KFs, Extended KFs, Unscented KFs, and Particle Filters. Today, we consolidate this knowledge and build a **Production-Grade Sensor Fusion Module** that fuses Lidar, Radar, and IMU data.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Select** the appropriate filter (KF, EKF, UKF, PF) for a given problem based on linearity and computational constraints.
2.  **Architect** a modular Sensor Fusion framework that can accept asynchronous measurements.
3.  **Implement** a robust EKF that fuses Lidar (Position), Radar (Range/Rate), and IMU (Acceleration).
4.  **Tune** Process Noise ($Q$) and Measurement Noise ($R$) matrices for optimal tracking performance.
5.  **Evaluate** your mastery of Week 2 concepts through a comprehensive assessment.

---

## 📚 Week 2 Review

### 1. Probability & Statistics
-   **Uncertainty:** Inherent in sensors and motion.
-   **Gaussian:** Defined by Mean ($\mu$) and Covariance ($\Sigma$).
-   **Bayes Rule:** Posterior $\propto$ Likelihood $\times$ Prior.

### 2. The Kalman Family
-   **Linear KF:** Optimal for linear systems. $F, H$ are constant matrices.
-   **EKF:** For non-linear systems. Uses Jacobians ($H_j, F_j$) to linearize.
    -   *Pros:* Fast, standard.
    -   *Cons:* Diverges if non-linearity is high; tedious Jacobians.
-   **UKF:** Uses Sigma Points (Unscented Transform).
    -   *Pros:* No Jacobians, better accuracy (3rd order).
    -   *Cons:* Slightly slower.

### 3. Non-Parametric Filters
-   **Particle Filter:** Represents belief as a cloud of samples.
    -   *Use Case:* Global Localization (Kidnapped Robot), Multimodal distributions.
    -   *Key Step:* Resampling (Survival of the fittest).

### 4. SLAM
-   **GraphSLAM:** Optimization problem (Least Squares).
-   **Loop Closure:** The key to correcting long-term drift.

---

## 🛠️ Capstone Project: Multi-Sensor Fusion Tracker

**Goal:** Build a Python class `SensorFusion` that tracks a vehicle's 2D position and velocity $[p_x, p_y, v_x, v_y]$.

**Inputs:**
1.  **Lidar:** $[p_x, p_y]$ at 10Hz. (Linear)
2.  **Radar:** $[\rho, \phi, \dot{\rho}]$ at 20Hz. (Non-linear)
3.  **IMU:** $[a_x, a_y]$ at 50Hz. (Used for Prediction/Process Noise).

**Architecture:**
-   **Predict:** Driven by time ($\Delta t$).
-   **Update:** Triggered by measurement arrival.

### Package Structure
Create `week2_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week2_project
cd ~/ros2_ws/src/week2_project
touch fusion_tracker.py
```

### 👨‍💻 Code: The Fusion Engine

```python
import numpy as np
import math
import matplotlib.pyplot as plt

class FusionEKF:
    def __init__(self):
        # State: [px, py, vx, vy]
        self.x = np.zeros((4, 1))
        
        # Covariance: High uncertainty initially
        self.P = np.eye(4) * 1000
        
        # Process Noise Covariance (Q) - Will be updated based on dt
        self.Q = np.zeros((4, 4))
        
        # Measurement Noise
        # Lidar: Standard deviation 0.15m
        self.R_lidar = np.array([[0.0225, 0],
                                 [0, 0.0225]])
        
        # Radar: rho=0.3m, phi=0.03rad, rho_dot=0.3m/s
        self.R_radar = np.array([[0.09, 0, 0],
                                 [0, 0.0009, 0],
                                 [0, 0, 0.09]])
        
        # Measurement Matrices
        self.H_lidar = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]])
                                 
        self.I = np.eye(4)
        
        # Timestamp of last measurement
        self.last_timestamp = 0

    def predict(self, dt, noise_ax=9.0, noise_ay=9.0):
        """
        Predict state after dt.
        noise_ax, noise_ay: Process noise spectral density (acceleration variance)
        """
        # 1. Update F matrix (Linear Constant Velocity)
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        
        # 2. Update Q matrix (Discrete Process Noise)
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        
        self.Q = np.array([
            [dt4/4*noise_ax, 0, dt3/2*noise_ax, 0],
            [0, dt4/4*noise_ay, 0, dt3/2*noise_ay],
            [dt3/2*noise_ax, 0, dt2*noise_ax, 0],
            [0, dt3/2*noise_ay, 0, dt2*noise_ay]
        ])
        
        # 3. Predict State
        self.x = F @ self.x
        
        # 4. Predict Covariance
        self.P = F @ self.P @ F.T + self.Q

    def update_lidar(self, z):
        # Linear Update
        y = z - self.H_lidar @ self.x
        S = self.H_lidar @ self.P @ self.H_lidar.T + self.R_lidar
        K = self.P @ self.H_lidar.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H_lidar) @ self.P

    def calculate_jacobian(self, x):
        px, py, vx, vy = x.flatten()
        c1 = px**2 + py**2
        c2 = math.sqrt(c1)
        c3 = (c1 * c2)
        
        if abs(c1) < 0.0001:
            return np.zeros((3, 4))
            
        Hj = np.array([
            [px/c2, py/c2, 0, 0],
            [-py/c1, px/c1, 0, 0],
            [py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2]
        ])
        return Hj

    def update_radar(self, z):
        # Non-Linear Update
        px, py, vx, vy = self.x.flatten()
        
        rho = math.sqrt(px**2 + py**2)
        phi = math.atan2(py, px)
        rho_dot = (px*vx + py*vy) / rho if rho > 0.0001 else 0
        
        z_pred = np.array([[rho], [phi], [rho_dot]])
        y = z - z_pred
        
        # Normalize angle
        while y[1] > math.pi: y[1] -= 2*math.pi
        while y[1] < -math.pi: y[1] += 2*math.pi
        
        Hj = self.calculate_jacobian(self.x)
        
        S = Hj @ self.P @ Hj.T + self.R_radar
        K = self.P @ Hj.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (self.I - K @ Hj) @ self.P

    def process_measurement(self, measurement_pack):
        """
        Main entry point.
        measurement_pack: {'sensor_type': 'L'/'R', 'timestamp': t, 'data': z}
        """
        timestamp = measurement_pack['timestamp']
        
        if self.last_timestamp == 0:
            # Initialize
            self.last_timestamp = timestamp
            z = measurement_pack['data']
            
            if measurement_pack['sensor_type'] == 'L':
                self.x[0] = z[0]
                self.x[1] = z[1]
            else:
                # Convert Radar polar to cartesian for initialization
                rho, phi = z[0], z[1]
                self.x[0] = rho * math.cos(phi)
                self.x[1] = rho * math.sin(phi)
            return

        # Calculate dt
        dt = (timestamp - self.last_timestamp) / 1000000.0 # Assuming micros
        self.last_timestamp = timestamp
        
        # Predict
        self.predict(dt)
        
        # Update
        if measurement_pack['sensor_type'] == 'L':
            self.update_lidar(measurement_pack['data'])
        elif measurement_pack['sensor_type'] == 'R':
            self.update_radar(measurement_pack['data'])

def generate_data():
    # Simulate a car moving in a curve
    t = np.linspace(0, 10, 200) # 10 seconds
    gt_x = t * 2
    gt_y = 2 * np.sin(t)
    gt_vx = 2 * np.ones_like(t)
    gt_vy = 2 * np.cos(t)
    
    measurements = []
    
    for i in range(len(t)):
        timestamp = t[i] * 1000000 # micros
        
        # Lidar (10Hz approx) - Every 2nd step
        if i % 2 == 0:
            z = np.array([[gt_x[i]], [gt_y[i]]]) + np.random.normal(0, 0.15, (2,1))
            measurements.append({'sensor_type': 'L', 'timestamp': timestamp, 'data': z})
            
        # Radar (20Hz approx) - Every step
        px, py = gt_x[i], gt_y[i]
        vx, vy = gt_vx[i], gt_vy[i]
        rho = math.sqrt(px**2 + py**2)
        phi = math.atan2(py, px)
        rho_dot = (px*vx + py*vy) / rho if rho > 0.0001 else 0
        
        z = np.array([[rho], [phi], [rho_dot]]) + np.random.normal(0, [0.3, 0.03, 0.3]).reshape(3,1)
        measurements.append({'sensor_type': 'R', 'timestamp': timestamp, 'data': z})
        
    return measurements, gt_x, gt_y

def run_tracker():
    measurements, gt_x, gt_y = generate_data()
    tracker = FusionEKF()
    
    est_x = []
    est_y = []
    
    for meas in measurements:
        tracker.process_measurement(meas)
        est_x.append(tracker.x[0,0])
        est_y.append(tracker.x[1,0])
        
    # Plotting
    # Note: est_x length might differ from gt_x due to multiple measurements per timestep
    # For simple viz, we just plot the path
    
    plt.figure(figsize=(10, 6))
    plt.plot(gt_x, gt_y, 'k--', label='Ground Truth')
    plt.plot(est_x, est_y, 'b-', label='EKF Estimate')
    plt.title("Multi-Sensor Fusion (Lidar + Radar)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    run_tracker()
```

---

## 🧪 Verification & Testing

### 1. RMSE Calculation
To validate the tracker, we calculate the **Root Mean Squared Error (RMSE)** against the ground truth.
$$ RMSE = \sqrt{\frac{1}{n} \sum (\hat{x} - x_{gt})^2} $$

**Acceptance Criteria:**
-   RMSE X < 0.11 m
-   RMSE Y < 0.11 m
-   RMSE VX < 0.52 m/s
-   RMSE VY < 0.52 m/s

### 2. Consistency Check (NIS)
**Normalized Innovation Squared (NIS):**
$$ \epsilon = y^T S^{-1} y $$
-   Follows a Chi-Squared distribution.
-   Plot NIS over time.
-   If NIS is consistently high (> 7.8 for 3 DOF), you are underestimating uncertainty (P is too small, or Q/R too small).
-   If NIS is consistently low, you are overestimating uncertainty.

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Kalman Filters
1.  **Q:** Why does the uncertainty ($P$) increase during Prediction and decrease during Update?
    *   **A:** Prediction adds Process Noise ($Q$), adding uncertainty. Update adds Measurement Information ($R$), reducing uncertainty.
2.  **Q:** What happens if you initialize $P$ with zeros?
    *   **A:** The filter assumes it knows the state perfectly and will ignore measurements (Kalman Gain $\approx 0$).

### Section 2: EKF vs UKF
3.  **Q:** In the Radar update, why do we need a Jacobian?
    *   **A:** To map the uncertainty (Gaussian covariance) from Cartesian space (State) to Polar space (Measurement) linearly.
4.  **Q:** If the system is highly non-linear, why might EKF diverge?
    *   **A:** The linear approximation (tangent) might be valid only for a tiny region. If the uncertainty is large, the tangent points away from the true function.

### Section 3: Particle Filters
5.  **Q:** What is the "Kidnapped Robot Problem"?
    *   **A:** When the robot is teleported to an unknown location. EKF cannot recover (unimodal). PF can recover (multimodal/global loc).
6.  **Q:** What is the cost of increasing the number of particles?
    *   **A:** Linear increase in CPU usage ($O(N)$).

---

## 🏆 Conclusion

Congratulations on completing Week 2! You have mastered the mathematical core of robotics.
-   You can track objects using Lidar and Radar.
-   You understand how to handle noise and uncertainty.
-   You have built the foundation for the Perception stack.

**Next Week:** We move to **Computer Vision & Deep Learning**. We will teach the car to *see* lanes, signs, and pedestrians using Cameras and CNNs.

---

**Day 14 Complete** | Phase 4: ADAS & Robotics Systems | Week 2: Sensor Fusion & State Estimation
