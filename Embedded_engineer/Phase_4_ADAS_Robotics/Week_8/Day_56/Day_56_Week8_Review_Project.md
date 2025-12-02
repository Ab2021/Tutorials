# Day 56: Week 8 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 8: State Estimation & Sensor Fusion

---

> **📝 Day 56 Focus:**
> We started with uncertainty ("I might be here"). We built mathematical tools (Bayes, KF, PF) to tame it. We combined sensors (Lidar, Radar, Camera) to overcome individual weaknesses. Today, we build **The Ultimate Tracker**—a robust fusion engine that combines everything we've learned into a single, production-grade state estimator.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Synthesize** the concepts of Prediction (Motion) and Update (Measurement).
2.  **Select** the right filter (EKF vs PF) for a given problem (Linear vs Non-Linear, Gaussian vs Multi-modal).
3.  **Implement** a Multi-Sensor Fusion algorithm combining Odometry, GPS, and Landmark observations.
4.  **Visualize** the covariance ellipses to understand the confidence of the estimator.
5.  **Evaluate** your mastery of Week 8 concepts through a comprehensive assessment.

---

## 📚 Week 8 Review

### 1. Probability & Bayes Filter
-   **Belief:** $p(x)$.
-   **Bayes Rule:** Posterior $\propto$ Likelihood $\times$ Prior.
-   **Algorithm:** Predict (Motion) -> Update (Measurement).

### 2. Kalman Filters (KF/EKF)
-   **KF:** Optimal for Linear Gaussian systems.
-   **EKF:** Linearizes non-linear models using Jacobians ($F_j, H_j$).
-   **Covariance ($P$):** Represents uncertainty.

### 3. Particle Filters (MCL)
-   **Non-Parametric:** Represents belief as samples.
-   **Resampling:** Survival of the fittest.
-   **Use Case:** Global Localization (Kidnapped Robot).

### 4. SLAM
-   **Problem:** Unknown Map + Unknown Pose.
-   **GMapping:** Particle Filter based (FastSLAM).
-   **Cartographer:** Graph Optimization based (Loop Closure).

### 5. Sensor Fusion
-   **Complementary:** Lidar (Pos) + Radar (Vel).
-   **Redundant:** GPS (Pos) + Lidar (Pos).
-   **Math:** Weighted average based on covariance matrices.

---

## 🛠️ Capstone Project: The Ultimate Tracker

**Goal:** Track a robot driving in a figure-8 pattern.
**Sensors:**
1.  **Odometry:** $[v, \omega]$ (Control Input). High frequency, drifts.
2.  **GPS:** $[x, y]$. Low frequency, noisy, no drift.
3.  **Landmark Sensor (Lidar):** $[r, \phi]$ to known landmarks. High accuracy, limited range.

**Task:** Implement an EKF to fuse all three.

### Architecture
-   **State:** $[x, y, \theta]$ (3D).
-   **Motion Model:** Velocity Motion Model (Non-Linear).
-   **Measurement Models:**
    -   GPS: Linear ($H = I$).
    -   Landmark: Non-Linear ($r = \sqrt{\Delta x^2 + \Delta y^2}$).

### Package Structure
Create `week8_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week8_project
cd ~/ros2_ws/src/week8_project
touch tracker.py
```

### 👨‍💻 Code: The Ultimate Tracker (EKF)

```python
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Ellipse

# --- Configuration ---
DT = 0.1
SIM_TIME = 50.0

# Noise Parameters
Q = np.diag([0.1, 0.1, np.deg2rad(1.0)]) ** 2 # Motion Noise [x, y, theta]
R_GPS = np.diag([1.0, 1.0]) ** 2 # GPS Noise [x, y]
R_LM = np.diag([0.5, np.deg2rad(5.0)]) ** 2 # Landmark Noise [r, phi]

# Landmarks
LANDMARKS = np.array([
    [10.0, 10.0],
    [-10.0, 10.0],
    [0.0, -10.0]
])

class EKFTracker:
    def __init__(self):
        self.x = np.zeros((3, 1)) # [x, y, theta]
        self.P = np.eye(3)
        
    def predict(self, u):
        # u: [v, omega]
        v = u[0, 0]
        w = u[1, 0]
        theta = self.x[2, 0]
        
        # Motion Model (Velocity)
        # x' = x + v*cos(theta)*dt
        # y' = y + v*sin(theta)*dt
        # theta' = theta + w*dt
        
        self.x[0, 0] += v * math.cos(theta) * DT
        self.x[1, 0] += v * math.sin(theta) * DT
        self.x[2, 0] += w * DT
        
        # Jacobian F
        F = np.eye(3)
        F[0, 2] = -v * math.sin(theta) * DT
        F[1, 2] = v * math.cos(theta) * DT
        
        # Predict Covariance
        self.P = F @ self.P @ F.T + Q
        
    def update_gps(self, z):
        # z: [x, y]
        H = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        
        y = z - H @ self.x # Innovation
        S = H @ self.P @ H.T + R_GPS
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P
        
    def update_landmark(self, z, lm_id):
        # z: [r, phi]
        lm_x = LANDMARKS[lm_id, 0]
        lm_y = LANDMARKS[lm_id, 1]
        
        dx = lm_x - self.x[0, 0]
        dy = lm_y - self.x[1, 0]
        q = dx**2 + dy**2
        r_pred = math.sqrt(q)
        phi_pred = math.atan2(dy, dx) - self.x[2, 0]
        
        # Normalize angle
        phi_pred = (phi_pred + np.pi) % (2 * np.pi) - np.pi
        
        z_pred = np.array([[r_pred], [phi_pred]])
        y = z - z_pred
        y[1, 0] = (y[1, 0] + np.pi) % (2 * np.pi) - np.pi # Normalize innovation
        
        # Jacobian H
        H = np.array([
            [-dx/math.sqrt(q), -dy/math.sqrt(q), 0],
            [dy/q, -dx/q, -1]
        ])
        
        S = H @ self.P @ H.T + R_LM
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

def plot_covariance_ellipse(xEst, PEst):
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)
    
    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0
        
    t = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    
    ell = Ellipse(xy=(xEst[0, 0], xEst[1, 0]),
                  width=a * 3.0 * 2, height=b * 3.0 * 2, # 3-sigma
                  angle=np.rad2deg(t), color='blue', alpha=0.3)
    plt.gca().add_artist(ell)

def main():
    ekf = EKFTracker()
    
    xTrue = np.zeros((3, 1))
    xDR = np.zeros((3, 1))
    
    h_xTrue = []
    h_xEst = []
    h_xDR = []
    h_zGPS = []
    
    time = 0.0
    while time <= SIM_TIME:
        time += DT
        
        # 1. Control (Figure 8)
        v = 1.0
        w = math.cos(time / 5.0) * 0.5
        u = np.array([[v], [w]])
        
        # 2. Ground Truth
        xTrue[0, 0] += v * math.cos(xTrue[2, 0]) * DT
        xTrue[1, 0] += v * math.sin(xTrue[2, 0]) * DT
        xTrue[2, 0] += w * DT
        
        # 3. Dead Reckoning (Noisy Control)
        u_noisy = u + np.random.randn(2, 1) * np.array([[0.1], [0.1]])
        xDR[0, 0] += u_noisy[0, 0] * math.cos(xDR[2, 0]) * DT
        xDR[1, 0] += u_noisy[0, 0] * math.sin(xDR[2, 0]) * DT
        xDR[2, 0] += u_noisy[1, 0] * DT
        
        # 4. EKF Predict
        ekf.predict(u_noisy)
        
        # 5. GPS Update (Every 1.0s)
        if int(time * 10) % 10 == 0:
            z_gps = np.array([
                [xTrue[0, 0] + np.random.randn() * 1.0],
                [xTrue[1, 0] + np.random.randn() * 1.0]
            ])
            ekf.update_gps(z_gps)
            h_zGPS.append(z_gps)
        else:
            h_zGPS.append(np.array([[np.nan], [np.nan]]))
            
        # 6. Landmark Update (Every step, if in range)
        for i in range(len(LANDMARKS)):
            dx = LANDMARKS[i, 0] - xTrue[0, 0]
            dy = LANDMARKS[i, 1] - xTrue[1, 0]
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist < 15.0: # Range limit
                z_lm = np.array([
                    [dist + np.random.randn() * 0.5],
                    [math.atan2(dy, dx) - xTrue[2, 0] + np.random.randn() * 0.05]
                ])
                ekf.update_landmark(z_lm, i)
        
        # Store
        h_xTrue.append(xTrue.copy())
        h_xEst.append(ekf.x.copy())
        h_xDR.append(xDR.copy())
        
        # Visualization
        if int(time * 10) % 5 == 0:
            plt.cla()
            plt.plot([x[0, 0] for x in h_xTrue], [x[1, 0] for x in h_xTrue], '-k', label='True')
            plt.plot([x[0, 0] for x in h_xDR], [x[1, 0] for x in h_xDR], '--k', label='DR')
            plt.plot([x[0, 0] for x in h_xEst], [x[1, 0] for x in h_xEst], '-r', label='EKF')
            
            # Plot GPS
            gps_x = [z[0, 0] for z in h_zGPS if not np.isnan(z[0, 0])]
            gps_y = [z[1, 0] for z in h_zGPS if not np.isnan(z[0, 0])]
            plt.plot(gps_x, gps_y, '.g', alpha=0.3, label='GPS')
            
            # Plot Landmarks
            plt.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], 'ob', markersize=10, label='Landmarks')
            
            # Plot Covariance
            plot_covariance_ellipse(ekf.x, ekf.P)
            
            plt.legend()
            plt.axis('equal')
            plt.pause(0.001)
            
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🧪 Verification & Testing

### 1. Drift Check
-   Disable GPS and Landmark updates.
-   **Observation:** The Red line (EKF) should overlap with the Dashed line (Dead Reckoning) and drift away from Truth. Covariance ellipse should grow indefinitely.

### 2. Fusion Check
-   Enable GPS.
-   **Observation:** The Red line snaps back near the Truth every 1 second. Covariance ellipse shrinks periodically.

### 3. Precision Check
-   Enable Landmarks.
-   **Observation:** When near a landmark, the Red line becomes very tight to the Truth. Landmarks provide high-precision relative constraints.

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Filters
1.  **Q:** When should you use a Particle Filter instead of an EKF?
    *   **A:** When the distribution is non-Gaussian (e.g., multi-modal belief in global localization) or the system is highly non-linear.
2.  **Q:** What is the cost of the EKF?
    *   **A:** $O(N^3)$ due to matrix inversion, where $N$ is the state size.
3.  **Q:** What is the cost of the PF?
    *   **A:** $O(M)$, where $M$ is the number of particles.

### Section 2: Fusion
4.  **Q:** Why do we fuse GPS and Odometry?
    *   **A:** Odometry is smooth but drifts (good short-term). GPS is noisy but drift-free (good long-term). Together they are smooth and accurate.
5.  **Q:** How does the EKF know which sensor to trust?
    *   **A:** Through the Covariance Matrices ($P$ vs $R$). If $P$ (State Uncertainty) is large and $R$ (Sensor Noise) is small, it trusts the sensor.

### Section 3: SLAM
6.  **Q:** What is the difference between Localization and SLAM?
    *   **A:** Localization assumes a known map. SLAM builds the map while localizing.

---

## 🏆 Conclusion

Congratulations on completing Week 8!
-   You have mastered **State Estimation**.
-   You can fuse noisy sensors to find the truth.
-   You understand the math behind the magic.

**Next Week:** We move to **Path Planning**. Now that we know *where* we are (Week 8) and *what* is around us (Week 7), we need to decide *how* to get to the destination. We will explore A*, RRT, and Trajectory Optimization.

---

**Day 56 Complete** | Phase 4: ADAS & Robotics Systems | Week 8: State Estimation & Sensor Fusion
