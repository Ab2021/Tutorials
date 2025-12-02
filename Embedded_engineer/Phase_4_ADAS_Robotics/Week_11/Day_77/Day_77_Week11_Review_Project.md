# Day 77: Week 11 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 11: Localization

---

> **📝 Day 77 Focus:**
> We have explored the "Where am I?" problem from space (GPS) to the wheel (Odometry) to the camera (VIO). Today, we combine these sensors into a robust **Urban Localization System** that can survive tunnels, urban canyons, and slippery roads.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Integrate** GNSS, IMU, and Odometry into a Loose-Coupled EKF.
2.  **Simulate** a city driving scenario with GPS outages (Tunnels).
3.  **Implement** a "Dead Reckoning" fallback when GPS is lost.
4.  **Visualize** the uncertainty ellipse growing and shrinking.
5.  **Evaluate** the accuracy of your fusion against Ground Truth.

---

## 📚 Week 11 Review

### 1. GNSS (Global Navigation Satellite System)
-   **Pros:** Absolute position (Lat/Lon). No drift.
-   **Cons:** Low rate (1-10Hz). Blocked by buildings/tunnels. Multipath error.
-   **RTK:** Centimeter accuracy using Base Station corrections.

### 2. IMU (Inertial Measurement Unit)
-   **Pros:** High rate (100Hz+). Works everywhere.
-   **Cons:** Drifts quickly (Double integration of bias).
-   **Errors:** Bias Instability, Random Walk.

### 3. Odometry (Wheel/Visual)
-   **Pros:** Good short-term accuracy. Measures relative motion.
-   **Cons:** Wheel slip. Scale drift (Visual).
-   **Dead Reckoning:** Estimating position by integrating speed/heading.

### 4. AMCL (Adaptive Monte Carlo Localization)
-   **Pros:** Global localization in a known map. Multi-modal (can handle ambiguity).
-   **Cons:** Needs a map. Computationally expensive (particles).

---

## 🛠️ Capstone Project: Urban Localization

**Goal:** Track a car driving through a city with a tunnel.
**Sensors:**
-   **GPS:** 1Hz, 5m noise. Unavailable in tunnel.
-   **IMU:** 100Hz, Bias + Noise.
-   **Odom:** 50Hz, Slip noise.

**Algorithm:** Error State Kalman Filter (ES-EKF) or simple EKF.

### Package Structure
Create `week11_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week11_project
cd ~/ros2_ws/src/week11_project
touch urban_localization.py
```

### 👨‍💻 Code: Urban Localization EKF

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
DT = 0.1 # 10 Hz simulation
TUNNEL_START = 200.0 # meters
TUNNEL_END = 400.0   # meters

class EKF:
    def __init__(self):
        # State: [x, y, theta, v]
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 1.0
        
    def predict(self, a, w, dt):
        # Motion Model (Bicycle)
        # x' = x + v*cos(theta)*dt
        # y' = y + v*sin(theta)*dt
        # theta' = theta + w*dt
        # v' = v + a*dt
        
        theta = self.x[2, 0]
        v = self.x[3, 0]
        
        # Jacobian F
        F = np.eye(4)
        F[0, 2] = -v * np.sin(theta) * dt
        F[0, 3] = np.cos(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt
        F[1, 3] = np.sin(theta) * dt
        
        # State Prediction
        self.x[0, 0] += v * np.cos(theta) * dt
        self.x[1, 0] += v * np.sin(theta) * dt
        self.x[2, 0] += w * dt
        self.x[3, 0] += a * dt
        
        # Covariance Prediction
        Q = np.diag([0.1, 0.1, 0.01, 0.1]) # Process Noise
        self.P = F @ self.P @ F.T + Q
        
    def update_gps(self, z):
        # z: [x_gps, y_gps]
        H = np.zeros((2, 4))
        H[0, 0] = 1
        H[1, 1] = 1
        
        R = np.eye(2) * 25.0 # GPS Noise Variance (5m std)
        
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

def main():
    ekf = EKF()
    
    # Simulation
    # Car drives straight at 20 m/s
    
    gt_x = []
    gt_y = []
    est_x = []
    est_y = []
    gps_x = []
    gps_y = []
    
    pos_true = np.array([0.0, 0.0])
    vel_true = 20.0
    theta_true = 0.0
    
    print("Starting Urban Simulation...")
    print(f"Tunnel: {TUNNEL_START}m to {TUNNEL_END}m")
    
    for t in np.arange(0, 30.0, DT):
        # 1. True Motion
        pos_true[0] += vel_true * np.cos(theta_true) * DT
        pos_true[1] += vel_true * np.sin(theta_true) * DT
        
        # 2. Sensors
        # IMU/Odom (Control Input)
        a_meas = 0.0 + np.random.normal(0, 0.5)
        w_meas = 0.0 + np.random.normal(0, 0.05)
        
        # GPS (Measurement)
        has_gps = True
        if TUNNEL_START < pos_true[0] < TUNNEL_END:
            has_gps = False
            
        z_gps = None
        if has_gps:
            z_gps = pos_true + np.random.normal(0, 5.0, 2)
            gps_x.append(z_gps[0])
            gps_y.append(z_gps[1])
        
        # 3. EKF
        ekf.predict(a_meas, w_meas, DT)
        if z_gps is not None:
            ekf.update_gps(z_gps.reshape(2, 1))
            
        # Store
        gt_x.append(pos_true[0])
        gt_y.append(pos_true[1])
        est_x.append(ekf.x[0, 0])
        est_y.append(ekf.x[1, 0])
        
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(gt_x, gt_y, 'k-', linewidth=2, label='Ground Truth')
    plt.plot(gps_x, gps_y, 'g.', alpha=0.3, label='GPS Measurements')
    plt.plot(est_x, est_y, 'b-', linewidth=2, label='EKF Estimate')
    
    # Draw Tunnel
    plt.axvspan(TUNNEL_START, TUNNEL_END, color='gray', alpha=0.3, label='Tunnel (No GPS)')
    
    plt.title("Urban Localization (GPS + IMU/Odom Fusion)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🧪 Verification & Testing

### 1. The Tunnel Test
-   **Observation:**
    -   Before Tunnel: EKF follows GPS dots (Green).
    -   Inside Tunnel: GPS dots disappear. EKF continues (Blue line) based on Prediction (IMU/Odom).
    -   **Drift:** The Blue line starts to drift away from Black (GT) slowly.
    -   After Tunnel: GPS returns. EKF snaps back to the Green dots.
-   **Insight:** The "Snap" shows the correction of the accumulated drift.

### 2. High Noise
-   **Experiment:** Increase IMU noise (Process Noise $Q$).
-   **Result:** The EKF trusts the Prediction less. Inside the tunnel, the uncertainty ($P$) grows faster. When GPS returns, the correction is stronger (larger Kalman Gain $K$).

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Sensors
1.  **Q:** Which sensor provides the "Prediction" step in EKF?
    *   **A:** IMU or Odometry (High frequency).
2.  **Q:** Which sensor provides the "Update" step?
    *   **A:** GPS or Camera/LiDAR (Low frequency, Absolute).

### Section 2: Fusion
3.  **Q:** What happens to the Covariance Matrix $P$ during prediction?
    *   **A:** It grows (Add Process Noise $Q$). We become less sure.
4.  **Q:** What happens to $P$ during update?
    *   **A:** It shrinks (Inverse of Measurement Noise $R$). We become more sure.

---

## 🏆 Conclusion

Congratulations on completing Week 11!
-   You have mastered **Localization**.
-   You can fuse noisy sensors to get a smooth, accurate position estimate.

**Next Week:** We move to **HD Maps**. Knowing your $(x, y)$ is useless if you don't know where the road is. We will learn how to map the world and match our position to lanes and traffic rules.

---

**Day 77 Complete** | Phase 4: ADAS & Robotics Systems | Week 11: Localization
