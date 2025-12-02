# Day 55: Sensor Fusion (Lidar + Radar + Camera)
## Phase 4: ADAS & Robotics Systems | Week 8: State Estimation & Sensor Fusion

---

> **📝 Day 55 Focus:**
> Lidar is precise but blind in rain. Radar sees through fog but has poor resolution. Cameras read signs but fail in the dark. No single sensor is perfect. **Sensor Fusion** combines them to create a "Super Sensor" that is robust, accurate, and reliable.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Classify** Fusion Architectures: Low-Level (Raw Data), Mid-Level (Feature), High-Level (Object).
2.  **Solve** the Data Association problem (Who is Who?) using Nearest Neighbor.
3.  **Implement** Track-to-Track Fusion using Weighted Covariance Intersection.
4.  **Fuse** Lidar (Position) and Radar (Velocity) data in an EKF.
5.  **Project** Lidar points onto Camera images for "Colorized Point Clouds".

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 51:** Kalman Filter.
-   **Day 27:** Extrinsic Calibration ($T_{cam}^{lidar}$).

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Fusion Levels

1.  **Low-Level (Raw Data):**
    -   Example: Projecting Lidar points onto Camera image pixels.
    -   Pros: No information loss.
    -   Cons: High bandwidth, sensitive to calibration errors.
2.  **Mid-Level (Feature):**
    -   Example: Combining "Vertical Edge" from Camera with "Line Segment" from Lidar.
3.  **High-Level (Object/Track):**
    -   Example: Camera says "Car at (10, 5)". Radar says "Object at (10.1, 5.2)".
    -   Fusion: "Car at (10.05, 5.1)".
    -   Pros: Low bandwidth, modular.
    -   Cons: Information loss (thresholding).

### 🔹 Part 2: Data Association

Before fusing, we must know *what* to fuse.
-   **Scenario:** Camera sees 2 cars. Radar sees 2 objects. Which Radar object matches which Camera car?
-   **Gating:** Ignore matches that are physically impossible (too far away).
-   **Nearest Neighbor (NN):** Match closest pairs (Mahalanobis Distance).
-   **Global Nearest Neighbor (GNN):** Minimize total distance (Hungarian Algorithm).

### 🔹 Part 3: Track Fusion Math

Given two estimates of the same state $x$:
1.  $x_1, P_1$ (Camera)
2.  $x_2, P_2$ (Radar)

**Optimal Fusion (Weighted Average):**
$$ P_{fused} = (P_1^{-1} + P_2^{-1})^{-1} $$
$$ x_{fused} = P_{fused} (P_1^{-1} x_1 + P_2^{-1} x_2) $$
*Note: This assumes errors are independent. If they share "Process Noise" (same motion model), we use Covariance Intersection.*

---

## 💻 Implementation: High-Level Fusion

**Scenario:**
-   **Target:** Moving in 1D ($x, v$).
-   **Lidar:** Measures position $x$ (High accuracy).
-   **Radar:** Measures velocity $v$ (High accuracy) and position $x$ (Low accuracy).
-   **Fusion:** Combine them to get best $x$ and $v$.

### 🛠️ Setup
Create `week8_day55` and `sensor_fusion.py`.

```bash
mkdir -p ~/ros2_ws/src/week8_day55
cd ~/ros2_ws/src/week8_day55
touch sensor_fusion.py
```

### 👨‍💻 Code: Lidar-Radar Fusion (EKF)

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
DT = 0.1
SIM_TIME = 20.0

# Process Noise (Motion Model Uncertainty)
Q = np.diag([0.1, 0.1]) ** 2 # [pos, vel]

# Measurement Noise
R_lidar = np.array([[0.1]]) ** 2 # Position variance
R_radar = np.diag([1.0, 0.1]) ** 2 # [pos, vel] variance

class FusionEKF:
    def __init__(self):
        # State: [x, v]
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)
        
        # Motion Model Matrix (Constant Velocity)
        self.F = np.array([[1, DT],
                           [0, 1]])
                           
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        
    def update_lidar(self, z):
        # Lidar measures [x]
        H = np.array([[1, 0]])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R_lidar
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P
        
    def update_radar(self, z):
        # Radar measures [x, v]
        H = np.eye(2)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R_radar
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

def main():
    ekf = FusionEKF()
    
    # Ground Truth
    x_true = 0.0
    v_true = 5.0 # m/s
    
    # History
    h_x_true = []
    h_v_true = []
    h_x_est = []
    h_v_est = []
    h_z_lidar = []
    
    time = 0.0
    while time <= SIM_TIME:
        time += DT
        
        # 1. Move Target
        x_true += v_true * DT
        
        # 2. Simulate Sensors
        z_lidar = np.array([[x_true + np.random.randn() * 0.1]])
        z_radar = np.array([[x_true + np.random.randn() * 1.0],
                            [v_true + np.random.randn() * 0.1]])
                            
        # 3. Fusion Cycle
        ekf.predict()
        
        # Asynchronous Updates (Simulate random arrival)
        if np.random.rand() < 0.5:
            ekf.update_lidar(z_lidar)
        else:
            ekf.update_radar(z_radar)
            
        # Store
        h_x_true.append(x_true)
        h_v_true.append(v_true)
        h_x_est.append(ekf.x[0, 0])
        h_v_est.append(ekf.x[1, 0])
        h_z_lidar.append(z_lidar[0, 0])
        
    # Plot
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(h_x_true, label='True Pos')
    plt.plot(h_z_lidar, '.g', alpha=0.3, label='Lidar Meas')
    plt.plot(h_x_est, 'r', label='Fused Est')
    plt.legend()
    plt.title("Position Fusion")
    
    plt.subplot(2, 1, 2)
    plt.plot(h_v_true, label='True Vel')
    plt.plot(h_v_est, 'r', label='Fused Est')
    plt.legend()
    plt.title("Velocity Fusion")
    
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Doppler Effect

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   Position estimate is smooth (Thanks to Lidar).
    -   Velocity estimate converges quickly (Thanks to Radar).
3.  **Experiment:**
    -   Disable Radar update (`# ekf.update_radar(z_radar)`).
    -   *Result:* Velocity estimate becomes noisy and lags. Lidar alone needs to differentiate position ($\Delta x / \Delta t$) to get velocity, which amplifies noise. Radar gives velocity *directly* (Doppler), which is instant and clean.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Data Association Fail
**Symptom:** Fusion output jumps between two targets.
**Cause:** The filter thinks measurement A belongs to Track B, then Track A.
**Solution:** Implement Gating. If $z$ is $> 3\sigma$ away from predicted $x$, do not update. Start a new track instead.

#### 2. Latency
**Symptom:** Estimate lags behind truth.
**Cause:** Sensors have processing delay (Camera ~100ms, Radar ~50ms).
**Solution:** **Time Compensation**. Predict the state forward to the timestamp of the measurement before updating.

---

## ⚡ Optimization & Best Practices

### 1. Frustum Culling
When projecting Lidar to Camera:
-   Only project points that are *in front* of the camera ($Z > 0$) and within the Field of View.
-   Saves computation.

### 2. Heterogeneous Fusion
-   **Lidar:** Good for Geometry (Size, Shape).
-   **Camera:** Good for Semantics (Class, Color).
-   **Radar:** Good for Kinematics (Velocity).
-   **Fusion:** Use Camera for "Car", Lidar for "Box", Radar for "Speed".

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is Radar better than Lidar for velocity?
    *   **A:** Radar uses the Doppler Effect to measure radial velocity directly in a single shot. Lidar requires at least two frames to calculate difference.
2.  **Q:** What is "Gating"?
    *   **A:** A technique to reject measurements that are statistically unlikely to belong to a track (e.g., outside the 99% confidence ellipsoid).
3.  **Q:** In Track-to-Track fusion, why can't we just average the states?
    *   **A:** Because one sensor might be much more uncertain than the other. We must weight them by the inverse of their covariance matrices (Precision).

### Challenge Task
**Task:** 3D Projection.
1.  Take a KITTI dataset frame (Image + Lidar).
2.  Load Calibration matrices ($P_{rect}, Tr_{velo\_to\_cam}$).
3.  Project Lidar points onto the Image.
4.  Color the points based on depth (Rainbow map).

---

## 📚 Further Reading & References
-   [Udacity Sensor Fusion Nanodegree](https://www.udacity.com/course/sensor-fusion-engineer-nanodegree--nd313)
-   [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)

---

**Day 55 Complete** | Phase 4: ADAS & Robotics Systems | Week 8: State Estimation & Sensor Fusion
