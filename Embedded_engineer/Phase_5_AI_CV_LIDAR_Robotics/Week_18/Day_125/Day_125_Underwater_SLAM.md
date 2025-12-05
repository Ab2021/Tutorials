# Day 125: Underwater SLAM (Acoustic Localization)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 18: Aerial & Underwater Robotics

---

> **📝 Content Creator Instructions:**
> GPS stops at the surface.
> - **Focus:** Long Baseline (LBL), Ultra-Short Baseline (USBL), Doppler Velocity Logs (DVL), and fusing acoustic drift with DVL dead reckoning.
> - **Code:** A Python EKF implementation fusing DVL velocity inputs with simulated LBL range updates to localize an AUV.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Distinguish** between LBL (Transponders on seabed), USBL (Transceiver on ship), and DVL (Velocity sensor).
2.  **Calibrate** Speed of Sound ($c$) based on Salinity/Temp/Depth (CTD).
3.  **Implement** a Dead Reckoning loop using DVL data.
4.  **Correct** Dead Reckoning drift using acoustic range measurements (EKF).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Simulation (LBL Beacons in Gazebo).

### Software Environment
```bash
pip install numpy scipy
```

### Prior Knowledge
- EKF (Day 36).
- Trilateration (Day 111).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Sensor Suite

1.  **DVL (Doppler Velocity Log):** Looks at the floor with 4 sonar beams. Measures velocity relative to bottom ($v_x, v_y, v_z$). Very accurate ($\pm 1\%$), but integrates error over time (Dead Reckoning).
2.  **Depth Sensor:** Measures $Z$ accurately ($\pm 1cm$).
3.  **LBL (Long Baseline):** Acoustic beacons at known lat/lon on seabed. Robot "pings" them. $R = c \times t$. Bounded error ($\pm 10cm$), but slow updates (0.1Hz).

### 🔹 Part 2: USBL (Ultra-Short Baseline)

Mount a transceiver on a surface ship.
*   Ship tracks Robot's position relative to Ship.
*   Ship has GPS.
*   Robot Position = Ship_GPS + USBL_Vector.
*   **Pros:** No seabed setup. **Cons:** Accuracy degrades with depth ($1\%$ of slant range).

### 🔹 Part 3: The SVP (Sound Velocity Profile)

$c$ is not constant. It varies with Depth ($D$), Temp ($T$), Salinity ($S$).
*   Sound curves (refracts) in water.
*   **Ray Tracing:** For deep water, straight-line distance math fails. You must trace the curved path of sound to get accurate ranges.

---

## 💻 Implementation: DVL + LBL Fusion (EKF)

We will fuse high-rate DVL (Velocity) with low-rate LBL (Position).

### 🛠️ Project Structure
```text
day125_underwater_slam/
├── src/
│   ├── auv_ekf.py
└── output/
    ├── trajectory.png
```

### 👨‍💻 EKF Node (`src/auv_ekf.py`)

Using a standard EKF formulation.
State $x = [p_x, p_y, v_x, v_y]$.

```python
import numpy as np
import matplotlib.pyplot as plt

class AUVEKF:
    def __init__(self):
        self.dt = 0.1
        # State: [x, y, vx, vy]
        self.x = np.zeros(4)
        
        # Covariance
        self.P = np.eye(4) * 1.0
        
        # Process Noise (Q) - DVL Noise / Model Uncertainty
        self.Q = np.diag([0.01, 0.01, 0.1, 0.1])
        
        # Measurement Noise (R) - LBL Noise
        self.R_lbl = np.eye(2) * 2.0 # 2m variance
        
        # State Transition (A) (Constant Velocity Model)
        self.A = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Input Matrix (B) - If we controlled thrust directly.
        # Here we treat DVL as a measurement or control input?
        # Let's treat DVL as "Control Input" (Dead Reckoning) driving the kinematic model.
        # New State: [x, y] driven by [vx_dvl, vy_dvl]
        
        # Revised State for simple fusion: [x, y]
        # Prediction: x += vx_dvl * dt
        # Measurement: z = LBL_trilateration
        
        self.state_dr = np.zeros(2) # For comparison
        self.cov_dr = np.eye(2)

    def predict(self, dvl_vel):
        # dvl_vel = [vx, vy]
        
        # 1. State Prediction
        self.x[0] += self.x[2] * self.dt # Using estimated velocity? 
        self.x[1] += self.x[3] * self.dt
        
        # Alternatively, use DVL as direct velocity observation for the velocity state
        # Filter Logic:
        # Predict: x = Fx
        # Update (DVL): H = [0 0 1 0; 0 0 0 1], z=[vx, vy]
        # Update (LBL): H = [1 0 0 0; 0 1 0 0], z=[px, py]
        
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update_dvl(self, measurement):
        # Measure Velocity directly
        H = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        z = measurement
        R = np.eye(2) * 0.05 # DVL is accurate
        
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_lbl(self, measurement):
        # Measure Position directly (Trilateration result)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        z = measurement
        R = self.R_lbl
        
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

def main():
    ekf = AUVEKF()
    
    # Ground Truth Trajectory (Circle)
    times = np.arange(0, 100, ekf.dt)
    gt_x = 100 * np.cos(0.1 * times)
    gt_y = 100 * np.sin(0.1 * times)
    gt_vx = -10 * np.sin(0.1 * times)
    gt_vy = 10 * np.cos(0.1 * times)
    
    est_history = []
    
    for i, t in enumerate(times):
        # 1. Simulate Sensors
        # DVL (Low Noise)
        dvl_noise = np.random.normal(0, 0.1, 2)
        dvl_meas = np.array([gt_vx[i], gt_vy[i]]) + dvl_noise
        
        # LBL (High Noise, Low Freq)
        lbl_meas = None
        if i % 10 == 0: # 1Hz update (dt=0.1)
            lbl_noise = np.random.normal(0, 2.0, 2)
            lbl_meas = np.array([gt_x[i], gt_y[i]]) + lbl_noise
            
        # 2. EKF Step
        ekf.predict(dvl_meas) # Prediction step
        ekf.update_dvl(dvl_meas) # Velocity Update
        
        if lbl_meas is not None:
             ekf.update_lbl(lbl_meas) # Correction Step
             
        est_history.append(ekf.x[:2].copy())
        
    est_history = np.array(est_history)
    
    # Plot
    plt.figure(figsize=(10,10))
    plt.plot(gt_x, gt_y, 'k-', label='Ground Truth')
    plt.plot(est_history[:,0], est_history[:,1], 'r--', label='EKF Fusion')
    
    # Simulate Dead Reckoning (DVL only) for contrast
    dr_x, dr_y = [100], [0]
    for i in range(len(times)-1):
        dr_x.append(dr_x[-1] + (-10*np.sin(0.1*times[i]) + np.random.normal(0,0.1))*0.1)
        dr_y.append(dr_y[-1] + (10*np.cos(0.1*times[i]) + np.random.normal(0,0.1))*0.1)
    
    plt.plot(dr_x, dr_y, 'g:', label='Dead Reckoning (DVL only)')
    
    plt.legend()
    plt.title("Underwater Localization")
    plt.grid()
    plt.savefig("output/trajectory.png")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Drift"

### 1. Lab Objectives
- **Run:** The simulation.
- **Observe:**
    *   **Ground Truth:** Perfect Circle.
    *   **Dead Reckoning:** Circles generally, but drifts away (Random Walk). End error > 50m.
    *   **EKF:** Stays locked to the black line. Jumps slightly when LBL corrects.
- **Fail Check:** Set `lbl_meas = None` always.
- **Result:** EKF degrades to Dead Reckoning.

---

## 🚀 Project: "Homng Beacon"

**Goal:** Return to Charging Station using USBL.
1.  **Sensor:** Station emits a query pulse.
2.  **Robot:** Returns a pulse.
3.  **Station:** Calculates range and bearing. Sends `(r, theta)` to Robot via acoustic modem.
4.  **Robot:** Updates EKF. Navigates to `(0,0)`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Bottom Lock Lost"
*   **Scenario:** DVL needs to see the bottom. If AUV flies too high (> 30m usually) or over a deep trench, DVL fails.
*   **Result:** Velocity data stops.
*   **Fix:** Switch to Inertial navigation (IMU only - Drifts terribly) or surface for GPS.

#### 2. "Sound Velocity Error"
*   **Scenario:** Fresh water vs Salt water.
*   **Impact:** 3% error in speed of sound = 3% error in Distance. Over 1km, that's 30m error.
*   **Fix:** Use a CTD sensor (Conductivity, Temp, Depth) to calculate $c$ continuously.

---

## ⚡ Optimization: Inverted USBL (iUSBL)

Put the fancy transceiver on the *Robot*, and cheap Transponders on the ship/buoys.
*   Robot calculates its own position instantly.
*   No need for acoustic modem feedback delay.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not use Kalman Filter for Sonar Mapping?
    *   **A:** Sonar Mapping is SLAM (adding landmarks). EKF Localization assumes known landmarks.
2.  **Q:** Data rate of Acoustic Modem?
    *   **A:** Very low. ~300 bits per second (bps). Not kilobits. You can send text messages, not video.
3.  **Q:** What is "Bottom Track" vs "Water Track"?
    *   **A:** Bottom Track: Velocity relative to seabed (Navigation). Water Track: Velocity relative to water layer (Current profiling).

### Challenge Task
> **Task:** Ray Tracing.
> 1. Assume $c(z) = 1500 + 0.01 z$.
> 2. Calculate time for sound to travel 1km horizontally at 100m depth compared to straight line.
> 3. Does it curve Up or Down? (Snell's Law: Sound bends towards lower velocity regions).

---

## 📚 Further Reading
- **WHOI (Woods Hole):** "Acoustic Communications and Navigation".
- **Stonefish Simulator:** Advanced marine robotics simulator.

---

**Day 125 Complete**
