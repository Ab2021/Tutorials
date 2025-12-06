# Day 163: GPS Spoofing & Defense
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 24: Cybersecurity & Robustness

---

> **📝 Content Creator Instructions:**
> Trust but verify.
> - **Focus:** GNSS Vulnerabilities (Low power signal), Spoofing (Fake signals), Jamming, and Defense mechanisms (IMU Dead Reckoning check, Array Antenna).
> - **Code:** A Python script `gps_defense.py` that fuses GPS and IMU. A "Spoofer" injects a teletransportation jump. The logic uses a Chi-Squared residual test to detect the jump and rejects the GPS update, relying on IMU integration.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** why GPS is easy to spoof (Unencrypted commercial L1 signal).
2.  **Implement** a Sensor Fusion integrity check (Innovation squared).
3.  **Simulate** a "Takeover Attack" where the car is slowly guided off course.
4.  **Demonstrate** Dead Reckoning as a failsafe.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None. (Real life: HackRF One for simulation in shielded cage).

### Software Environment
```bash
pip install numpy matplotlib filterpy
```

### Prior Knowledge
- Kalman Filtering (Week 4).
- GPS Triangulation.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Signal Strength Problem

GPS satellites are 20,000 km away.
*   **Power on Earth:** $-160 dBm$ (Barely above thermal noise).
*   **Attack:** A $\$200$ HackRF nearby transmits at $-50 dBm$.
*   **Result:** The receiver locks onto the louder (fake) signal.
*   **Effect:** Time shift $\to$ Position shift.

### 🔹 Part 2: Detection Strategies

How to know you are being spoofed?
1.  **Physical Strength:** Signal suddenly becomes 100x stronger? Suspicious.
2.  **Clock Jump:** Receiver clock bias jumps discontinuously.
3.  **Cross-Check:** "GPS says I am moving East at 100 km/h. IMU says I am stationary." $\to$ **Reject GPS.**

### 🔹 Part 3: Chi-Squared Test ($\chi^2$)

In Kalman Filter:
$$ y = z - Hx $$
$$ S = H P H^T + R $$
$$ \epsilon = y^T S^{-1} y $$
*   If $\epsilon > Threshold$ (e.g., 9.0), the measurement is statistically unlikely (3-sigma).
*   **Action:** Ignore $z$. Do not update state.

---

## 💻 Implementation: The Integrity Monitor

We simulate a vehicle driving straight. Detection is spoofed to jump 100m.

### 🛠️ Project Structure
```text
day163_gps_sec/
├── src/
│   ├── gps_defense.py
└── output/
    ├── spoof_plot.png
```

### 👨‍💻 GPS Defense (`src/gps_defense.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class IntegrityKF:
    def __init__(self):
        # 1D State: [position, velocity]
        self.x = np.array([0.0, 20.0]) # Linear motion 20 m/s
        self.P = np.diag([5.0, 1.0])
        
        # F: x = x + v*dt
        self.dt = 0.1
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])
        
        # Q: Process Noise (IMU/Physics uncertainty)
        self.Q = np.array([[0.1, 0.0], [0.0, 0.1]])
        
        # H: We measure Position
        self.H = np.array([[1.0, 0.0]])
        
        # R: Measurement Noise (GPS accuracy)
        self.R = np.array([[4.0]]) # Variance = 2m^2
        
        # Defense State
        self.gps_rejected_count = 0
        
    def predict(self, u=0):
        # x = Fx + Gu
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z):
        # 1. Calculate Residual
        y = z - self.H @ self.x
        
        # 2. Innovation Covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # 3. Mahalanobis Dist Squared (Chi-Square)
        # y is 1x1, S is 1x1 here
        nis = (y.T @ np.linalg.inv(S) @ y)[0,0]
        
        # THRESHOLD (99% confidence for 1-DOF is ~6.63. Let's use 9.0 for 3-sigma)
        threshold = 9.0 
        
        if nis > threshold:
            print(f"Warning: High Residual (NIS={nis:.1f}). Spoofing suspected! Rejecting GPS {z[0]:.1f}")
            self.gps_rejected_count += 1
            return False # Reject
        else:
            # Standard Update
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            I = np.eye(2)
            self.P = (I - K @ self.H) @ self.P
            return True # Accept

def main():
    kf = IntegrityKF()
    
    true_pos = []
    gps_meas = []
    est_pos = []
    
    print("Simulating Vehicle Trace...")
    
    # Sim Loop
    curr_x = 0.0
    curr_v = 20.0
    
    for t in range(50):
        # 1. Physics
        curr_x += curr_v * kf.dt
        true_pos.append(curr_x)
        
        # 2. GPS Measurement Generation
        if t == 25:
             # >>> ATTACK: Sudden 50m Jump <<<
             noise = 50.0 
        elif t > 25:
             # Spoofer maintains offset
             noise = 50.0
        else:
             # Normal noise
             noise = np.random.normal(0, 1.0)
             
        z = np.array([curr_x + noise])
        gps_meas.append(z[0])
        
        # 3. Filter
        kf.predict()
        accepted = kf.update(z)
        
        est_pos.append(kf.x[0])
        
    # Visualization
    plt.figure(figsize=(10, 6))
    time_axis = np.arange(0, 50) * kf.dt
    
    plt.plot(time_axis, true_pos, 'k--', label='True Position')
    plt.plot(time_axis, gps_meas, 'rx', label='GPS Measurement')
    plt.plot(time_axis, est_pos, 'b-', linewidth=2, label='KF Estimate (Fused)')
    
    # Highlight Attack
    plt.axvline(2.5, color='orange', linestyle=':', label='Spoofing Start')
    
    plt.title("GPS Spoofing Defense: Innovation Check")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()
    plt.savefig("output/spoof_plot.png")
    
    print(f"Total Rejected Updates: {kf.gps_rejected_count}")
    print("If KF worked, the Blue line should ignore the Red Crosses after t=2.5s")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Slow Drift"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** KF rejects the jump. Good.
- **Modify:** Change Spoofer. Instead of `+50`, use `noise += 0.5` per step (Ramp).
- **Result:** The residual `y` stays small (within 3-sigma).
- **Fail:** The KF accepts the "Drift". The car gets pulled off course slowly.
- **Lesson:** Residual checks catch *Jumps* (Step function). They fail against *Caryatid/Ramp* attacks. Defense requires Absolute Truth (e.g., Lidar Map matching, Visual Odometry).

---

## 🚀 Project: "Antenna Arrays"

**Goal:** Determine Angle of Arrival (AoA).
1.  **Satellites:** Come from the Sky (High Elevation).
2.  **Spoofers:** Come from the Ground (Horizontal).
3.  **Logic:** If signal is strong and Elevation < 5 degrees, it's a truck stop spoofer. Ignore it.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Rejected Valid GPS"
*   **Cause:** Tunnel usage. When exiting tunnel, GPS might re-acquire 20m away from IMU estimate (Drift).
*   **Result:** KF keeps rejecting Good GPS because IMU drifted too far.
*   **Fix:** Convergence Logic. If GPS is consistent at the new location for 5 seconds, "Snap" the KF to it (Reset).

#### 2. "Leap Seconds"
*   **Cause:** GPS time vs UTC.
*   **Fix:** Always use GPS Weeks + Seconds for internal logic.

---

## ⚡ Optimization: Tightly Coupled Fusion

Loosely coupled (Pos LLH) vs Tightly Coupled (Pseudoranges).
*   **Tight:** Fuse raw satellite range data.
*   **Benefit:** If 1 satellite is spoofed, it becomes an outlier among 10 satellites. Easier to isolate than if the whole Computed Position is shifted.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Difference between Jamming and Spoofing?
    *   **A:** Jamming = Denial of Service (Noise). Spoofing = Deception (Fake Coordinates).
2.  **Q:** Why is encrypted GPS (M-Code) limited to Military?
    *   **A:** Key management. We can't distribute secret keys to 1 billion civilian cars. Solution: OSNMA (Open Service Navigation Message Authentication) in Galileo.
3.  **Q:** What is Dead Reckoning?
    *   **A:** Estimating position by integrating speed/heading from last known fix. Drifts over time ($\propto t^2$ for accelerometer).

### Challenge Task
> **Task:** Visual Odometry Cross-Check.
> 1. Use camera flow to estimate $V_{cam}$.
> 2. Compare with $V_{gps}$.
> 3. If $|V_{gps}| > 150 km/h$ (Impossible), Trigger Alarm.

---

## 📚 Further Reading
- **Humphreys et al.:** "Assessing the spoofing threat".
- **Galileo OSNMA:** Authentication spec for civilian GNSS.

---

**Day 163 Complete**
