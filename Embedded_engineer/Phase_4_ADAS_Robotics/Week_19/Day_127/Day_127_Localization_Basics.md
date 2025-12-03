# Day 127: Localization Basics (GNSS/IMU)
## Phase 4: ADAS & Robotics Systems | Week 19: Localization & SLAM

---

> **📝 Day 127 Focus:**
> "Where am I?" is the most fundamental question for a robot. **Localization** answers this. We start with the basics: Satellites (**GNSS**) and Accelerometers (**IMU**). We will learn why GPS is not enough and how Dead Reckoning fills the gaps.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the working principle of GNSS (Triangulation) and RTK (Real-Time Kinematic).
2.  **Convert** Geodetic coordinates (Lat/Lon) to Cartesian (UTM/ENU).
3.  **Integrate** IMU data (Accel/Gyro) to perform Dead Reckoning.
4.  **Analyze** the drift characteristics of an IMU.
5.  **Simulate** a GNSS/IMU sensor stream.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Physics:** Kinematics ($x = x_0 + vt + 0.5at^2$).
-   **Coordinate Systems:** Earth Centered Earth Fixed (ECEF).

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `pymap3d` (for coordinate conversion), `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: GNSS (Global Navigation Satellite System)

-   **GPS (USA), Galileo (EU), GLONASS (RU), BeiDou (CN).**
-   **Principle:** Measure time-of-flight from 4+ satellites.
-   **Accuracy:**
    -   Standard Phone GPS: 5-10 meters.
    -   **RTK (Real-Time Kinematic):** Uses a base station to correct ionospheric errors. Accuracy: **2-3 cm**.
-   **Limitation:** Low update rate (1-10 Hz). Blocked by tunnels/buildings (Urban Canyon).

### 🔹 Part 2: IMU (Inertial Measurement Unit)

-   **Accelerometer:** Measures proper acceleration ($a_x, a_y, a_z$). Includes gravity!
-   **Gyroscope:** Measures angular velocity ($\omega_x, \omega_y, \omega_z$).
-   **Dead Reckoning:** Integrating accel/gyro to estimate position.
-   **Drift:** Errors accumulate quadratically with time ($x \propto t^2$). High-grade IMUs (Fiber Optic) drift less than MEMS (Phone).

### 🔹 Part 3: Coordinate Frames

1.  **WGS84 (Geodetic):** Latitude, Longitude, Altitude. (Ellipsoid).
2.  **ECEF:** X, Y, Z from Earth center.
3.  **ENU (East-North-Up):** Local tangent plane. Flat Cartesian grid. Used for local path planning.

---

## 💻 Implementation: Dead Reckoning Simulator

**Scenario:**
-   Car drives in a circle.
-   We have noisy IMU data (100 Hz) and noisy GPS data (1 Hz).
-   Task: Integrate IMU to estimate position between GPS updates.

### 🛠️ Setup
Create `week19_day127` and `dead_reckoning.py`.

```bash
mkdir -p ~/ros2_ws/src/week19_day127
cd ~/ros2_ws/src/week19_day127
pip install pymap3d
touch dead_reckoning.py
```

### 👨‍💻 Code: IMU Integration

```python
import numpy as np
import matplotlib.pyplot as plt

class IMU:
    def __init__(self, accel_noise, gyro_noise):
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        
    def measure(self, true_accel, true_gyro):
        # Add noise and bias (simplified)
        a = true_accel + np.random.normal(0, self.accel_noise, 2)
        w = true_gyro + np.random.normal(0, self.gyro_noise)
        return a, w

def main():
    # Config
    dt = 0.01 # 100 Hz
    duration = 10.0
    t = np.arange(0, duration, dt)
    
    # Ground Truth: Circle
    radius = 20.0
    speed = 5.0 # m/s
    omega = speed / radius # rad/s
    
    gt_x = radius * np.cos(omega * t)
    gt_y = radius * np.sin(omega * t)
    gt_yaw = omega * t + np.pi/2 # Tangent to circle
    
    # True IMU readings (Body Frame)
    # Centripetal Accel = v^2 / r
    a_lat = speed**2 / radius
    true_accel = np.array([0, a_lat]) # Longitudinal, Lateral
    true_gyro = omega
    
    # Sensor Setup
    imu = IMU(accel_noise=0.1, gyro_noise=0.01)
    
    # State Estimation
    est_x = [gt_x[0]]
    est_y = [gt_y[0]]
    est_yaw = gt_yaw[0]
    est_v = speed # Assume we know start speed
    
    curr_x, curr_y = gt_x[0], gt_y[0]
    
    print("Simulating Dead Reckoning...")
    for i in range(1, len(t)):
        # 1. Measure
        meas_a, meas_w = imu.measure(true_accel, true_gyro)
        
        # 2. Integrate Gyro (Yaw)
        est_yaw += meas_w * dt
        
        # 3. Integrate Accel (Velocity)
        # Note: Accelerometer measures force. In a turn, it measures centripetal.
        # But for simple DR, we usually integrate longitudinal accel.
        # Here, let's assume we integrate speed directly for simplicity, 
        # or assume we have wheel encoders (Odometry).
        # Pure IMU integration for position is very hard without gravity removal.
        # Let's assume we use the Gyro for orientation and constant speed (or noisy speed).
        
        # Simple Kinematic Model:
        # x += v * cos(yaw) * dt
        # y += v * sin(yaw) * dt
        
        curr_x += est_v * np.cos(est_yaw) * dt
        curr_y += est_v * np.sin(est_yaw) * dt
        
        est_x.append(curr_x)
        est_y.append(curr_y)
        
    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot(gt_x, gt_y, 'k--', label='Ground Truth')
    plt.plot(est_x, est_y, 'r-', label='Dead Reckoning (Gyro Only)')
    plt.title("Dead Reckoning Drift")
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

## 🔬 Lab Exercise: The Drift

### Lab Objectives
1.  Run the script.
2.  **Observation:** The Red line starts on the Black circle but slowly spirals away.
3.  **Experiment:**
    -   Increase `gyro_noise` to 0.1.
    -   **Result:** The spiral diverges much faster.
    -   **Calculation:** Error in angle $\theta_{err} \propto \int \text{noise} dt$. Position error $\propto \int \sin(\theta_{err}) dt$. It grows fast!
    -   **Lesson:** You cannot rely on IMU alone for more than a few seconds. You need GPS to "reset" the error.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Gravity Leakage
**Symptom:** Z-axis acceleration is always ~9.81.
**Cause:** Gravity.
**Solution:** You must subtract gravity vector (rotated by orientation) to get linear acceleration. This requires a very accurate orientation estimate (Quaternion).

#### 2. Coordinate Confusion
**Symptom:** GPS says North, IMU says X.
**Cause:** Frame mismatch.
**Solution:** Standardize on **ENU** (East-North-Up) or **NED** (North-East-Down). ROS uses ENU. Aviation uses NED.

---

## ⚡ Optimization & Best Practices

### 1. Wheel Odometry
IMU is noisy. Wheel encoders (counting ticks) are much better for distance.
-   **Fusion:** Use Encoders for Distance ($ds$) and Gyro for Angle ($d\theta$).
-   This is the standard "Odometry" for differential drive robots.

### 2. Dual Antenna GNSS
How to get Heading from GPS?
-   **Single Antenna:** Only knows Course-Over-Ground (COG) when moving. Undefined when stopped.
-   **Dual Antenna:** Two antennas on the roof. Can calculate heading even when stationary (Compass).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between GPS and GNSS?
    *   **A:** GPS is the US system. GNSS is the generic term for all satellite systems.
2.  **Q:** Why does IMU drift?
    *   **A:** Because we are integrating noise. $\int (a + \epsilon) dt = v + \epsilon t$. $\int (v + \epsilon t) dt = x + 0.5 \epsilon t^2$. The error grows quadratically.
3.  **Q:** What is RTK?
    *   **A:** Real-Time Kinematic. It uses carrier-phase measurements and a base station to achieve cm-level accuracy.

### Challenge Task
**Task:** GPS Integration.
1.  Simulate GPS updates every 1.0s (100 steps).
2.  Reset `curr_x, curr_y` to the noisy GPS position every 100 steps.
3.  Observe the "Sawtooth" pattern. Smooth drift -> Jump correction -> Smooth drift.

---

## 📚 Further Reading & References
-   [Coordinate Systems for Navigation](https://www.mathworks.com/help/aeroblks/coordinate-systems.html)
-   [Kalman Filter for GPS/IMU Fusion](https://www.coursera.org/learn/state-estimation-localization-self-driving-cars)

---

**Day 127 Complete** | Phase 4: ADAS & Robotics Systems | Week 19: Localization & SLAM
