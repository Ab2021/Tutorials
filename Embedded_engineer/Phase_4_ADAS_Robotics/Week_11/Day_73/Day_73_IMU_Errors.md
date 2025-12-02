# Day 73: IMU Error Models (Bias, Drift, Noise)
## Phase 4: ADAS & Robotics Systems | Week 11: Localization

---

> **📝 Day 73 Focus:**
> GPS tells you where you *are*. IMU tells you where you are *going*. But IMUs are notorious liars. A small error in acceleration becomes a huge error in position after a few seconds. Today, we dissect the **Inertial Measurement Unit** and learn why "Dead Reckoning" is often "Dead Wrong".

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Differentiate** between Accelerometer (Linear Accel) and Gyroscope (Angular Rate).
2.  **Model** IMU errors: Static Bias, White Noise, and Bias Instability (Random Walk).
3.  **Simulate** the "Double Integration" problem and visualize position drift.
4.  **Perform** a Zero Velocity Update (ZUPT) to correct drift.
5.  **Analyze** an Allan Variance plot to characterize sensor quality.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Calculus:** Integration ($\int a dt = v, \int v dt = p$).
-   **Statistics:** Gaussian Noise, Variance.

### Hardware Requirements
-   **IMU:** (Optional) MPU-6050 or BNO055.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The IMU Measurement Model

**Accelerometer ($a_m$):**
$$ a_m = a_{true} + R(\theta) g + b_a + n_a $$
-   $a_{true}$: Real acceleration.
-   $g$: Gravity ($9.81 m/s^2$). The accelerometer measures gravity!
-   $b_a$: Bias (Constant offset).
-   $n_a$: Noise (White Gaussian).

**Gyroscope ($\omega_m$):**
$$ \omega_m = \omega_{true} + b_g + n_g $$
-   $b_g$: Bias (Drifts over time due to temperature).
-   $n_g$: Noise.

### 🔹 Part 2: The Drift Problem

To get position, we integrate acceleration twice:
$$ p(t) = \iint (a_{true} + b_a) dt^2 = \frac{1}{2} a_{true} t^2 + \frac{1}{2} b_a t^2 $$
-   **Quadratic Error:** A tiny bias $b_a = 0.01 m/s^2$ results in $0.5 \times 0.01 \times (100)^2 = 50$ meters error after 100 seconds!
-   **Gyro Drift:** Angle error grows linearly. Since gravity vector depends on angle ($R(\theta) g$), angle error causes gravity to leak into horizontal acceleration, causing **Cubic** position error!

### 🔹 Part 3: Allan Variance

How do we measure "Goodness"?
-   **White Noise (Angle Random Walk):** High frequency jitter.
-   **Bias Instability:** Low frequency wander (Temperature).
-   **Allan Deviation Plot:** Log-Log plot of error vs averaging time $\tau$. The minimum point is the Bias Instability.

---

## 💻 Implementation: IMU Drift Simulation

**Scenario:**
-   **Stationary Robot:** $a_{true} = 0, \omega_{true} = 0$.
-   **Sensor:** Noisy IMU with Bias.
-   **Goal:** Observe how position flies away.

### 🛠️ Setup
Create `week11_day73` and `imu_sim.py`.

```bash
mkdir -p ~/ros2_ws/src/week11_day73
cd ~/ros2_ws/src/week11_day73
touch imu_sim.py
```

### 👨‍💻 Code: IMU Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
DT = 0.01 # 100 Hz
DURATION = 60.0 # Seconds
STEPS = int(DURATION / DT)

# IMU Characteristics (Consumer Grade, e.g., MPU6050)
ACCEL_NOISE_STD = 0.05 # m/s^2
ACCEL_BIAS = 0.1 # m/s^2 (Uncalibrated)

GYRO_NOISE_STD = 0.01 # rad/s
GYRO_BIAS = 0.005 # rad/s

class IMUSim:
    def __init__(self):
        self.pos = 0.0
        self.vel = 0.0
        self.angle = 0.0
        
        # History
        self.pos_h = []
        self.vel_h = []
        self.angle_h = []
        self.t_h = []
        
    def run(self):
        for i in range(STEPS):
            t = i * DT
            
            # 1. Generate Measurements (Stationary)
            # True Accel = 0. True Gyro = 0.
            # Accel measures Gravity? Assume 1D horizontal for simplicity (Gravity removed)
            
            a_meas = 0.0 + ACCEL_BIAS + np.random.normal(0, ACCEL_NOISE_STD)
            w_meas = 0.0 + GYRO_BIAS + np.random.normal(0, GYRO_NOISE_STD)
            
            # 2. Dead Reckoning (Integration)
            
            # Update Angle
            self.angle += w_meas * DT
            
            # Update Velocity
            # In 2D, a_x = a_meas * cos(angle)
            # Here, 1D simplified:
            self.vel += a_meas * DT
            
            # Update Position
            self.pos += self.vel * DT
            
            # Store
            self.pos_h.append(self.pos)
            self.vel_h.append(self.vel)
            self.angle_h.append(self.angle)
            self.t_h.append(t)

    def plot(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        
        axs[0].plot(self.t_h, self.angle_h, 'r')
        axs[0].set_title('Angle Drift (Gyro Integration)')
        axs[0].set_ylabel('Rad')
        axs[0].grid()
        
        axs[1].plot(self.t_h, self.vel_h, 'g')
        axs[1].set_title('Velocity Drift (Accel Integration)')
        axs[1].set_ylabel('m/s')
        axs[1].grid()
        
        axs[2].plot(self.t_h, self.pos_h, 'b')
        axs[2].set_title('Position Drift (Double Integration)')
        axs[2].set_ylabel('Meters')
        axs[2].set_xlabel('Time (s)')
        axs[2].grid()
        
        plt.tight_layout()
        plt.show()

def main():
    print(f"Simulating {DURATION} seconds of stationary IMU...")
    sim = IMUSim()
    sim.run()
    
    final_pos = sim.pos_h[-1]
    print(f"Final Position Error: {final_pos:.2f} meters")
    print(f"Expected Error (0.5 * bias * t^2): {0.5 * ACCEL_BIAS * DURATION**2:.2f} meters")
    
    sim.plot()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The ZUPT Fix

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   Velocity grows linearly ($v = at$).
    -   Position grows quadratically ($p = 0.5at^2$).
    -   After 60s with 0.1 bias, error is ~180 meters!
3.  **Experiment (ZUPT):**
    -   Modify the loop: `if i % 100 == 0: self.vel = 0`.
    -   Simulate a "Zero Velocity Update" (e.g., robot stops every 1s).
    -   **Result:** Velocity error resets to 0. Position error grows linearly instead of quadratically. Much better!

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Gravity Leakage
**Symptom:** Huge acceleration when tilted.
**Cause:** Accelerometer measures $g$. If you don't subtract $g$ correctly (using the precise angle), the remainder looks like movement.
**Solution:** Use a complementary filter or Kalman Filter to estimate the gravity vector.

#### 2. Temperature Drift
**Symptom:** Bias changes when the robot warms up.
**Cause:** MEMS sensors are temperature sensitive.
**Solution:** Calibrate at different temperatures or let the sensor "warm up" for 10 mins before use.

---

## ⚡ Optimization & Best Practices

### 1. Calibration
Never use a raw IMU.
-   **Static Calibration:** Place on flat surface. Average 1000 samples. Subtract this mean (Bias) from future readings.
-   **6-Point Calibration:** Rotate IMU to all 6 faces (+X, -X, +Y, -Y, +Z, -Z) to find scale factors and biases.

### 2. Vibration Isolation
Motors cause high-frequency vibration.
-   **Aliasing:** If vibration > Nyquist Frequency, it folds down to low frequency noise.
-   **Damping:** Mount IMU on soft foam or rubber dampers.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why does position error grow quadratically?
    *   **A:** Because $p = \iint a dt^2$. A constant error in $a$ becomes $t$ in $v$ and $t^2$ in $p$.
2.  **Q:** What is "Random Walk"?
    *   **A:** The integral of white noise. It wanders around unpredictably. $\sigma \propto \sqrt{t}$.
3.  **Q:** Can we use IMU for long-term navigation?
    *   **A:** No. Not without external corrections (GPS, Cameras, LiDAR). It is only good for short-term dead reckoning (seconds).

### Challenge Task
**Task:** Allan Variance Calculation.
1.  Record 1 hour of static IMU data.
2.  Calculate variance of means for different averaging times $\tau$.
3.  Plot $\sigma(\tau)$ vs $\tau$ on Log-Log scale.
4.  Identify the "Bias Instability" (the bottom of the bucket).

---

## 📚 Further Reading & References
-   [An Introduction to Inertial Navigation (Oliver J. Woodman)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.pdf)
-   [Kalibr (IMU Calibration Tool)](https://github.com/ethz-asl/kalibr)

---

**Day 73 Complete** | Phase 4: ADAS & Robotics Systems | Week 11: Localization
