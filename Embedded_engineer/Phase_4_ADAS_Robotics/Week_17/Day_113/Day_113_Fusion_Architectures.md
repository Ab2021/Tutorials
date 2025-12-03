# Day 113: Sensor Fusion Architectures
## Phase 4: ADAS & Robotics Systems | Week 17: Sensor Fusion

---

> **📝 Day 113 Focus:**
> No single sensor is perfect. Cameras are blind in fog. LiDAR is blind in heavy rain. Radar has low resolution. **Sensor Fusion** combines them to create a "Super Sensor". Today, we design the architecture: Do we fuse raw pixels (Early) or object lists (Late)?

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Compare** Early Fusion (Low-Level), Mid-Level Fusion, and Late Fusion (Object-Level).
2.  **Analyze** the strengths and weaknesses of Camera, LiDAR, and Radar.
3.  **Design** a Centralized vs. Decentralized fusion architecture.
4.  **Explain** the concept of "Sensor Modality Complementarity".
5.  **Simulate** a voting logic for redundant sensors.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Weeks 11, 12, 16:** Camera, LiDAR, Radar basics.

### Hardware Requirements
-   **None:** System Design day.

### Software Stack
-   **Python:** `matplotlib` for visualization.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Sensor Triad

| Feature | Camera | LiDAR | Radar |
|---|---|---|---|
| **Range** | Poor (Stereo helps) | Excellent (cm) | Excellent (m) |
| **Velocity** | Poor (Optical Flow) | Poor (Differentiation) | Excellent (Doppler) |
| **Resolution** | High (Color/Texture) | High (3D Shape) | Low (Blobs) |
| **Weather** | Fails in Fog/Dark | Fails in Heavy Rain | Works in Everything |
| **Cost** | Low | High | Medium |

**Conclusion:** We need all three.

### 🔹 Part 2: Fusion Levels

1.  **Low-Level (Early) Fusion:**
    -   Fuse Raw Data (Pixels + Point Cloud).
    -   Example: Project LiDAR points onto Image -> RGB-D Image -> CNN.
    -   *Pros:* AI can learn complex correlations.
    -   *Cons:* High Bandwidth, Sensitive to calibration errors.
2.  **High-Level (Late) Fusion:**
    -   Each sensor detects objects independently.
    -   Fuse Object Lists (Track-to-Track Fusion).
    -   Example: Camera says "Car at (10,5)", Radar says "Object at (10.1, 5)". Fuse them.
    -   *Pros:* Modular, Low Bandwidth.
    -   *Cons:* Information loss (Thresholding drops weak signals).

### 🔹 Part 3: Architectures

-   **Decentralized (Smart Sensors):**
    -   Sensors have internal CPUs. They send Object Lists via CAN.
    -   Fusion ECU is simple.
-   **Centralized (Raw Data):**
    -   Sensors are "Dumb". They send raw data via LVDS/Ethernet.
    -   Central ECU (Orin/Xavier) does everything.
    -   *Trend:* Moving towards Centralized for better AI performance.

---

## 💻 Implementation: Fusion Simulator

**Scenario:**
-   **Ground Truth:** Object at $x=50$.
-   **Camera:** $x=48 \pm 5$ (Noisy Range).
-   **Radar:** $x=50.1 \pm 0.5$ (Accurate Range).
-   **Task:** Fuse them using a Weighted Average (Simple Kalman-like logic).

### 🛠️ Setup
Create `week17_day113` and `fusion_arch.py`.

```bash
mkdir -p ~/ros2_ws/src/week17_day113
cd ~/ros2_ws/src/week17_day113
touch fusion_arch.py
```

### 👨‍💻 Code: Weighted Fusion

```python
import numpy as np
import matplotlib.pyplot as plt

class Sensor:
    def __init__(self, name, sigma):
        self.name = name
        self.sigma = sigma # Standard Deviation (Uncertainty)
        
    def measure(self, true_val):
        return true_val + np.random.normal(0, self.sigma)

def fuse_measurements(z1, sigma1, z2, sigma2):
    # Optimal Fusion (Inverse Variance Weighting)
    # x = (w1*z1 + w2*z2) / (w1 + w2)
    # w = 1 / sigma^2
    
    w1 = 1 / (sigma1**2)
    w2 = 1 / (sigma2**2)
    
    x_fused = (w1 * z1 + w2 * z2) / (w1 + w2)
    
    # Fused Uncertainty
    sigma_fused = np.sqrt(1 / (w1 + w2))
    
    return x_fused, sigma_fused

def main():
    # Setup
    true_pos = 50.0
    
    # Camera: Good Angle, Bad Range
    cam = Sensor("Camera", sigma=5.0)
    
    # Radar: Bad Angle, Good Range
    rad = Sensor("Radar", sigma=0.5)
    
    # Simulation
    n_steps = 100
    results = []
    
    for _ in range(n_steps):
        z_cam = cam.measure(true_pos)
        z_rad = rad.measure(true_pos)
        
        z_fused, sigma_fused = fuse_measurements(z_cam, cam.sigma, z_rad, rad.sigma)
        
        results.append({
            'cam': z_cam,
            'rad': z_rad,
            'fused': z_fused
        })
        
    # Plotting
    cam_data = [r['cam'] for r in results]
    rad_data = [r['rad'] for r in results]
    fused_data = [r['fused'] for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.plot(cam_data, 'r.', label=f'Camera (Sigma={cam.sigma})', alpha=0.3)
    plt.plot(rad_data, 'b.', label=f'Radar (Sigma={rad.sigma})', alpha=0.3)
    plt.plot(fused_data, 'g-', label='Fused Estimate', linewidth=2)
    plt.axhline(true_pos, color='k', linestyle='--', label='Ground Truth')
    
    # Calculate Error
    rmse_cam = np.sqrt(np.mean((np.array(cam_data) - true_pos)**2))
    rmse_rad = np.sqrt(np.mean((np.array(rad_data) - true_pos)**2))
    rmse_fused = np.sqrt(np.mean((np.array(fused_data) - true_pos)**2))
    
    plt.title(f"Sensor Fusion Demo\nRMSE: Cam={rmse_cam:.2f}, Rad={rmse_rad:.2f}, Fused={rmse_fused:.2f}")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Uncertainty Principle

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   Camera dots are scattered widely.
    -   Radar dots are tight.
    -   Fused line tracks the Radar closely.
    -   **RMSE Fused < RMSE Radar**. Fusion is *always* better than the best single sensor.
3.  **Experiment:**
    -   Set `cam.sigma = 0.5` and `rad.sigma = 5.0` (Simulating Angle measurement where Camera is better).
    -   **Result:** The Fused line now tracks the Camera.
    -   **Lesson:** The fusion algorithm automatically trusts the better sensor.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Overconfidence
**Symptom:** Fused estimate is biased.
**Cause:** Underestimating `sigma`. If you tell the filter "I am 100% sure" (sigma=0.001) but your sensor is noisy, the filter will ignore the other sensors and follow the noise.
**Solution:** Tune `sigma` based on real-world variance analysis.

#### 2. Outliers
**Symptom:** A glitch (Radar reflection) pulls the fused estimate away.
**Cause:** Gaussian assumption. Real noise has heavy tails.
**Solution:** **Gating**. If a measurement is $> 3\sigma$ away from prediction, reject it.

---

## ⚡ Optimization & Best Practices

### 1. ROI Fusion (Region of Interest)
Don't process the whole image with LiDAR.
-   Radar detects object at $(u, v)$.
-   Project Radar point to Image.
-   Crop a small ROI around $(u, v)$.
-   Run CNN on ROI.
-   **Benefit:** Massive speedup.

### 2. Temporal Consistency
Don't just fuse spatially. Fuse temporally.
-   If Camera sees a car at $t=1$ and $t=3$, but misses it at $t=2$, the Fusion Tracker should fill the gap (Prediction).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the main advantage of Late Fusion?
    *   **A:** Modularity. You can swap the Radar supplier without rewriting the Camera code.
2.  **Q:** Why is Early Fusion harder?
    *   **A:** Requires precise calibration (pixel-to-point alignment) and high bandwidth (raw data transmission).
3.  **Q:** If Radar fails, what happens to the fused variance?
    *   **A:** It increases. $1/\sigma_{f}^2 = 1/\sigma_{c}^2 + 0$. The system becomes less confident but keeps working (Fail-Operational).

### Challenge Task
**Task:** 3-Sensor Fusion.
1.  Add `lidar = Sensor("LiDAR", sigma=1.0)`.
2.  Update formula: $w_{total} = w_c + w_r + w_l$.
3.  $x_{fused} = (w_c z_c + w_r z_r + w_l z_l) / w_{total}$.
4.  Observe RMSE improvement.

---

## 📚 Further Reading & References
-   [Sensor Fusion for Autonomous Vehicles (Medium)](https://medium.com/@kyle.c.sullivan/sensor-fusion-for-autonomous-vehicles-a-review-8a6f6f6f6f6f)
-   [Kalman Filter Explained](https://www.kalmanfilter.net/)

---

**Day 113 Complete** | Phase 4: ADAS & Robotics Systems | Week 17: Sensor Fusion
