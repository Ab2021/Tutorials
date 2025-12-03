# Day 150: Occlusions and Shadows
## Phase 4: ADAS & Robotics Systems | Week 22: Edge Cases & Corner Cases

---

> **📝 Day 150 Focus:**
> Objects don't disappear just because you can't see them. A pedestrian walking behind a truck is still there. **Occlusion Handling** requires memory (Tracking). Also, **Shadows** look like obstacles to simple algorithms. We must distinguish dark patches from solid objects.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** Partial vs. Full Occlusion.
2.  **Implement** a "Coast" mode in Kalman Filter for tracking through occlusions.
3.  **Differentiate** shadows from obstacles using texture/edge analysis.
4.  **Predict** the re-appearance of occluded objects.
5.  **Visualize** the uncertainty covariance growing during occlusion.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 117:** Multi-Object Tracking (SORT).
-   **Day 114:** Kalman Filter (Prediction Step).

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `filterpy` (or custom KF), `opencv-python`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Occlusion Problem

-   **Partial Occlusion:** Only the head of a pedestrian is visible.
    -   *Solution:* Part-based detectors (Pose Estimation) or Amodal Segmentation (predicting the full shape).
-   **Full Occlusion:** Object is completely hidden behind a bus.
    -   *Solution:* **Object Permanence**. The tracker remembers the object and predicts its motion (Coast) until it reappears.

### 🔹 Part 2: Shadows vs Obstacles

-   **Shadow:** Dark region on road. No height. Texture is preserved (road grain visible).
-   **Obstacle:** Dark region. Has height (Lidar/Stereo). Texture changes.
-   **Problem:** Monocular vision often confuses shadows for potholes or cars.
-   **Solution:** Check edges. Shadows have soft edges (penumbra). Obstacles have hard edges. Use Stereo/Lidar for ground truth.

---

## 💻 Implementation: Tracking Through Occlusion

**Scenario:**
-   Target moves $x=0 \to 100$.
-   Wall at $x=40 \to 60$ blocks the sensor.
-   Task: Track the target through the wall.

### 🛠️ Setup
Create `week22_day150` and `occlusion_tracker.py`.

```bash
mkdir -p ~/ros2_ws/src/week22_day150
cd ~/ros2_ws/src/week22_day150
touch occlusion_tracker.py
```

### 👨‍💻 Code: Kalman Filter Coasting

```python
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self):
        # State: [x, vx]
        self.x = np.array([[0.0], [1.0]]) # Start at 0, vel=1
        self.P = np.eye(2) * 1.0
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]]) # CV Model
        self.H = np.array([[1.0, 0.0]]) # Measure position
        self.R = np.array([[0.5]]) # Measurement Noise
        self.Q = np.eye(2) * 0.01 # Process Noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0, 0]

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

def main():
    kf = KalmanFilter()
    
    # Ground Truth
    gt_x = np.arange(0, 100)
    
    # Measurements (with Occlusion)
    measurements = []
    for x in gt_x:
        if 40 <= x <= 60:
            measurements.append(None) # Occluded
        else:
            measurements.append(x + np.random.normal(0, 0.5))
            
    # Tracking
    est_x = []
    uncertainty = []
    
    for z in measurements:
        # 1. Predict (Always runs)
        pred = kf.predict()
        
        # 2. Update (Only if measurement exists)
        if z is not None:
            kf.update(z)
            
        est_x.append(kf.x[0, 0])
        uncertainty.append(kf.P[0, 0])
        
    # Plot
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(gt_x, label='Ground Truth')
    plt.plot(measurements, 'g.', label='Measurements')
    plt.plot(est_x, 'r--', label='KF Estimate')
    plt.axvspan(40, 60, color='gray', alpha=0.3, label='Occlusion')
    plt.legend()
    plt.title("Tracking Through Occlusion")
    
    plt.subplot(2, 1, 2)
    plt.plot(uncertainty)
    plt.title("Position Uncertainty (P)")
    plt.axvspan(40, 60, color='gray', alpha=0.3)
    plt.xlabel("Time Step")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Uncertainty Explosion

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   During occlusion (Grey zone), the Red line (Estimate) continues straight. This is the "Coast".
    -   The Uncertainty (Bottom plot) grows quadratically during occlusion.
    -   When measurement returns ($x=61$), the estimate snaps back to truth and uncertainty collapses.
3.  **Experiment:**
    -   Change `self.x` velocity to `1.5` initially (Wrong guess).
    -   **Result:** The coasting diverges from truth.
    -   **Lesson:** Prediction is only as good as the last estimate. Long occlusions are dangerous.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. ID Switching
**Symptom:** Object reappears but gets a new ID.
**Cause:** Covariance grew too large, or `max_age` (memory) was too short.
**Solution:** Increase `max_age` (frames to keep dead tracks). Use Appearance Features (Re-ID) to match the reappearing object to the old track.

#### 2. Ghost Tracks
**Symptom:** Tracker keeps predicting an object that actually stopped or turned during occlusion.
**Cause:** Constant Velocity assumption.
**Solution:** We can't solve this without data. But we can increase Process Noise ($Q$) during occlusion to reflect "anything could happen".

---

## ⚡ Optimization & Best Practices

### 1. Shadow Removal (Invariant Images)
-   Convert RGB to a shadow-invariant color space (based on log-chromaticity).
-   Shadows disappear, textures remain.
-   Computationally expensive but robust.

### 2. Negative Obstacles (Potholes)
-   Shadows project *on* the ground. Potholes go *into* the ground.
-   Lidar/Stereo is essential here.
-   **Geometry:** If the 3D points are *below* the ground plane, it's a pothole. If *on* the plane, it's a shadow/stain.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is "Coasting" in tracking?
    *   **A:** Continuing to predict the state (using the motion model) without performing the measurement update step.
2.  **Q:** Why does uncertainty grow during occlusion?
    *   **A:** Because $P_{k+1} = F P_k F^T + Q$. We keep adding Process Noise ($Q$) but never subtract Measurement Information.
3.  **Q:** How do we distinguish a shadow from a car?
    *   **A:** A car has 3D structure (Lidar points stick up). A shadow lies flat on the road.

### Challenge Task
**Task:** Shadow Detector.
1.  Load an image with shadows.
2.  Convert to HSV.
3.  Shadows usually have Low Value (V) but High Saturation (S) (blue skylight fills the shadow).
4.  Threshold based on Ratio $S/V$.

---

## 📚 Further Reading & References
-   [Shadow Detection Review](https://arxiv.org/abs/1711.03362)
-   [SORT Tracking Paper](https://arxiv.org/abs/1602.00763)

---

**Day 150 Complete** | Phase 4: ADAS & Robotics Systems | Week 22: Edge Cases & Corner Cases
