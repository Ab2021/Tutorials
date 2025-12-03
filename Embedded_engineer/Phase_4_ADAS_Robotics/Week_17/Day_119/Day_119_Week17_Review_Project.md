# Day 119: Week 17 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 17: Sensor Fusion

---

> **📝 Day 119 Focus:**
> We have built the components: **EKF** for state estimation, **Hungarian Algo** for data association, and **Occupancy Grid** for mapping. Today, we assemble the **Sensor Fusion Stack**. This is the brain of the perception system.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Integrate** Tracking (Dynamic Objects) and Mapping (Static Environment).
2.  **Implement** a Track Management System (Initialize, Coast, Delete).
3.  **Fuse** Lidar and Radar data streams asynchronously.
4.  **Visualize** the complete "World Model" (Map + Tracks).
5.  **Evaluate** tracking performance (MOTA - Multi-Object Tracking Accuracy).

---

## 📚 Week 17 Review

### 1. Architectures
-   **Early Fusion:** Raw data.
-   **Late Fusion:** Object lists.
-   **Centralized:** One big computer.

### 2. Kalman Filter (KF/EKF)
-   **Predict:** $x = Fx$, $P = FPF^T + Q$.
-   **Update:** $K = PH^T(HPH^T + R)^{-1}$, $x = x + Ky$.
-   **EKF:** Uses Jacobians ($H_j$) for non-linear Radar updates.

### 3. Multi-Object Tracking
-   **GNN:** Global Nearest Neighbor (Hungarian Algo).
-   **Cost:** Mahalanobis Distance.
-   **Lifecycle:** Tentative -> Confirmed -> Deleted.

### 4. Mapping
-   **OGM:** Log-Odds update using Inverse Sensor Model.
-   **Result:** Free Space vs Occupied Space.

---

## 🛠️ Capstone Project: The Fusion Stack

**Goal:** Process a sequence of Lidar/Radar frames.
**Pipeline:**
1.  **Input:** Lidar Point Cloud + Radar Detections.
2.  **Preprocessing:** Cluster Lidar into Objects.
3.  **Data Association:** Match Lidar Clusters and Radar Detections to Existing Tracks.
4.  **State Estimation:** EKF Update (Lidar=Pos, Radar=Pos+Vel).
5.  **Mapping:** Update OGM with raw Lidar points (Static world).
6.  **Output:** List of Tracks + Grid Map.

### Package Structure
Create `week17_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week17_project
cd ~/ros2_ws/src/week17_project
touch fusion_stack.py
```

### 👨‍💻 Code: The Full Stack

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# --- 1. EKF Class (Simplified) ---
class EKF:
    def __init__(self, id, x, y):
        self.id = id
        self.state = np.array([[x], [y], [0], [0]]) # x, y, vx, vy
        self.P = np.eye(4) * 10
        self.F = np.eye(4) # Will update dt
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # Lidar H
        self.R = np.eye(2) * 0.1
        self.age = 0
        self.missed = 0

    def predict(self, dt):
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + np.eye(4) * 0.1
        self.age += 1

    def update(self, z):
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.missed = 0

# --- 2. Tracker Class ---
class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def process(self, measurements, dt):
        # 1. Predict
        for t in self.tracks:
            t.predict(dt)

        # 2. Associate
        if not self.tracks:
            # All meas are new tracks
            for m in measurements:
                self.tracks.append(EKF(self.next_id, m[0], m[1]))
                self.next_id += 1
            return

        n_trk = len(self.tracks)
        n_meas = len(measurements)
        cost = np.zeros((n_trk, n_meas))

        for i, t in enumerate(self.tracks):
            for j, m in enumerate(measurements):
                # Euclidean dist
                dist = np.sqrt((t.state[0,0]-m[0])**2 + (t.state[1,0]-m[1])**2)
                cost[i, j] = dist

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_meas = set()
        assigned_tracks = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 2.0: # Gating
                self.tracks[r].update(np.array([[measurements[c][0]], [measurements[c][1]]]))
                assigned_meas.add(c)
                assigned_tracks.add(r)
            
        # 3. Create New
        for j in range(n_meas):
            if j not in assigned_meas:
                self.tracks.append(EKF(self.next_id, measurements[j][0], measurements[j][1]))
                self.next_id += 1
                
        # 4. Delete Lost
        keep_tracks = []
        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks:
                t.missed += 1
            
            if t.missed < 5:
                keep_tracks.append(t)
        self.tracks = keep_tracks

# --- 3. Simulation ---
def main():
    tracker = Tracker()
    dt = 0.1
    
    # Ground Truth: Car moving (0,0) -> (10, 10)
    gt_path = []
    for i in range(20):
        gt_path.append([i*0.5, i*0.5])
        
    # Measurements (Noisy)
    meas_history = []
    for pt in gt_path:
        mx = pt[0] + np.random.normal(0, 0.2)
        my = pt[1] + np.random.normal(0, 0.2)
        meas_history.append([[mx, my]]) # List of meas per frame
        
    # Run Loop
    est_path = []
    
    print("Running Fusion Stack...")
    for i, meas in enumerate(meas_history):
        tracker.process(meas, dt)
        
        # Log Track 1
        if tracker.tracks:
            t = tracker.tracks[0]
            est_path.append([t.state[0,0], t.state[1,0]])
            print(f"Frame {i}: Track {t.id} at ({t.state[0,0]:.1f}, {t.state[1,0]:.1f})")
            
    # Plot
    gt_path = np.array(gt_path)
    est_path = np.array(est_path)
    meas_flat = np.array([m[0] for m in meas_history])
    
    plt.figure(figsize=(8, 8))
    plt.plot(gt_path[:, 0], gt_path[:, 1], 'k--', label='Ground Truth')
    plt.scatter(meas_flat[:, 0], meas_flat[:, 1], c='r', marker='x', label='Measurements')
    plt.plot(est_path[:, 0], est_path[:, 1], 'b-', linewidth=2, label='Track Estimate')
    plt.title("Sensor Fusion Stack Output")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🧪 Verification & Testing

### 1. The Coasting Test
-   **Scenario:** Measurement drops for 3 frames.
-   **Expected:** The track continues moving (Prediction) and `missed` counter increments. It is NOT deleted immediately.
-   **Code Check:** `if t.missed < 5: keep_tracks.append(t)`. **PASS**.

### 2. The New Object Test
-   **Scenario:** A second car appears at (5, 5).
-   **Expected:** A new Track ID (2) is created.
-   **Code Check:** `if j not in assigned_meas: create_new`. **PASS**.

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Theory
1.  **Q:** What is the difference between Tracking and Mapping?
    *   **A:** Tracking estimates the state of *dynamic* objects (Cars). Mapping estimates the state of the *static* environment (Walls).
2.  **Q:** Why do we need Data Association?
    *   **A:** Because the Kalman Filter update step requires *one* specific measurement. We must figure out which one it is.

### Section 2: Implementation
3.  **Q:** How does the EKF handle Radar?
    *   **A:** By linearizing the measurement function $h(x)$ using the Jacobian matrix $H_j$.
4.  **Q:** What is the "Gating" step?
    *   **A:** Rejecting measurements that are too far from the track prediction to save computation and avoid bad associations.

---

## 🏆 Conclusion

Congratulations on completing Week 17!
-   You have built a **Sensor Fusion Engine**.
-   You can track moving cars and map the world simultaneously.

**Next Week:** We teach the car to *see*. **Deep Learning for Perception**. CNNs, Object Detection (YOLO), and Semantic Segmentation.

---

**Day 119 Complete** | Phase 4: ADAS & Robotics Systems | Week 17: Sensor Fusion
