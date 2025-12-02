# Day 64: Tracking Fundamentals (Data Association)
## Phase 4: ADAS & Robotics Systems | Week 10: Multi-Object Tracking

---

> **📝 Day 64 Focus:**
> Detecting a car in one frame is easy. Knowing it's the *same* car in the next frame is hard. This is the **Tracking** problem. Today, we explore the core challenge: **Data Association**. How do we match new measurements to existing tracks?

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the Tracking Cycle: Predict -> Associate -> Update.
2.  **Explain** the Data Association problem: Measurement-to-Track assignment.
3.  **Implement** Gating (Validation Regions) to reject impossible matches.
4.  **Code** a Nearest Neighbor (NN) associator using Mahalanobis Distance.
5.  **Visualize** the association process and handle "Track Loss" vs "New Track".

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 51:** Kalman Filter (Predict/Update).
-   **Linear Algebra:** Euclidean vs Mahalanobis Distance.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `matplotlib`, `scipy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Tracking Loop

1.  **Prediction:** Move all existing tracks forward in time (using Motion Model).
    -   $x_{k|k-1} = F x_{k-1|k-1}$.
    -   $P_{k|k-1} = F P_{k-1|k-1} F^T + Q$.
2.  **Gating:** For each track, define a region where the measurement *could* be.
    -   Ellipsoidal Gate: $(z - \hat{z})^T S^{-1} (z - \hat{z}) < \gamma$.
3.  **Association:** Match Measurements ($z$) to Tracks ($x$).
    -   One-to-One? One-to-Many?
4.  **Update:** Correct the matched tracks (KF Update).
5.  **Management:** Create new tracks for unmatched measurements. Delete old tracks.

### 🔹 Part 2: Distance Metrics

-   **Euclidean:** $\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$.
    -   Good if uncertainty is circular and equal.
-   **Mahalanobis:** $\sqrt{(z - \hat{z})^T S^{-1} (z - \hat{z})}$.
    -   "Statistical Distance". Accounts for uncertainty ($S$).
    -   If $S$ is large (uncertain), a far point might still be a match (low statistical distance).

### 🔹 Part 3: Association Algorithms

-   **Nearest Neighbor (NN):** Greedy match. Pick the closest. Simple but prone to errors.
-   **Global Nearest Neighbor (GNN):** Minimize total cost (Hungarian Algorithm).
-   **JPDA / MHT:** Probabilistic (Soft) assignment.

---

## 💻 Implementation: Nearest Neighbor Tracker

**Scenario:**
-   **Tracks:** 2 targets moving in 2D.
-   **Measurements:** Noisy detections.
-   **Clutter:** Random false detections.

### 🛠️ Setup
Create `week10_day64` and `nn_tracker.py`.

```bash
mkdir -p ~/ros2_ws/src/week10_day64
cd ~/ros2_ws/src/week10_day64
touch nn_tracker.py
```

### 👨‍💻 Code: NN Tracker

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# --- Configuration ---
DT = 0.1
GATE_THRESH = 4.0 # Mahalanobis distance threshold (chi-square 2 DOF ~9.21 for 99%)
# Using Euclidean for simplicity in this first step, so threshold in meters
EUCLIDEAN_THRESH = 2.0 

class Track:
    def __init__(self, id, z):
        self.id = id
        self.x = np.array([[z[0]], [z[1]], [0], [0]]) # x, y, vx, vy
        self.P = np.eye(4) * 0.1
        self.age = 1
        self.missed = 0
        
    def predict(self):
        # Constant Velocity Model
        F = np.eye(4)
        F[0, 2] = DT
        F[1, 3] = DT
        Q = np.eye(4) * 0.01
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += 1
        
    def update(self, z):
        # Kalman Update
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        R = np.eye(2) * 0.1
        
        y = z.reshape(2, 1) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.missed = 0

class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        
    def update(self, measurements):
        # 1. Predict
        for t in self.tracks:
            t.predict()
            
        # 2. Association (Nearest Neighbor)
        if len(self.tracks) == 0:
            # All measurements are new tracks
            for z in measurements:
                self.tracks.append(Track(self.next_id, z))
                self.next_id += 1
            return
            
        if len(measurements) == 0:
            # All tracks missed
            for t in self.tracks:
                t.missed += 1
            return
            
        # Cost Matrix (Euclidean Distance)
        track_preds = np.array([t.x[:2, 0] for t in self.tracks])
        meas_arr = np.array(measurements)
        
        # Rows: Tracks, Cols: Measurements
        dists = cdist(track_preds, meas_arr)
        
        # Greedy Assignment
        assigned_tracks = set()
        assigned_meas = set()
        
        # Iterate through all possible matches sorted by distance
        # Flatten and sort indices
        flat_indices = np.argsort(dists, axis=None)
        
        for idx in flat_indices:
            r, c = np.unravel_index(idx, dists.shape)
            
            if r in assigned_tracks or c in assigned_meas:
                continue
                
            if dists[r, c] > EUCLIDEAN_THRESH:
                continue # Gating
                
            # Match found
            self.tracks[r].update(measurements[c])
            assigned_tracks.add(r)
            assigned_meas.add(c)
            
        # 3. Management
        # Unassigned Tracks -> Missed
        for i in range(len(self.tracks)):
            if i not in assigned_tracks:
                self.tracks[i].missed += 1
                
        # Unassigned Measurements -> New Tracks
        for i in range(len(measurements)):
            if i not in assigned_meas:
                self.tracks.append(Track(self.next_id, measurements[i]))
                self.next_id += 1
                
        # Delete Dead Tracks
        self.tracks = [t for t in self.tracks if t.missed < 5]

def main():
    tracker = Tracker()
    
    # Ground Truth
    target1 = np.array([0, 0, 1, 0.5]) # x, y, vx, vy
    target2 = np.array([10, 5, -1, 0.2])
    
    plt.figure(figsize=(10, 10))
    
    for i in range(50):
        # Move Targets
        target1[0] += target1[2] * DT
        target1[1] += target1[3] * DT
        target2[0] += target2[2] * DT
        target2[1] += target2[3] * DT
        
        # Generate Measurements (with noise)
        z1 = target1[:2] + np.random.randn(2) * 0.2
        z2 = target2[:2] + np.random.randn(2) * 0.2
        
        # Clutter (Random point)
        z3 = np.array([np.random.uniform(0, 10), np.random.uniform(0, 10)])
        
        measurements = [z1, z2]
        if i % 5 == 0: measurements.append(z3) # Add clutter occasionally
        
        # Tracker Update
        tracker.update(measurements)
        
        # Visualization
        plt.cla()
        plt.xlim(-5, 15)
        plt.ylim(-5, 15)
        
        # Plot Measurements
        for z in measurements:
            plt.plot(z[0], z[1], 'xg', label='Meas')
            
        # Plot Tracks
        for t in tracker.tracks:
            plt.plot(t.x[0], t.x[1], 'ob', label=f'Track {t.id}')
            plt.text(t.x[0], t.x[1], str(t.id))
            # Velocity vector
            plt.arrow(t.x[0, 0], t.x[1, 0], t.x[2, 0], t.x[3, 0], head_width=0.2)
            
        plt.title(f"Step {i} | Tracks: {len(tracker.tracks)}")
        plt.pause(0.1)
        
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Crossing

### Lab Objectives
1.  Run the simulation.
2.  **Observation:** Two tracks (ID 1, ID 2) are created. They follow the targets.
3.  **Experiment:**
    -   Make the targets cross paths (collide).
    -   Set `target2` start to `[0, 5]` and `vy = -0.5`. They will cross at `(2.5, 2.5)`.
    -   **Result:** The Nearest Neighbor tracker might swap IDs! (ID 1 becomes ID 2). This is the "ID Switch" problem.
    -   *Why?* At the crossing point, the measurement for Target 2 might be closer to Track 1's prediction than Target 1's measurement.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Track Fragmentation
**Symptom:** Track ID changes constantly (1 -> 3 -> 5).
**Cause:** Gating threshold too small. The measurement falls outside the gate, so a new track is created.
**Solution:** Increase `EUCLIDEAN_THRESH` or use Mahalanobis gating.

#### 2. Clutter Tracks
**Symptom:** Random tracks appear and disappear.
**Cause:** Clutter measurements create new tracks immediately.
**Solution:** **Confirmation Logic**. A track is "Tentative" until it has been matched $N$ times (e.g., 3 hits in 5 frames). Only then display it.

---

## ⚡ Optimization & Best Practices

### 1. Mahalanobis Gating
Euclidean assumes circular uncertainty.
Mahalanobis uses the Kalman Covariance $S$.
-   $d^2 = y^T S^{-1} y$.
-   Allows the gate to be an ellipse aligned with the uncertainty (e.g., long in velocity direction, narrow in cross-track).

### 2. KD-Tree for Association
For 1000 tracks and 1000 measurements, $N \times M$ matrix is slow.
-   Use a KD-Tree to find neighbors in $O(\log N)$.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is "Gating"?
    *   **A:** A hard threshold to reject unlikely matches. It reduces the search space for association.
2.  **Q:** Why is Nearest Neighbor greedy?
    *   **A:** It picks the best match for the *first* track it checks, without considering if that measurement would be *even better* for a later track.
3.  **Q:** How do we handle "Missed" tracks?
    *   **A:** We keep predicting them (Coast) for a few frames (Age/Coast count). If they are missed for too long (Time-to-Live), we delete them.

### Challenge Task
**Task:** Mahalanobis Implementation.
1.  Replace `cdist` with a loop that calculates Mahalanobis distance using `t.P` and `R`.
2.  Use `scipy.spatial.distance.mahalanobis`.
3.  Observe if it handles crossing targets better (if covariances are different).

---

## 📚 Further Reading & References
-   [Multiple View Geometry (Hartley & Zisserman)](https://www.amazon.com/Multiple-View-Geometry-Computer-Vision/dp/0521540518)
-   [Sort Paper (Simple Online and Realtime Tracking)](https://arxiv.org/abs/1602.00763)

---

**Day 64 Complete** | Phase 4: ADAS & Robotics Systems | Week 10: Multi-Object Tracking
