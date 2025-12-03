# Day 117: Multi-Object Tracking (GNN & Hungarian Algorithm)
## Phase 4: ADAS & Robotics Systems | Week 17: Sensor Fusion

---

> **📝 Day 117 Focus:**
> The Kalman Filter tracks *one* object. But the road has many cars. If Lidar gives us 5 points and we have 3 tracks, which point updates which track? This is the **Data Association** problem. Today, we solve it using the **Hungarian Algorithm**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the Assignment Problem (Tracks vs Measurements).
2.  **Calculate** the Cost Matrix using Euclidean or Mahalanobis Distance.
3.  **Apply** the Hungarian Algorithm (Munkres) to find the optimal assignment.
4.  **Handle** Unassigned Tracks (Deletion) and Unassigned Measurements (Creation).
5.  **Implement** a basic Multi-Object Tracker.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 115:** Kalman Filter.
-   **Linear Algebra:** Matrices.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `scipy.optimize.linear_sum_assignment` (Hungarian Algo).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Association Problem

-   **Tracks ($T$):** Existing objects we are tracking (Predictions).
-   **Measurements ($M$):** New detections from sensors.
-   **Goal:** Match $T_i$ to $M_j$ such that the total error is minimized.

### 🔹 Part 2: The Cost Matrix

We create a matrix $C$ where $C_{ij}$ is the "distance" between Track $i$ and Measurement $j$.
-   **Euclidean Distance:** $\sqrt{(x_T - x_M)^2 + (y_T - y_M)^2}$.
-   **Mahalanobis Distance:** Normalized by uncertainty ($P$).
    $$ d_M = \sqrt{(z - Hx)^T S^{-1} (z - Hx)} $$
    -   If the covariance is large (uncertain), we allow larger distances.

### 🔹 Part 3: The Hungarian Algorithm

A combinatorial optimization algorithm that solves the assignment problem in $O(n^3)$.
-   Input: Cost Matrix.
-   Output: List of pairs $(i, j)$ that minimizes $\sum C_{ij}$.

### 🔹 Part 4: Track Lifecycle

1.  **Unassigned Measurement:** Create a new Tentative Track.
2.  **Tentative Track:** If matched for $N$ frames -> Confirmed.
3.  **Unassigned Track:** If not matched for $K$ frames -> Deleted.

---

## 💻 Implementation: Multi-Object Tracker

**Scenario:**
-   3 Tracks (Predictions).
-   4 Measurements (3 real, 1 clutter).
-   Task: Associate and Update.

### 🛠️ Setup
Create `week17_day117` and `mot_hungarian.py`.

```bash
mkdir -p ~/ros2_ws/src/week17_day117
cd ~/ros2_ws/src/week17_day117
touch mot_hungarian.py
```

### 👨‍💻 Code: GNN Tracker

```python
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

class Track:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.age = 0
        self.missed_frames = 0
        
    def predict(self):
        # Simple Constant Velocity Prediction (Mock)
        self.x += 1.0 # Moving right
        self.age += 1
        
    def update(self, meas_x, meas_y):
        # Simple Update (Mock KF)
        self.x = 0.8 * self.x + 0.2 * meas_x
        self.y = 0.8 * self.y + 0.2 * meas_y
        self.missed_frames = 0

def calculate_cost_matrix(tracks, measurements):
    n_tracks = len(tracks)
    n_meas = len(measurements)
    cost_matrix = np.zeros((n_tracks, n_meas))
    
    for i, trk in enumerate(tracks):
        for j, meas in enumerate(measurements):
            # Euclidean Distance
            dist = np.sqrt((trk.x - meas[0])**2 + (trk.y - meas[1])**2)
            cost_matrix[i, j] = dist
            
    return cost_matrix

def main():
    # --- Setup ---
    tracks = [
        Track(1, 10, 10),
        Track(2, 20, 20),
        Track(3, 30, 30)
    ]
    
    # Measurements:
    # 1. Matches Track 1 (11, 10)
    # 2. Matches Track 2 (21, 21)
    # 3. Matches Track 3 (29, 31)
    # 4. Clutter (50, 50)
    measurements = [
        [11, 10],
        [21, 21],
        [29, 31],
        [50, 50]
    ]
    
    print("--- Step 1: Prediction ---")
    for t in tracks:
        t.predict()
        print(f"Track {t.id} Predicted at ({t.x:.1f}, {t.y:.1f})")
        
    print("\n--- Step 2: Cost Matrix ---")
    cost_matrix = calculate_cost_matrix(tracks, measurements)
    print(np.round(cost_matrix, 1))
    
    print("\n--- Step 3: Hungarian Assignment ---")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Gating Threshold (Max distance to accept a match)
    GATE = 5.0
    
    assigned_tracks = set()
    assigned_meas = set()
    
    matches = []
    
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < GATE:
            matches.append((r, c))
            assigned_tracks.add(r)
            assigned_meas.add(c)
            print(f"Match: Track {tracks[r].id} <-> Meas {c} (Cost: {cost_matrix[r,c]:.1f})")
        else:
            print(f"Reject: Track {tracks[r].id} <-> Meas {c} (Cost too high)")
            
    print("\n--- Step 4: Update / Create / Delete ---")
    
    # Update Matched
    for r, c in matches:
        tracks[r].update(measurements[c][0], measurements[c][1])
        print(f"Track {tracks[r].id} Updated.")
        
    # Create New Tracks (Unassigned Measurements)
    for j in range(len(measurements)):
        if j not in assigned_meas:
            print(f"New Track Created from Meas {j} at {measurements[j]}")
            tracks.append(Track(len(tracks)+1, measurements[j][0], measurements[j][1]))
            
    # Delete Lost Tracks (Unassigned Tracks)
    for i in range(len(tracks)):
        if i not in assigned_tracks and i < 3: # Only check original tracks
            tracks[i].missed_frames += 1
            print(f"Track {tracks[i].id} Missed Measurement.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Crossing

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   Tracks 1, 2, 3 find their measurements.
    -   Measurement 4 (Clutter) creates a New Track (Track 4).
3.  **Experiment:**
    -   Move Measurement 1 to `(20, 20)` (Close to Track 2).
    -   **Result:** The Hungarian Algorithm will decide globally.
    -   If Meas 1 is closer to Track 2 than Meas 2 is, it might steal it!
    -   This is the **Track Switch** problem.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. ID Switching
**Symptom:** Track 1 becomes Track 2 after they cross paths.
**Cause:** Position-only association fails when objects overlap.
**Solution:** Include Velocity in the Cost Matrix. Or use Appearance (Color/ReID) features.

#### 2. Track Fragmentation
**Symptom:** One car generates 10 tracks over 10 seconds.
**Cause:** Deletion threshold too low. Creation threshold too low.
**Solution:** Require $N=3$ consecutive matches to confirm a track. Keep tracks alive for $K=5$ frames (Coasting) before deletion.

---

## ⚡ Optimization & Best Practices

### 1. Gating (Validation Gate)
Don't calculate the cost for *every* pair.
-   If Track is at (0,0) and Meas is at (100,100), skip it.
-   **Ellipsoidal Gate:** Only consider measurements within the $3\sigma$ covariance ellipse ($d_M < \chi^2$).
-   This makes the cost matrix sparse and faster to solve.

### 2. JPDA (Joint Probabilistic Data Association)
GNN (Hungarian) makes a "Hard Decision" (1-to-1).
-   **JPDA:** Makes a "Soft Decision".
-   If Track 1 could be Meas A (60%) or Meas B (40%), it updates with a weighted average of both.
-   Better for clutter, but computationally heavy.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the complexity of the Hungarian Algorithm?
    *   **A:** $O(n^3)$. If you have 1000 tracks, it's slow.
2.  **Q:** Why use Mahalanobis Distance instead of Euclidean?
    *   **A:** Because it accounts for uncertainty. A measurement far away might still be the correct one if the sensor is very noisy (large covariance).
3.  **Q:** What happens to unassigned measurements?
    *   **A:** They initiate new tracks (potential new objects).

### Challenge Task
**Task:** Track Deletion Logic.
1.  Modify the code to delete tracks if `missed_frames > 3`.
2.  Simulate a track that gets no measurements for 5 frames.
3.  Verify it disappears from the list.

---

## 📚 Further Reading & References
-   [Hungarian Algorithm Explanation](https://brilliant.org/wiki/hungarian-matching-algorithm/)
-   [Multiple Object Tracking Review](https://arxiv.org/abs/1409.7618)

---

**Day 117 Complete** | Phase 4: ADAS & Robotics Systems | Week 17: Sensor Fusion
