# Day 65: Hungarian Algorithm (Optimal Assignment)
## Phase 4: ADAS & Robotics Systems | Week 10: Multi-Object Tracking

---

> **📝 Day 65 Focus:**
> Nearest Neighbor is greedy. It grabs the closest apple, even if that apple belongs to someone else. The **Hungarian Algorithm** (or Munkres Algorithm) is fair. It looks at *all* apples and *all* people and finds the assignment that minimizes the **Total Cost** for everyone.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Formulate** the Data Association problem as a Bipartite Graph Matching problem.
2.  **Construct** a Cost Matrix (Tracks vs Measurements).
3.  **Apply** the Hungarian Algorithm to find the optimal assignment.
4.  **Implement** a GNN (Global Nearest Neighbor) Tracker using `scipy.optimize.linear_sum_assignment`.
5.  **Compare** Greedy vs Optimal assignment in dense clutter scenarios.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 64:** Tracking Fundamentals.
-   **Matrix Algebra:** Cost Matrices.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `scipy` (Essential for Hungarian).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Assignment Problem

Given $N$ tracks and $M$ measurements.
We want to assign each track to at most one measurement.
**Cost Matrix $C$:** An $N \times M$ matrix where $C_{ij}$ is the distance (Euclidean/Mahalanobis) between Track $i$ and Measurement $j$.

**Goal:** Find a boolean matrix $X$ such that:
$$ \min \sum_{i,j} C_{ij} X_{ij} $$
Subject to:
-   $\sum_j X_{ij} \le 1$ (Each track has $\le 1$ meas).
-   $\sum_i X_{ij} \le 1$ (Each meas has $\le 1$ track).

### 🔹 Part 2: The Hungarian Algorithm (Munkres)

An $O(N^3)$ algorithm to solve this.
1.  **Subtract Row Min:** Subtract the minimum of each row from that row.
2.  **Subtract Col Min:** Subtract the minimum of each column from that column.
3.  **Cover Zeros:** Cover all zeros with minimum number of lines.
4.  **Create Zeros:** If lines < $N$, modify matrix to create more zeros.
5.  **Assign:** Select zeros such that no two are in same row/col.

*Note: In Python, we just use `scipy.optimize.linear_sum_assignment`.*

### 🔹 Part 3: Handling Unbalanced Assignment

What if $N \ne M$?
-   **More Meas than Tracks ($M > N$):** Some measurements are clutter (New Tracks).
-   **More Tracks than Meas ($N > M$):** Some tracks are missed.
-   **Gating:** Even optimal assignment might be bad (matching a track to a point 1km away). We must still apply a Gate (Max Cost).

---

## 💻 Implementation: Hungarian Tracker

**Scenario:**
-   **Tracks:** 3 crossing targets.
-   **Measurements:** Noisy + Clutter.
-   **Comparison:** We will see how it handles the "Crossing" better than Greedy NN.

### 🛠️ Setup
Create `week10_day65` and `hungarian_tracker.py`.

```bash
mkdir -p ~/ros2_ws/src/week10_day65
cd ~/ros2_ws/src/week10_day65
touch hungarian_tracker.py
```

### 👨‍💻 Code: Hungarian Tracker

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# --- Configuration ---
DT = 0.1
GATE_THRESH = 2.0 # Euclidean meters

class Track:
    def __init__(self, id, z):
        self.id = id
        self.x = np.array([[z[0]], [z[1]], [0], [0]]) # x, y, vx, vy
        self.P = np.eye(4) * 0.1
        self.missed = 0
        self.age = 1
        
    def predict(self):
        F = np.eye(4)
        F[0, 2] = DT
        F[1, 3] = DT
        Q = np.eye(4) * 0.01
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += 1
        
    def update(self, z):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        R = np.eye(2) * 0.1
        y = z.reshape(2, 1) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.missed = 0

class HungarianTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        
    def update(self, measurements):
        # 1. Predict
        for t in self.tracks:
            t.predict()
            
        # 2. Association (Hungarian)
        if len(self.tracks) == 0:
            for z in measurements:
                self.tracks.append(Track(self.next_id, z))
                self.next_id += 1
            return
            
        if len(measurements) == 0:
            for t in self.tracks:
                t.missed += 1
            return
            
        # Cost Matrix
        track_preds = np.array([t.x[:2, 0] for t in self.tracks])
        meas_arr = np.array(measurements)
        
        # Calculate Distance Matrix
        cost_matrix = cdist(track_preds, meas_arr)
        
        # Hungarian Algorithm
        # row_ind: Track Indices, col_ind: Measurement Indices
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assigned_tracks = set()
        assigned_meas = set()
        
        for r, c in zip(row_ind, col_ind):
            # Gating Check
            if cost_matrix[r, c] < GATE_THRESH:
                self.tracks[r].update(measurements[c])
                assigned_tracks.add(r)
                assigned_meas.add(c)
            # Else: Assignment rejected (too far), treat as missed/new
            
        # 3. Management
        # Missed Tracks
        for i in range(len(self.tracks)):
            if i not in assigned_tracks:
                self.tracks[i].missed += 1
                
        # New Tracks
        for i in range(len(measurements)):
            if i not in assigned_meas:
                self.tracks.append(Track(self.next_id, measurements[i]))
                self.next_id += 1
                
        # Delete Dead Tracks
        self.tracks = [t for t in self.tracks if t.missed < 5]

def main():
    tracker = HungarianTracker()
    
    # Crossing Scenario
    # Target 1: (0,0) -> (10,10)
    # Target 2: (0,10) -> (10,0)
    # They cross at (5,5)
    
    t1 = np.array([0.0, 0.0, 1.0, 1.0])
    t2 = np.array([0.0, 10.0, 1.0, -1.0])
    
    plt.figure(figsize=(8, 8))
    
    for i in range(20):
        # Move
        t1[0] += t1[2] * DT; t1[1] += t1[3] * DT
        t2[0] += t2[2] * DT; t2[1] += t2[3] * DT
        
        # Measure
        z1 = t1[:2] + np.random.randn(2) * 0.1
        z2 = t2[:2] + np.random.randn(2) * 0.1
        
        # At crossing (step 10), measurements are very close
        
        tracker.update([z1, z2])
        
        # Viz
        plt.cla()
        plt.xlim(-1, 11)
        plt.ylim(-1, 11)
        plt.plot(z1[0], z1[1], 'xg')
        plt.plot(z2[0], z2[1], 'xg')
        
        for t in tracker.tracks:
            plt.plot(t.x[0], t.x[1], 'ob')
            plt.text(t.x[0], t.x[1], str(t.id))
            
        plt.title(f"Step {i}")
        plt.pause(0.2)
        
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Crossing Test

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   At Step 10 (approx), the targets cross at (5, 5).
    -   The measurements are very close.
    -   Greedy NN might assign Meas 2 to Track 1 if it's slightly closer, leaving Meas 1 for Track 2 (bad) or unassigned.
    -   **Hungarian** minimizes the *sum* of distances. Even if Meas 2 is close to Track 1, assigning it there might force Track 2 to take a *very* far measurement (Meas 1). The algorithm sees this "Global Pain" and swaps them correctly.
3.  **Result:** ID 1 stays with Target 1. ID 2 stays with Target 2. No ID Switch.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Cost Matrix Size
**Symptom:** `ValueError: Cost matrix is not square`.
**Cause:** `linear_sum_assignment` handles non-square matrices automatically, but older versions might not.
**Solution:** Upgrade scipy. Or pad the matrix with dummy values (infinity) to make it square.

#### 2. Gating After Assignment
**Symptom:** Tracks jump to clutter.
**Cause:** Hungarian forces an assignment even if the cost is huge (1000m).
**Solution:** ALWAYS check `cost_matrix[r, c] < GATE` after getting the indices. If it fails, treat it as unassigned.

---

## ⚡ Optimization & Best Practices

### 1. Gating *Before* Hungarian
Calculating the full $N \times M$ matrix is wasteful if tracks are miles apart.
-   **Validation Matrix:** Create a sparse boolean matrix of feasible matches.
-   Split the problem into connected components (subgraphs) and solve Hungarian on smaller matrices.

### 2. Jaccard Distance (IoU)
For 2D Bounding Boxes (Object Detection), Euclidean distance is bad.
-   Use **1 - IoU** (Intersection over Union) as the cost.
-   Hungarian minimizes (1 - IoU) => Maximizes IoU.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the complexity of the Hungarian Algorithm?
    *   **A:** $O(N^3)$. For $N=1000$, this is $10^9$ ops (slow). For $N=50$, it's fast.
2.  **Q:** Why do we need to subtract row/col minimums?
    *   **A:** It creates zeros in the matrix without changing the optimal assignment. (Adding a constant to a row adds that constant to the total cost of any solution, so the relative order of solutions stays the same).
3.  **Q:** Can Hungarian handle "One-to-Many" (Splitting tracks)?
    *   **A:** No. It is strictly One-to-One. For One-to-Many, you need MHT (Multiple Hypothesis Tracking).

### Challenge Task
**Task:** IoU Association.
1.  Change the `Track` state to be a Bounding Box $[x, y, w, h]$.
2.  Change `cost_matrix` to calculate `1 - IoU`.
3.  Run Hungarian. This is the core of the **SORT** algorithm.

---

## 📚 Further Reading & References
-   [The Hungarian Method for the Assignment Problem (Kuhn)](https://onlinelibrary.wiley.com/doi/abs/10.1002/nav.3800020109)
-   [Scipy Linear Sum Assignment Docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)

---

**Day 65 Complete** | Phase 4: ADAS & Robotics Systems | Week 10: Multi-Object Tracking
