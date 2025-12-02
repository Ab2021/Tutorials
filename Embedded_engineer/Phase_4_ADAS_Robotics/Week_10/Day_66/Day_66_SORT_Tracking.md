# Day 66: SORT (Simple Online and Realtime Tracking)
## Phase 4: ADAS & Robotics Systems | Week 10: Multi-Object Tracking

---

> **📝 Day 66 Focus:**
> We have the pieces: Kalman Filter (Day 51) and Hungarian Algorithm (Day 65). Now we put them together. **SORT** is a legendary algorithm (2016) that showed you don't need complex deep learning to track objects. Just good old geometry and physics.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the SORT architecture: Detection -> Kalman Predict -> IoU Match -> Kalman Update.
2.  **Implement** the Intersection over Union (IoU) metric for bounding boxes.
3.  **Design** a Constant Velocity Kalman Filter for Bounding Box state $[u, v, s, r]$.
4.  **Code** the complete SORT algorithm in Python.
5.  **Evaluate** SORT on a synthetic video sequence.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 65:** Hungarian Algorithm.
-   **Day 51:** Kalman Filter.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `scipy`, `filterpy` (Optional, we will write KF from scratch).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The State Space

SORT tracks Bounding Boxes (BBox).
State vector $x = [u, v, s, r, \dot{u}, \dot{v}, \dot{s}]^T$.
-   $u, v$: Center of the box (pixels).
-   $s$: Scale (Area = $w \times h$).
-   $r$: Aspect Ratio ($w / h$).
-   $\dot{u}, \dot{v}, \dot{s}$: Velocities.
-   Note: Aspect Ratio $r$ is assumed constant ($\dot{r} = 0$).

### 🔹 Part 2: The Cost Metric (IoU)

Euclidean distance is bad for boxes (doesn't account for size).
**IoU (Intersection over Union):**
$$ \text{IoU} = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} $$
-   IoU = 1.0 (Perfect Match).
-   IoU = 0.0 (No Overlap).
-   **Cost:** $1.0 - \text{IoU}$.

### 🔹 Part 3: The Algorithm

1.  **Prediction:** Propagate all tracks using KF.
2.  **Association:** Match Detections to Predicted Tracks using Hungarian on IoU Cost.
3.  **Update:** Correct matched tracks.
4.  **Creation:** Unmatched detections become new tracks.
5.  **Deletion:** Tracks with no match for $T_{lost}$ frames are deleted.

---

## 💻 Implementation: SORT Tracker

**Scenario:**
-   **Input:** List of detections per frame `[[x1, y1, x2, y2, score], ...]`.
-   **Output:** List of tracks `[[x1, y1, x2, y2, id], ...]`.

### 🛠️ Setup
Create `week10_day66` and `sort.py`.

```bash
mkdir -p ~/ros2_ws/src/week10_day66
cd ~/ros2_ws/src/week10_day66
touch sort.py
```

### 👨‍💻 Code: SORT Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# --- Kalman Filter for BBox ---
class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        # bbox: [x1, y1, x2, y2]
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # State: [u, v, s, r, u_dot, v_dot, s_dot]
        # u,v: center; s: area; r: aspect ratio
        self.kf_x = np.zeros((7, 1))
        self.kf_P = np.eye(7) * 10.0
        self.kf_P[4:, 4:] *= 1000.0 # High uncertainty for velocity initially
        
        # Initialize State
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.kf_x[0] = bbox[0] + w/2
        self.kf_x[1] = bbox[1] + h/2
        self.kf_x[2] = w * h
        self.kf_x[3] = w / h

    def predict(self):
        # Constant Velocity Model
        # F matrix (7x7)
        F = np.eye(7)
        F[0, 4] = 1.0
        F[1, 5] = 1.0
        F[2, 6] = 1.0
        
        # Q matrix (Process Noise)
        Q = np.eye(7) * 0.01
        Q[4:, 4:] *= 0.01
        
        self.kf_x = F @ self.kf_x
        self.kf_P = F @ self.kf_P @ F.T + Q
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        return self.get_state()

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        # Measurement: [u, v, s, r]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([
            [bbox[0] + w/2],
            [bbox[1] + h/2],
            [w * h],
            [w / h]
        ])
        
        # H matrix (Measurement)
        H = np.eye(4, 7)
        
        # R matrix (Measurement Noise)
        R = np.eye(4) * 1.0
        R[2, 2] *= 10.0
        R[3, 3] *= 10.0
        
        # KF Update
        y = z - H @ self.kf_x
        S = H @ self.kf_P @ H.T + R
        K = self.kf_P @ H.T @ np.linalg.inv(S)
        
        self.kf_x = self.kf_x + K @ y
        self.kf_P = (np.eye(7) - K @ H) @ self.kf_P

    def get_state(self):
        # Convert [u, v, s, r] back to [x1, y1, x2, y2]
        u = self.kf_x[0, 0]
        v = self.kf_x[1, 0]
        s = self.kf_x[2, 0]
        r = self.kf_x[3, 0]
        
        w = np.sqrt(s * r)
        h = s / w
        
        return np.array([
            u - w/2,
            v - h/2,
            u + w/2,
            v + h/2
        ])

# --- IoU Helper ---
def iou(bb_test, bb_gt):
    # bb: [x1, y1, x2, y2]
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + 
              (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o

# --- SORT Tracker ---
class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1
        
        # 1. Predict
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Clean up NaNs
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)
            
        # 2. Association
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)
        
        # 3. Update Matched
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])
                
        # 4. Create New
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
            
        # 5. Output & Delete
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            # Remove dead track
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
            
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = iou(det, trk)
                
        # Hungarian (Maximize IoU -> Minimize -IoU)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        matched_indices = np.stack((row_ind, col_ind), axis=1)
        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
                
        # Filter by Threshold
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
                
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
            
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def main():
    # Simulation
    tracker = Sort()
    
    # Target: Moving right
    # Frame 0: [10, 10, 50, 50]
    # Frame 1: [20, 10, 60, 50]
    
    plt.figure(figsize=(10, 5))
    
    for i in range(20):
        # Detection moves 10px right per frame
        x1 = 10 + i * 10
        det = np.array([[x1, 10, x1+40, 50, 0.9]])
        
        trackers = tracker.update(det)
        
        # Viz
        plt.cla()
        plt.xlim(0, 300)
        plt.ylim(0, 100)
        
        # Draw Det
        plt.gca().add_patch(plt.Rectangle((det[0,0], det[0,1]), 40, 40, fill=False, edgecolor='g', linewidth=2, label='Det'))
        
        # Draw Track
        for d in trackers:
            w = d[2] - d[0]
            h = d[3] - d[1]
            plt.gca().add_patch(plt.Rectangle((d[0], d[1]), w, h, fill=False, edgecolor='b', linewidth=2, label='Track'))
            plt.text(d[0], d[1]-5, f"ID: {int(d[4])}", color='b')
            
        plt.title(f"Frame {i}")
        plt.pause(0.1)
        
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Occlusion

### Lab Objectives
1.  Run the simulation.
2.  **Observation:** The Blue Box (Track) follows the Green Box (Detection). ID stays 0.
3.  **Experiment:**
    -   Simulate an occlusion: For frames 10-12, pass an empty list `[]` to `tracker.update()`.
    -   **Result:**
        -   Frame 10: Track predicts position (Coast). No update.
        -   Frame 13: Detection reappears.
        -   If `max_age` is high enough (e.g., 5), the track survives and re-associates. ID stays 0.
        -   If `max_age` is low (e.g., 1), the track dies. Frame 13 creates a NEW track (ID 1).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. ID Switching
**Symptom:** ID changes when objects pass close to each other.
**Cause:** IoU is zero if boxes don't overlap. If prediction is slightly off (fast motion), IoU=0, match fails.
**Solution:** Use **Mahalanobis Distance** as a fallback if IoU=0. Or use a higher frame rate.

#### 2. Box Explosion
**Symptom:** Predicted box becomes huge.
**Cause:** Unstable velocity estimate in KF (especially aspect ratio or scale).
**Solution:** Tune Process Noise $Q$. Assume constant aspect ratio ($Q_r = 0$).

---

## ⚡ Optimization & Best Practices

### 1. DeepSORT
SORT fails if occlusion lasts too long (KF drift).
**DeepSORT** adds visual features.
-   Extract "Appearance Embedding" (128D vector) from the box image using a CNN.
-   Cost = $\lambda \times (1 - \text{IoU}) + (1 - \lambda) \times \text{CosineDistance(Embeddings)}$.
-   Allows re-identification even after long occlusions.

### 2. OSPA Metric
How to evaluate tracking?
-   **MOTA (Multi-Object Tracking Accuracy):** Accounts for Misses, False Positives, and ID Switches.
-   **OSPA (Optimal Subpattern Assignment):** A geometric distance between two sets of tracks.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why do we track $[u, v, s, r]$ instead of $[x1, y1, x2, y2]$?
    *   **A:** $s$ (Area) and $r$ (Aspect Ratio) are more independent than corner coordinates. Width and Height are correlated.
2.  **Q:** What is the limitation of IoU matching?
    *   **A:** It requires overlap. If the object moves faster than its own width in one frame, IoU=0, and matching fails.
3.  **Q:** How does SORT handle new objects?
    *   **A:** Any detection not matched to an existing track spawns a new track.

### Challenge Task
**Task:** Implement "Coast" Visualization.
1.  Modify `main` to stop sending detections for 5 frames.
2.  Modify the plotting to show "Coasting" tracks in Red (Predicted but not Updated) and "Active" tracks in Blue.

---

## 📚 Further Reading & References
-   [SORT Paper (Bewley et al.)](https://arxiv.org/abs/1602.00763)
-   [FilterPy Documentation](https://filterpy.readthedocs.io/en/latest/)

---

**Day 66 Complete** | Phase 4: ADAS & Robotics Systems | Week 10: Multi-Object Tracking
