# Day 70: Week 10 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 10: Multi-Object Tracking

---

> **📝 Day 70 Focus:**
> We have built the components of a tracking system: Kalman Filter, Hungarian Algorithm, IoU, and Track Manager. Today, we assemble them into a complete **Highway Vehicle Tracker**. This is the system that tells the planner: "There is a car in front of you, moving at 25 m/s, and its ID is 42."

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Integrate** Detection, Tracking, and Management into a single pipeline.
2.  **Process** a sequence of frames (video stream) to maintain persistent IDs.
3.  **Handle** real-world challenges like occlusion, clutter, and interacting targets.
4.  **Visualize** tracking results with Bounding Boxes, IDs, and Velocity Vectors.
5.  **Evaluate** your tracker using the MOTA (Multi-Object Tracking Accuracy) concept.

---

## 📚 Week 10 Review

### 1. Data Association
-   **Problem:** Matching measurements to tracks.
-   **Gating:** Rejecting impossible matches (Mahalanobis/Euclidean).
-   **Nearest Neighbor:** Greedy, fast, prone to errors.
-   **Hungarian Algorithm:** Optimal, minimizes total cost.

### 2. Tracking Algorithms
-   **SORT:** Kalman Filter + Hungarian + IoU. Fast (260 Hz), good for simple scenes. Fails on occlusion.
-   **DeepSORT:** SORT + Appearance Embeddings (ReID). Handles occlusion better. Slower.
-   **AB3DMOT:** 3D Tracking using 3D Kalman Filter and 3D IoU.

### 3. Track Management
-   **Lifecycle:** New -> Tentative -> Confirmed -> Coasting -> Deleted.
-   **Confirmation:** M/N hits logic to reject clutter.
-   **Coasting:** Predicting position during occlusion to prevent ID switches.

---

## 🛠️ Capstone Project: Highway Vehicle Tracker

**Goal:** Track multiple vehicles on a highway.
**Inputs:** A list of detections for each frame (simulated).
**Outputs:** A video/plot showing stable IDs and trajectories.

### Package Structure
Create `week10_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week10_project
cd ~/ros2_ws/src/week10_project
touch highway_tracker.py
```

### 👨‍💻 Code: The Highway Tracker

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# --- Constants ---
IOU_THRESH = 0.3
MIN_HITS = 3
MAX_AGE = 5

# --- Helper Functions ---
def iou(bb_test, bb_gt):
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

# --- Kalman Filter ---
class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.state = 0 # 0=Tentative, 1=Confirmed
        
        # [u, v, s, r, ud, vd, sd]
        self.kf_x = np.zeros((7, 1))
        self.kf_P = np.eye(7) * 10.0
        self.kf_P[4:, 4:] *= 1000.0
        
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.kf_x[0] = bbox[0] + w/2
        self.kf_x[1] = bbox[1] + h/2
        self.kf_x[2] = w * h
        self.kf_x[3] = w / h

    def predict(self):
        F = np.eye(7); F[0,4]=1; F[1,5]=1; F[2,6]=1
        Q = np.eye(7) * 0.01; Q[4:,4:] *= 0.01
        self.kf_x = F @ self.kf_x
        self.kf_P = F @ self.kf_P @ F.T + Q
        self.age += 1
        if self.time_since_update > 0: self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        if self.hits >= MIN_HITS: self.state = 1 # Confirmed
        
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
        z = np.array([[bbox[0]+w/2], [bbox[1]+h/2], [w*h], [w/h]])
        H = np.eye(4, 7)
        R = np.eye(4) * 1.0; R[2,2]*=10.0; R[3,3]*=10.0
        y = z - H @ self.kf_x
        S = H @ self.kf_P @ H.T + R
        K = self.kf_P @ H.T @ np.linalg.inv(S)
        self.kf_x = self.kf_x + K @ y
        self.kf_P = (np.eye(7) - K @ H) @ self.kf_P

    def get_state(self):
        u=self.kf_x[0,0]; v=self.kf_x[1,0]; s=self.kf_x[2,0]; r=self.kf_x[3,0]
        w = np.sqrt(s * r); h = s / w
        return np.array([u-w/2, v-h/2, u+w/2, v+h/2])

# --- Tracker ---
class HighwayTracker:
    def __init__(self):
        self.trackers = []
        
    def update(self, dets):
        # 1. Predict
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)): to_del.append(t)
        for t in reversed(to_del): self.trackers.pop(t)
        
        # 2. Associate
        matched, unmatched_dets, unmatched_trks = self.associate(dets, trks)
        
        # 3. Update Matched
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])
                
        # 4. Create New
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :]))
            
        # 5. Output & Delete
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if trk.state == 1 and trk.time_since_update < 1:
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > MAX_AGE:
                self.trackers.pop(i)
        if len(ret) > 0: return np.concatenate(ret)
        return np.empty((0, 5))

    def associate(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = iou(det, trk)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.stack((row_ind, col_ind), axis=1)
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]: unmatched_detections.append(d)
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]: unmatched_trackers.append(t)
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < IOU_THRESH:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0: matches = np.empty((0, 2), dtype=int)
        else: matches = np.concatenate(matches, axis=0)
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def main():
    tracker = HighwayTracker()
    
    # Simulation: 3 Lanes
    # Car 1: Lane 1, Fast
    # Car 2: Lane 2, Slow
    # Car 3: Lane 3, Merging
    
    plt.figure(figsize=(10, 6))
    
    for i in range(50):
        dets = []
        
        # Car 1 (Fast)
        x1 = i * 10
        dets.append([x1, 10, x1+40, 50])
        
        # Car 2 (Slow)
        x2 = 200 + i * 5
        dets.append([x2, 60, x2+40, 100])
        
        # Car 3 (Merge: Y changes)
        x3 = i * 8
        y3 = 110 + (i/50.0) * 40 # 110 -> 150
        dets.append([x3, y3, x3+40, y3+40])
        
        # Clutter (Random)
        if i % 5 == 0:
            dets.append([np.random.uniform(0, 500), np.random.uniform(0, 200), 0, 0]) # Bad box
            # Fix box size
            dets[-1][2] = dets[-1][0] + 20
            dets[-1][3] = dets[-1][1] + 20
            
        tracks = tracker.update(np.array(dets))
        
        # Viz
        plt.cla()
        plt.xlim(0, 600)
        plt.ylim(0, 200)
        
        # Draw Lanes
        plt.plot([0, 600], [55, 55], '--k')
        plt.plot([0, 600], [105, 105], '--k')
        
        # Draw Tracks
        for t in tracks:
            w = t[2] - t[0]; h = t[3] - t[1]
            plt.gca().add_patch(plt.Rectangle((t[0], t[1]), w, h, fill=False, edgecolor='b', linewidth=2))
            plt.text(t[0], t[1]-5, f"ID:{int(t[4])}", color='b', fontsize=12, weight='bold')
            
        plt.title(f"Frame {i} | Tracks: {len(tracks)}")
        plt.pause(0.05)
        
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🧪 Verification & Testing

### 1. Persistence Test
-   **Observation:** Car 1 (ID 0), Car 2 (ID 1), Car 3 (ID 2) should maintain their IDs throughout the simulation.
-   **Result:** Even as Car 1 overtakes Car 3 (in X), their Y separation prevents ID switching.

### 2. Clutter Rejection
-   **Observation:** Random boxes appear every 5 frames.
-   **Result:** They should NOT appear in the Blue Track boxes because they don't last long enough (`MIN_HITS=3`) to become Confirmed.

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Algorithms
1.  **Q:** Why is the Kalman Filter essential for tracking?
    *   **A:** It predicts where the object *will be*, allowing us to search a smaller area (Gating) and handle missing detections (Coasting).
2.  **Q:** What is the difference between Local and Global association?
    *   **A:** Local (NN) is greedy. Global (Hungarian) optimizes the entire set of assignments.

### Section 2: Systems
3.  **Q:** How does this tracker handle a car leaving the frame?
    *   **A:** The track stops receiving updates. It coasts for `MAX_AGE` frames, then is deleted.
4.  **Q:** What happens if two cars overlap perfectly in 2D?
    *   **A:** 2D IoU becomes high. The tracker might get confused. This is why we need 3D tracking (Depth) or Appearance features (DeepSORT).

---

## 🏆 Conclusion

Congratulations on completing Week 10!
-   You have mastered **Multi-Object Tracking**.
-   You can now detect objects (Week 8), track them (Week 10), and plan paths around them (Week 9).

**Next Week:** We move to **Localization**. To plan a path, you need to know where *you* are. We will fuse GPS, IMU, and Odometry to estimate the robot's position with centimeter-level accuracy.

---

**Day 70 Complete** | Phase 4: ADAS & Robotics Systems | Week 10: Multi-Object Tracking
