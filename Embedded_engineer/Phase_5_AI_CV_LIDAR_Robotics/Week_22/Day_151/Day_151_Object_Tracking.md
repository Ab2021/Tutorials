# Day 151: Object Tracking (SORT/DeepSORT)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 22: Autonomous Driving Stack

---

> **📝 Content Creator Instructions:**
> Detection is momentary. Tracking is historical.
> - **Focus:** Multi-Object Tracking (MOT), Data Association (Hungarian Algorithm), Kalman Filtering for Bounding Boxes, handling Occlusion, and Deep Association (Re-ID).
> - **Code:** A Python script `sort_tracker.py` implementing the SORT algorithm manually. It must associate noisy detections across frames to maintain stable IDs.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Solve** the Data Association problem: "Which detection from Frame T matches Track #5 from Frame T-1?"
2.  **Apply** the Hungarian Algorithm (Munkres) to an IoU Cost Matrix.
3.  **Implement** a Constant Velocity Kalman Filter for box state $[u, v, s, r, \dot{u}, \dot{v}, \dot{s}]$.
4.  **Handle** ID Switches and Track Birth/Death logic.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy scipy filterpy matplotlib
```

### Prior Knowledge
- Kalman Filters (Week 4).
- IoU (Intersection over Union).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Detection vs Tracking

*   **Detector:** Says "Car at (10, 10)" then "Car at (11, 10)". It has no memory.
*   **Tracker:** Says "Track #42 moved from (10, 10) to (11, 10)".
*   **Challenges:**
    *   **Missed Detection:** Detector fails for 1 frame (Occlusion). Tracker must "coast".
    *   **False Positive:** Detector sees a ghost. Tracker not start a track immediately.
    *   **ID Switch:** Track #42 becomes Track #43 (Bad).

### 🔹 Part 2: SORT (Simple Online and Realtime Tracking)

1.  **Predict:** Kalman Filter predicts next box location.
2.  **Measure:** Detector gives new boxes.
3.  **Match:** Calculate IoU between all Predictions and all Measurements.
    *   Cost Matrix $C_{ij} = 1 - IoU(Pred_i, Meas_j)$.
    *   Use Hungarian Algo (`linear_sum_assignment`) to minimize cost.
4.  **Update:** Update Kalman Filter with matched measurement.

### 🔹 Part 3: DeepSORT

SORT fails if objects cross each other (Occupying same space). IoU is confused.
*   **Deep Extension:** Add "Appearance Feature" (Cos Distance of CNN Embedding).
*   If boxes overlap, check if they *look* the same.

---

## 💻 Implementation: SORT from Scratch

We track simulated bouncing boxes.

### 🛠️ Project Structure
```text
day151_tracking/
├── src/
│   ├── sort_tracker.py
└── output/
    ├── tracking_result.png
```

### 👨‍💻 The Tracker (`src/sort_tracker.py`)

Using `filterpy` for Kalman and `scipy` for Hungarian.

```python
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

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

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        # State: [x_center, y_center, scale(area), ratio, dx, dy, ds]
        # Ratio is constant (ideally)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # F: State Transition
        self.kf.F = np.eye(7)
        self.kf.F[:3, 4:] = np.eye(3) # CV Model for x, y, s
        
        # H: Measurement Function
        self.kf.H = np.eye(4, 7) # We measure x, y, s, r
        
        # P, R, Q matrices omitted for brevity (Need tuning)
        self.kf.P *= 10.0
        self.kf.R *= 1.0 # Measurement noise
        self.kf.Q *= 0.01 # Process noise
        
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 0
        
        # Initialize
        # Convert bbox [x1, y1, x2, y2] to [xc, yc, s, r]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.kf.x[:4] = np.array([bbox[0]+w/2, bbox[1]+h/2, w*h, w/float(h)]).reshape(4,1)

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Meas Update
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([bbox[0]+w/2, bbox[1]+h/2, w*h, w/float(h)]).reshape(4,1)
        self.kf.update(z)

    def predict(self):
        # Scale area can't be negative
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        return self.get_state()
        
    def get_state(self):
        # Convert back to [x1, y1, x2, y2]
        x = self.kf.x
        w = np.sqrt(x[2]*x[3])
        h = x[2]/w
        return [x[0]-w/2, x[1]-h/2, x[0]+w/2, x[1]+h/2]

class Sort:
    def __init__(self):
        self.trackers = []
        self.iou_threshold = 0.3
        self.max_age = 5 # Frames to keep missed track
        
    def update(self, detections):
        # detections: [ [x1,y1,x2,y2,score], ... ]
        
        # 1. Predict existing
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trk_rect = [pos[0], pos[1], pos[2], pos[3], 0]
            trks[t,:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)): to_del.append(t)
            
        self.trackers = [t for i, t in enumerate(self.trackers) if i not in to_del]
        
        # 2. Association
        matched, unmatched_dets, unmatched_trks = self.associate(detections, trks)
        
        # 3. Update Matched
        for t, d in matched:
            self.trackers[t].update(detections[d, :4])
            
        # 4. Create New
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i, :4]))
            
        # 5. Delete Dead
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        # Return Active
        ret = []
        for t in self.trackers:
            if t.time_since_update < 1 and t.hit_streak >= 3:
                pos = t.get_state()
                ret.append(np.concatenate((pos, [t.id])).reshape(1,-1))
        if len(ret) > 0: return np.concatenate(ret)
        return np.empty((0,5))

    def associate(self, detections, trackers):
        if(len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
            
        iou_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)
        for t, trk in enumerate(trackers):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = iou(trk, det)
                
        # Hungarian (Maximize IoU -> Minimize -IoU)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        matches = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] < self.iou_threshold:
                pass # Reject low overlap
            else:
                matches.append([r, c])
        matches = np.array(matches)
        
        # Unmatched
        if len(matches) == 0:
            unmatched_trks = np.arange(len(trackers))
            unmatched_dets = np.arange(len(detections))
        else:
            unmatched_trks = []
            for t in range(len(trackers)):
                if t not in matches[:,0]: unmatched_trks.append(t)
            unmatched_dets = []
            for d in range(len(detections)):
                if d not in matches[:,1]: unmatched_dets.append(d)
                
        return matches, np.array(unmatched_dets), np.array(unmatched_trks)

def main():
    tracker = Sort()
    
    # Simulate Object moving right
    # [x1, y1, x2, y2]
    gt_track = []
    for i in range(20):
        x = 10 + i * 5
        gt_track.append([x, 10, x+10, 20, 0.9])
        
    gt_track = np.array(gt_track)
    
    # Track
    print("Tracking Object ID...")
    for frame in range(20):
        dets = gt_track[frame:frame+1] # 1 object
        
        # Simulate Missed Detection at frame 10
        if frame == 10: dets = np.empty((0, 5)) 
        
        tracks = tracker.update(dets)
        
        print(f"Frame {frame}: Dets={len(dets)} | Tracks={len(tracks)}")
        if len(tracks) > 0:
            print(f"   -> Track ID: {int(tracks[0,4])} at {tracks[0,:4].round(1)}")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Occlusion"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** At Frame 10, Detector sees nothing.
- **Tracker:** Still outputs a box! (Prediction).
- **ID:** At Frame 11, Detector returns. Does ID stay 0?
- **Result:** Yes, because Frame 11 measurement matched the Frame 11 prediction (which drifted slightly during the gap).
- **Fail:** Increase gap to 10 frames. The Prediction variance ($P$) grows. The Box drifts. Match fails. New ID assigned (ID Switch).

---

## 🚀 Project: "Multi-Camera Tracking"

**Goal:** Global ID.
1.  **Cam 1:** Tracks ID #5 (Red Shirt) exiting Left.
2.  **Cam 2:** Detects new Person entering Right.
3.  **Assoc:** DeepSORT embedding matches ID #5.
4.  **Result:** Handover ID #5 to Cam 2.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Box Explosion"
*   **Cause:** KF Scale prediction unstable. Area becomes negative or huge.
*   **Fix:** Clamp area/aspect ratio in prediction step.

#### 2. "Lag"
*   **Cause:** Low FPS. Object moves too far between frames. IoU is 0.
*   **Fix:** High FPS (30+). Or use more robust detector. Or use Texture Matching (Optical Flow) instead of just Box IoU.

---

## ⚡ Optimization: JDE (Joint Detection and Embedding)

Running YOLO + ResNet (for DeepSORT) is slow.
*   **JDE:** One Network.
*   **Head 1:** Box/Class.
*   **Head 2:** Re-ID Embedding (Vector of 128 floats).
*   **Speed:** Real-time MOT.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why Kalman Filter?
    *   **A:** It smooths the jittery detections and gives a velocity estimate for prediction.
2.  **Q:** What is the "Mahalanobis Distance"?
    *   **A:** A distance metric that accounts for state covariance. If the KF is very uncertain (Big P covariance), a measurement far away might still be a valid match (Low Mahalanobis dist).
3.  **Q:** ID Switch vs Fragment?
    *   **A:** Switch: ID 1 becomes ID 2. Fragment: ID 1 disappears, then ID 1 reappears later (Gap). Switch is worse for behavior prediction.

### Challenge Task
> **Task:** Radial Velocity (Radar).
> 1. Use Radar data ($r, \dot{r}$).
> 2. Fuse with Camera Box.
> 3. Use Radar velocity to update KF state more accurately than visual pixel delta.

---

## 📚 Further Reading
- **Bewley et al.:** "Simple Online and Realtime Tracking" (SORT Paper).
- **Wojke et al.:** "Simple Online and Realtime Tracking with a Deep Association Metric" (DeepSORT).

---

**Day 151 Complete**
