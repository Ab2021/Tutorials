# Day 67: DeepSORT (Deep Simple Online and Realtime Tracking)
## Phase 4: ADAS & Robotics Systems | Week 10: Multi-Object Tracking

---

> **📝 Day 67 Focus:**
> SORT is great, but it's blind. If a red car goes behind a truck and comes out, SORT thinks it's a new car because the Kalman Filter lost track. **DeepSORT** gives the tracker "eyes". It remembers what the car *looks* like (Appearance Embedding) and can re-identify it even after a long occlusion.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Identify** the limitations of SORT (Occlusion, ID Switching).
2.  **Explain** the DeepSORT architecture: Motion (Mahalanobis) + Appearance (Cosine Distance).
3.  **Implement** a Feature Extractor (Conceptually) using a CNN (e.g., ResNet).
4.  **Code** the "Cascade Matching" algorithm (Prioritize recent tracks).
5.  **Build** a simplified DeepSORT tracker that uses pre-computed embeddings.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 66:** SORT.
-   **Deep Learning:** CNNs, Embeddings.

### Hardware Requirements
-   **GPU:** Recommended for feature extraction (though we will simulate it).

### Software Stack
-   **Python:** `numpy`, `scipy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Appearance Embeddings

We crop the bounding box from the image and feed it into a CNN (trained on a ReID dataset like MARS or Market-1501).
-   **Output:** A vector $r$ (e.g., 128 floats) with $\|r\| = 1$.
-   **Property:** Images of the *same* car have small distance ($1 - r_i^T r_j$). Images of *different* cars have large distance.

### 🔹 Part 2: The Metric Ensemble

DeepSORT combines two metrics:
1.  **Motion ($D_m$):** Mahalanobis distance between Kalman prediction and measurement.
    -   Gating: $D_m < 9.48$ (Chi-square).
2.  **Appearance ($D_a$):** Smallest Cosine Distance between measurement embedding and track's *gallery* (history of embeddings).
    -   Gating: $D_a < 0.2$.

**Final Cost:** $C_{ij} = \lambda D_a + (1-\lambda) D_m$. (Usually just $D_a$ with $D_m$ gating).

### 🔹 Part 3: Cascade Matching

Instead of matching everyone at once, we prioritize tracks seen *recently*.
-   **Age 0:** Match detections to tracks seen in last frame.
-   **Age 1:** Match remaining detections to tracks missed for 1 frame.
-   ...
-   **Age $A_{max}$:** Match remaining detections to tracks missed for $A_{max}$ frames.

This prevents a new detection from being wrongly matched to a very old, uncertain track if a better, newer track exists.

---

## 💻 Implementation: DeepSORT (Simplified)

**Scenario:**
-   **Tracks:** Objects with position and "Color" (Embedding).
-   **Occlusion:** Object disappears and reappears.
-   **Goal:** Maintain ID across occlusion using Color.

### 🛠️ Setup
Create `week10_day67` and `deepsort.py`.

```bash
mkdir -p ~/ros2_ws/src/week10_day67
cd ~/ros2_ws/src/week10_day67
touch deepsort.py
```

### 👨‍💻 Code: DeepSORT Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# --- Configuration ---
MAX_DIST = 0.2 # Cosine distance threshold
MAX_AGE = 30
NN_BUDGET = 100 # Max embeddings per track

class Track:
    count = 0
    def __init__(self, detection, feature):
        self.id = Track.count
        Track.count += 1
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = 1 # 1=Confirmed, 2=Tentative, 3=Deleted
        
        # Motion State (Simplified: Just x, y)
        self.pos = detection[:2]
        
        # Appearance Gallery
        self.features = [feature]
        
    def predict(self):
        # Constant Velocity (Simplified)
        # In real DeepSORT, this is a Kalman Filter
        self.age += 1
        self.time_since_update += 1
        
    def update(self, detection, feature):
        self.pos = detection[:2]
        self.features.append(feature)
        if len(self.features) > NN_BUDGET:
            self.features.pop(0)
        self.hits += 1
        self.time_since_update = 0

def cosine_distance(a, b):
    # a: N x D, b: M x D
    # Assumes normalized vectors
    return 1. - np.dot(a, b.T)

class DeepSort:
    def __init__(self):
        self.tracks = []
        
    def update(self, detections, features):
        # detections: N x 2 (x, y)
        # features: N x 128
        
        # 1. Predict
        for t in self.tracks:
            t.predict()
            
        # 2. Matching Cascade
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.state == 1]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if t.state != 1]
        
        # Match Confirmed Tracks by Age
        matches_a = []
        unmatched_detections = list(range(len(detections)))
        
        for age in range(MAX_AGE):
            # Get tracks of this age
            trk_idx = [i for i in confirmed_tracks if self.tracks[i].time_since_update == age + 1]
            if len(trk_idx) == 0: continue
            
            # Get unmatched detections
            det_idx = unmatched_detections
            if len(det_idx) == 0: break
            
            # Calculate Cost Matrix (Appearance)
            # For each track, find min distance to its gallery
            cost_matrix = np.zeros((len(trk_idx), len(det_idx)))
            
            for i, ti in enumerate(trk_idx):
                track_features = np.array(self.tracks[ti].features)
                dists = cdist(track_features, features[det_idx], metric='cosine')
                # Min distance across gallery
                cost_matrix[i, :] = dists.min(axis=0)
                
            # Hungarian
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Update Matches
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < MAX_DIST:
                    matches_a.append((trk_idx[r], det_idx[c]))
                    if det_idx[c] in unmatched_detections:
                        unmatched_detections.remove(det_idx[c])
                        
        # 3. Update Tracks
        for t_idx, d_idx in matches_a:
            self.tracks[t_idx].update(detections[d_idx], features[d_idx])
            
        # 4. Create New Tracks
        for d_idx in unmatched_detections:
            self.tracks.append(Track(detections[d_idx], features[d_idx]))
            
        # 5. Delete Dead Tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < MAX_AGE]
        
        return self.tracks

def main():
    tracker = DeepSort()
    
    # Simulate Embeddings (Random unit vectors)
    # ID 1: "Red" (Vector A + noise)
    # ID 2: "Blue" (Vector B + noise)
    
    vec_A = np.random.rand(128); vec_A /= np.linalg.norm(vec_A)
    vec_B = np.random.rand(128); vec_B /= np.linalg.norm(vec_B)
    
    # Simulation
    # Frame 0-10: Both visible
    # Frame 11-20: ID 1 occluded (No detection)
    # Frame 21-30: ID 1 reappears
    
    plt.figure(figsize=(10, 5))
    
    for i in range(31):
        dets = []
        feats = []
        
        # Object 1 (Moves X: 0->30)
        if i < 11 or i > 20:
            pos1 = [i * 1.0, 10]
            feat1 = vec_A + np.random.randn(128) * 0.05
            feat1 /= np.linalg.norm(feat1)
            dets.append(pos1)
            feats.append(feat1)
            
        # Object 2 (Moves X: 0->30, Y: 20)
        pos2 = [i * 1.0, 20]
        feat2 = vec_B + np.random.randn(128) * 0.05
        feat2 /= np.linalg.norm(feat2)
        dets.append(pos2)
        feats.append(feat2)
        
        tracks = tracker.update(np.array(dets), np.array(feats))
        
        # Viz
        plt.cla()
        plt.xlim(0, 40)
        plt.ylim(0, 30)
        
        # Draw Detections
        for d in dets:
            plt.plot(d[0], d[1], 'xg', label='Det')
            
        # Draw Tracks
        for t in tracks:
            if t.time_since_update == 0:
                plt.plot(t.pos[0], t.pos[1], 'ob')
                plt.text(t.pos[0], t.pos[1], f"ID:{t.id}")
                
        plt.title(f"Frame {i}")
        plt.pause(0.1)
        
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: Re-Identification

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   Frame 0-10: ID 0 (Bottom) and ID 1 (Top) are tracked.
    -   Frame 11-20: ID 0 disappears (Occlusion).
    -   Frame 21: ID 0 reappears.
    -   **Result:** The tracker assigns ID 0 to the reappearing object!
    -   *Why?* The embedding of the new detection matches the gallery of ID 0 (Cosine Distance < 0.2).
3.  **Contrast:**
    -   If we used SORT (Motion only), the Kalman Filter covariance would explode during occlusion, or the track would be deleted. The reappearing object would get a NEW ID (ID 2).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. ID Switches in Crowds
**Symptom:** People wearing similar clothes (e.g., uniforms) swap IDs.
**Cause:** Appearance embeddings are too similar.
**Solution:** Rely more on Motion Gating ($D_m$). If they are spatially far, don't match even if they look alike.

#### 2. Slow Performance
**Symptom:** FPS drops with many tracks.
**Cause:** Calculating cosine distance between every detection and every track's *entire gallery* (100 features).
**Solution:** Keep only the last feature (Budget=1) or a moving average (EMA) of the features.

---

## ⚡ Optimization & Best Practices

### 1. TensorRT ReID
The CNN feature extractor is the bottleneck.
-   Export the ReID model (e.g., ResNet-50) to ONNX.
-   Optimize with TensorRT (FP16/INT8).
-   Run inference in batches.

### 2. Stronger ReID Models
-   **OSNet:** Omni-Scale Network designed for ReID.
-   **Triplet Loss:** Train the network specifically to minimize intra-class distance and maximize inter-class distance.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the "Gallery"?
    *   **A:** A list of the last $N$ appearance features extracted for a track. It helps handle view changes (front view vs side view).
2.  **Q:** Why do we need Cascade Matching?
    *   **A:** To prevent "greedy" matching of uncertain (old) tracks to detections that should belong to certain (new) tracks. We prioritize the "easy" matches first.
3.  **Q:** Can DeepSORT work without a Kalman Filter?
    *   **A:** Yes, purely on appearance. But it would fail if two people look alike. Motion provides a spatial constraint.

### Challenge Task
**Task:** Motion Gating.
1.  Add a simple Euclidean distance check inside the matching loop.
2.  If `dist(det, track) > 5.0`, set cost to Infinity.
3.  This prevents matching a "Red Car" on the left to a "Red Car" on the right instantly.

---

## 📚 Further Reading & References
-   [DeepSORT Paper (Wojke et al.)](https://arxiv.org/abs/1703.07402)
-   [OpenCV Tracking API](https://docs.opencv.org/master/d9/df8/group__tracking.html)

---

**Day 67 Complete** | Phase 4: ADAS & Robotics Systems | Week 10: Multi-Object Tracking
