# Day 69: Track Management and Lifecycle
## Phase 4: ADAS & Robotics Systems | Week 10: Multi-Object Tracking

---

> **📝 Day 69 Focus:**
> Algorithms like Kalman Filter and Hungarian are the "Engine" of tracking. **Track Management** is the "Steering Wheel". It decides who gets to be a track, who gets deleted, and who is just noise. Without good management, your tracker will be flooded with ghost objects.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the Track Lifecycle: New -> Tentative -> Confirmed -> Coasting -> Deleted.
2.  **Implement** "M out of N" logic for track confirmation (e.g., 3 hits in 5 frames).
3.  **Handle** Track Termination using Time-to-Live (TTL) and Missed Counts.
4.  **Design** a Track Score metric to rank track quality.
5.  **Build** a robust `TrackManager` class that wraps the core tracking algorithms.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 64-68:** Tracking Algorithms.
-   **State Machines:** Lifecycle logic.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Lifecycle State Machine

1.  **New:** A detection that didn't match any existing track.
2.  **Tentative:** We are watching it. Is it noise or real?
    -   *Transition:* If `hits >= min_hits`, go to **Confirmed**.
    -   *Transition:* If `missed >= max_missed_tentative`, go to **Deleted**.
3.  **Confirmed:** It's a real object. We publish it to the ADAS system.
    -   *Transition:* If `missed > 0`, go to **Coasting**.
4.  **Coasting (Predicted):** We lost detection, but we predict where it is.
    -   *Transition:* If `matched`, go back to **Confirmed**.
    -   *Transition:* If `missed >= max_age`, go to **Deleted**.
5.  **Deleted:** Garbage collection.

### 🔹 Part 2: Confirmation Logic (M/N)

We don't trust a single detection (could be a reflection, a leaf, or sensor noise).
**Rule:** A track is confirmed only if it is detected in $M$ out of the last $N$ frames.
-   Common: 3 out of 5.
-   Reduces False Positives (Ghosts).

### 🔹 Part 3: Track Scoring

Not all tracks are equal.
$$ \text{Score} = w_1 \times \text{DetScore} + w_2 \times \text{TrackLength} + w_3 \times \text{Consistency} $$
-   Used for downstream decision making (e.g., AEB should only brake for High Score tracks).

---

## 💻 Implementation: Robust Track Manager

**Scenario:**
-   **Input:** Detections (some are clutter).
-   **Output:** Only "Confirmed" tracks.

### 🛠️ Setup
Create `week10_day69` and `track_manager.py`.

```bash
mkdir -p ~/ros2_ws/src/week10_day69
cd ~/ros2_ws/src/week10_day69
touch track_manager.py
```

### 👨‍💻 Code: Track Manager

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
MIN_HITS = 3
MAX_AGE = 5
MAX_AGE_TENTATIVE = 2

class TrackState:
    NEW = 0
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3

class Track:
    id_counter = 0
    def __init__(self, detection):
        self.id = Track.id_counter
        Track.id_counter += 1
        
        self.pos = detection # Simple 1D state for demo
        self.state = TrackState.TENTATIVE
        
        self.hits = 1
        self.missed = 0
        self.age = 1
        self.history = [self.pos]
        
    def predict(self):
        # Constant Velocity (Simplified)
        if len(self.history) > 1:
            vel = self.history[-1] - self.history[-2]
        else:
            vel = 0
        self.pos += vel
        self.age += 1
        
    def update(self, detection):
        self.pos = detection
        self.hits += 1
        self.missed = 0
        self.history.append(self.pos)
        
        # Lifecycle Logic
        if self.state == TrackState.TENTATIVE:
            if self.hits >= MIN_HITS:
                self.state = TrackState.CONFIRMED
                
    def mark_missed(self):
        self.missed += 1
        
        # Lifecycle Logic
        if self.state == TrackState.TENTATIVE:
            if self.missed >= MAX_AGE_TENTATIVE:
                self.state = TrackState.DELETED
        elif self.state == TrackState.CONFIRMED:
            if self.missed >= MAX_AGE:
                self.state = TrackState.DELETED

class TrackManager:
    def __init__(self):
        self.tracks = []
        
    def update(self, detections):
        # 1. Predict
        for t in self.tracks:
            t.predict()
            
        # 2. Associate (Simple Distance for Demo)
        # In real system: Use Hungarian + IoU/Mahalanobis
        assigned_tracks = set()
        assigned_dets = set()
        
        for i, det in enumerate(detections):
            best_dist = 10.0 # Gating
            best_idx = -1
            
            for j, trk in enumerate(self.tracks):
                if j in assigned_tracks: continue
                dist = abs(det - trk.pos)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j
                    
            if best_idx != -1:
                self.tracks[best_idx].update(det)
                assigned_tracks.add(best_idx)
                assigned_dets.add(i)
                
        # 3. Manage Missed
        for i, trk in enumerate(self.tracks):
            if i not in assigned_tracks:
                trk.mark_missed()
                
        # 4. Create New
        for i, det in enumerate(detections):
            if i not in assigned_dets:
                self.tracks.append(Track(det))
                
        # 5. Delete Dead
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]
        
        # Return only CONFIRMED tracks for downstream
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]

def main():
    manager = TrackManager()
    
    # Simulation
    # True Object: 0 -> 100
    # Clutter: Random noise appearing for 1 frame
    
    history_confirmed = []
    history_all = []
    
    for i in range(20):
        dets = []
        
        # True Object (Visible frames 0-10, 15-20)
        if i <= 10 or i >= 15:
            dets.append(i * 5.0)
            
        # Clutter (Randomly appears)
        if np.random.rand() < 0.3:
            dets.append(np.random.uniform(0, 100))
            
        confirmed_tracks = manager.update(dets)
        
        # Log
        history_confirmed.append(len(confirmed_tracks))
        history_all.append(len(manager.tracks))
        
        print(f"Frame {i}: Dets={dets} | All Tracks={len(manager.tracks)} | Confirmed={len(confirmed_tracks)}")
        
    # Plot
    plt.plot(history_all, label="All Tracks (Internal)")
    plt.plot(history_confirmed, label="Confirmed Tracks (Output)")
    plt.legend()
    plt.title("Track Lifecycle Management")
    plt.xlabel("Frame")
    plt.ylabel("Count")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Ghost

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   Frame 0-2: "Confirmed" count is 0. "All" count is 1. (Track is Tentative).
    -   Frame 3: "Confirmed" becomes 1. (Hits >= 3).
    -   Clutter detections create "All" tracks, but they die quickly and never become "Confirmed".
    -   Frame 11-14: Object is occluded. "Confirmed" stays 1 (Coasting).
    -   Frame 15: Object reappears. Track resumes.
3.  **Experiment:**
    -   Set `MIN_HITS = 1`.
    -   **Result:** Every clutter point immediately becomes a Confirmed Track. The output is noisy.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Latency
**Symptom:** System reacts too slowly to new objects.
**Cause:** `MIN_HITS` is too high. Waiting 5 frames at 10Hz = 0.5s delay.
**Solution:** Use lower `MIN_HITS` for critical zones (e.g., close to ego vehicle) or high-confidence detections.

#### 2. Zombie Tracks
**Symptom:** Tracks persist long after the car has left.
**Cause:** `MAX_AGE` is too high.
**Solution:** Reduce `MAX_AGE`. Or use "Motion Consistency" check (if variance grows too large, kill it).

---

## ⚡ Optimization & Best Practices

### 1. ID Recycling
If you run for hours, `id_counter` will overflow (int32).
-   Use a **Free List** of IDs. When a track dies, put its ID back in the pool.
-   Or use `int64` / UUIDs.

### 2. Scene Management
-   **Enter/Exit Zones:** If a track exits the drivable area (e.g., goes off-road), delete it immediately. Don't wait for timeout.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why do we have a "Tentative" state?
    *   **A:** To filter out transient noise (clutter) that only appears for 1-2 frames.
2.  **Q:** What is "Coasting"?
    *   **A:** Predicting the track state without a measurement update. Essential for handling occlusions.
3.  **Q:** How does Track Management affect Safety?
    *   **A:** False Positives (Phantom Braking) are reduced by Confirmation Logic. False Negatives (Missed Detection) are reduced by Coasting.

### Challenge Task
**Task:** Confidence Decay.
1.  Add a `confidence` score (0.0 to 1.0) to the Track.
2.  Increase confidence on `update` (+0.1), decrease on `missed` (-0.2).
3.  Delete if `confidence < 0.0`.
4.  Confirm if `confidence > 0.8`.

---

## 📚 Further Reading & References
-   [Design of Multi-Target Tracking Systems (Bar-Shalom)](https://www.amazon.com/Design-Multi-Target-Tracking-Systems-Bar-Shalom/dp/0890067570)
-   [ROS 2 depth_image_proc](http://wiki.ros.org/depth_image_proc)

---

**Day 69 Complete** | Phase 4: ADAS & Robotics Systems | Week 10: Multi-Object Tracking
