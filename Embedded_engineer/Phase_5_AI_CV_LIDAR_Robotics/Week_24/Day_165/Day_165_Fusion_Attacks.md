# Day 165: Sensor Fusion Attacks (Falsified Lidar)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 24: Cybersecurity & Robustness

---

> **📝 Content Creator Instructions:**
> Three sensors walk into a bar. One is lying.
> - **Focus:** Lidar Spoofing (Replaying laser pulses), Camera Dazzling (Lasers), and Robust Fusion Architectures (Voting Logic, Bayesian Conflict).
> - **Code:** A Python script `fusion_voting.py` simulating 3 sensors (Camera, Lidar, Radar). A "Ghost Object" is injected into the Lidar stream. The fusion engine uses a "2-of-3" Voting Logic to reject the anomaly.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Understand** how to spoof a Lidar (Pulse delay injection).
2.  **Implement** a Voting Gate: Reject detections that are not confirmed by at least N sensors.
3.  **Calculate** Object Existence Probability $P(E|z_{cam}, z_{lid}, z_{rad})$.
4.  **Debate** the "Fail-Safe" vs "Fail-Operational" response to sensor disagreement.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Sensor Fusion (Kalman/Bayes).
- Logic Gates.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Lidar Hacking

Lidar measures Time of Flight (ToF).
*   **Attack:** Attacker fires a laser pulse *at* the Lidar receiver just before the real reflection would arrive.
*   **Result:** Lidar calculates shorter time $\to$ Closer distance.
*   **Effect:** The AV thinks a wall suddenly appeared 5m ahead and slams the brakes (Phantom Braking).

### 🔹 Part 2: Voting Logic (m-of-n)

Safety Critical Systems (like Planes) uses Triple Modular Redundancy (TMR).
*   **1-of-3 (Union):** If ANY sensor sees it, stop. (Safe but lots of False Alarms).
*   **2-of-3 (Majority):** Need 2 sensors to agree. (Robust to single sensor failure).
*   **3-of-3 (Intersection):** Need all 3. (Low False Alarms, but dangerous if one sensor is blind).

### 🔹 Part 3: Bayesian Existence

Instead of Binary Voting, keep a Probability.
$$ P(E|Z) = \frac{P(Z|E)P(E)}{P(Z)} $$
*   Start $P(E) = 0.5$.
*   Lidar detection: $P(E) \to 0.9$.
*   Camera miss: $P(E) \to 0.4$.
*   Radar miss: $P(E) \to 0.1$.
*   **Decision:** $P(E) < 0.5 \implies$ Ignore.

---

## 💻 Implementation: The Voting Machine

We simulate an AV approaching a real obstacle and a ghost obstacle.

### 🛠️ Project Structure
```text
day165_fusion_attack/
├── src/
│   ├── fusion_voting.py
└── output/
    ├── voting_log.txt
```

### 👨‍💻 Fusion Logic (`src/fusion_voting.py`)

```python
import numpy as np

class Sensor:
    def __init__(self, name, reliability):
        self.name = name
        self.reliability = reliability # Probability of correct detection
        self.false_positive_rate = 0.05
        
    def detect(self, real_dist, ghost_dist=None):
        # Return detected distance or None
        
        # Real Object
        if np.random.random() < self.reliability:
            meas_real = real_dist + np.random.normal(0, 0.5)
        else:
            meas_real = None # Miss
            
        # Ghost Object (Attack) - Only Lidar sees this if injected
        meas_ghost = None
        if ghost_dist is not None:
            # Attack vector active for this sensor?
            if self.name == "Lidar": 
                 meas_ghost = ghost_dist # Perfect spoof
                 
        return meas_real, meas_ghost

class FusionEngine:
    def __init__(self):
        self.tracks = {} # ID -> [P_existence, List of measurements]
        
    def fuse(self, detections):
        # detections: list of {'sensor': name, 'dist': d, 'type': 'real/ghost'}
        
        # Simplified: We just cluster by distance
        # Bin 1: Real Object (around 20m)
        # Bin 2: Ghost (around 10m)
        
        clusters = {'A': [], 'B': []}
        
        for d in detections:
            if d['dist'] is None: continue
            
            if abs(d['dist'] - 20.0) < 2.0:
                clusters['A'].append(d['sensor'])
            elif abs(d['dist'] - 10.0) < 2.0:
                clusters['B'].append(d['sensor'])
                
        self.evaluate_cluster(clusters['A'], "Object A (20m)")
        self.evaluate_cluster(clusters['B'], "Object B (Ghost 10m)")
        
    def evaluate_cluster(self, sensors, name):
        if len(sensors) == 0: return
        
        # Voting Logic: 2 of 3
        count = len(sensors)
        
        status = "REJECTED"
        if count >= 2:
            status = "CONFIRMED"
        elif count == 1:
            # Special Case: If it's Lidar, maybe? No, strict voting.
            pass
            
        print(f"   Scan {name}: Seen by {sensors} -> {status} ({count}/3)")

def main():
    # Sensors
    cam = Sensor("Camera", 0.8)   # Good, sometimes misses due to light
    lid = Sensor("Lidar", 0.99)   # Excellent... but vulnerable
    rad = Sensor("Radar", 0.9)    # Reliable
    
    fusion = FusionEngine()
    
    print("--- Sensor Fusion Security Test ---")
    
    # Scene: Real Obstacle at 20m.
    # Attack: Lidar Spoofing creates Ghost at 10m.
    
    for t in range(5):
        print(f"\nFrame {t}:")
        
        # 1. Generate Raw Data
        c_r, c_g = cam.detect(20.0, 10.0) 
        l_r, l_g = lid.detect(20.0, 10.0) # Lidar sees Ghost!
        r_r, r_g = rad.detect(20.0, 10.0) 
        
        # Camera/Radar are NOT spoofed physically (harder to do simultaneously)
        # So they return None for ghost
        c_g = None
        r_g = None
        
        detections = []
        det_list = [(cam, c_r, c_g), (lid, l_r, l_g), (rad, r_r, r_g)]
        
        for sens, real, ghost in det_list:
            if real: detections.append({'sensor': sens.name, 'dist': real})
            if ghost: detections.append({'sensor': sens.name, 'dist': ghost})
            
        # 2. Fuse
        fusion.fuse(detections)
        
    print("\nConclusion: The Ghost at 10m was consistently REJECTED because only Lidar saw it.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Fog"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** Voting works. Ghost rejected.
- **Modify:** Reduce Camera/Radar reliability to 0.1 (Heavy Fog).
- **Result:** Now Camera and Radar miss the *Real Object* at 20m often.
- **Vote:** Real Object (seen only by Lidar) is *also* REJECTED.
- **Outcome:** Crash into Real Object.
- **Lesson:** Voting assumes independent failures. In adverse weather, correlated failures (Cam+Lidar fail in rain) break the logic. We need "Sensor Confidence" models (Day 163).

---

## 🚀 Project: "Lidar Authenticity Check"

**Goal:** Detect Spoofing locally.
1.  **Idea:** Lidar pulses are encrypted? No, physics.
2.  **Idea:** Randomize Pulse Interval (jitter).
    *   Fire at $t, t+1.1, t+2.3 \dots$
    *   If return comes at $t+1.0$ (Attacker guessing 1.0 spacing), reject it.
3.  **Task:** Implement Jittered Timing logic in Python.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "False Rejection"
*   **Cause:** Radar sees under the truck. Camera sees the truck. Lidar sees the truck. Radar vote missing.
*   **Fix:** Class-specific voting. "If Camera says Truck, Radar might miss it. Trust Camera."

#### 2. "Latency Mismatch"
*   **Cause:** Lidar 10Hz, Camera 30Hz. Voting fails because data isn't synced.
*   **Fix:** Time buffers. Hold Camera data until Lidar arrives.

---

## ⚡ Optimization: Bayesian Occupancy Filter

Grid-based fusion.
*   **Grid:** Each cell has $P(Occupied)$.
*   **Update:**
    $$ P(O|Z) = \text{BayesUpdate}(P(O), P(Z|O)) $$
*   **Robustness:** A single Lidar frame increases $P$ slightly. Need consistent frames to reach "Confirmed" threshold. Spoofers are usually glitchy/transient.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Fail-Safe vs Fail-Operational?
    *   **A:** Fail-Safe: System shuts down safely (Pull over). Fail-Operational: System continues to operate (Degraded mode). For L4, must be Fail-Operational (can't stop on highway).
2.  **Q:** What is a "Ghost Object"?
    *   **A:** A detection that corresponds to no physical object. Caused by reflection, noise, or hacking.
3.  **Q:** Why is Camera harder to spoof?
    *   **A:** You need a projector or a screen. The pixel density is high. Lidar is just a timing measurement (1D).

### Challenge Task
> **Task:** GPS-Lidar Cross Check.
> 1. GPS says speed 100km/h.
> 2. Lidar features (SLAM) say speed 0km/h (hacking: feeding static point cloud).
> 3. Trigger "Sensor Mismatch Alarm".

---

## 📚 Further Reading
- **Shin et al.:** "Illusion and Dazzle: Adversarial Optical Attacks on Lidar Systems".
- **Koopman:** "Safety of the Intended Functionality (SOTIF)".

---

**Day 165 Complete**
