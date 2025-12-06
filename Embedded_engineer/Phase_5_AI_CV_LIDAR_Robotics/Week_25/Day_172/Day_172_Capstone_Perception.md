# Day 172: Capstone: Perception & Logic
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 25: Final Integration & Graduation

---

> **📝 Content Creator Instructions:**
> Seeing is believing.
> - **Focus:** Integrating the Perception Stack. Combining Lane Lines (Polynomials), Object Detection (YOLO-ish), and State Estimation (Kalman).
> - **Code:** `capstone_perception.py`. A multi-threaded perception engine. Thread 1: Lane Processing. Thread 2: Object Detection. Main Thread: Fuses them into a unified `WorldModel` (e.g., "Car A is in Left Lane").

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Integrate** distinct vision modules into a thread-safe Perception Engine.
2.  **Associate** Objects with Lanes ("Is the truck in my lane or the neighbor lane?").
3.  **Produce** a Semantic World Model for the Planner.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU recommended for real YOLO (Simulated here).

### Software Environment
```bash
pip install numpy opencv-python
```

### Prior Knowledge
- Lane Detection (Week 22).
- Object Tracking (Week 22).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Unified World Model

The Planner doesn't want "Pixels". It wants "Semantics".
*   **Input:** Raw Image.
*   **Perception:**
    *   Lanes: $y = ax^2 + bx + c$.
    *   Objects: Box $(x, y, w, h)$.
*   **Fusion (Association):**
    *   Calculate Object Center $C_obj$.
    *   Calculate Lane Lateral Position at object depth $X_{lane}(dist)$.
    *   If $X_{left\_lane} < C_obj < X_{right\_lane}$, Object is in **Ego Lane**.

### 🔹 Part 2: Threading & Latency

Lanes ($50Hz$) vs YOLO ($10Hz$).
*   **Architecture:** Async Threads.
*   **Latest Data Policy:** The Fusion loop runs at $20Hz$, grabbing the *latest available* lane info and object list.
*   **Locking:** Use `threading.Lock()` when updating the shared World State to prevent race conditions (torn reads).

---

## 💻 Implementation: The Eye of the Robot

We simulate the inputs (Video Frame) and run the logic.

### 🛠️ Project Structure
```text
day172_perception/
├── src/
│   ├── capstone_perception.py
│   └── assets/
│       └── road_video_mock.mp4 (Simulated)
└── output/
    ├── world_state_log.json
```

### 👨‍💻 Perception Engine (`src/capstone_perception.py`)

```python
import threading
import time
import numpy as np
import json
import random

# --- DATA STRUCTURES ---
class DetectedObject:
    def __init__(self, id, x, y, width, class_name):
        self.id = id
        self.rel_pos = np.array([x, y]) # Relative to Ego (x=lat, y=long)
        self.width = width
        self.class_name = class_name
        self.lane_assignment = "UNKNOWN"

class LanePoly:
    def __init__(self, coeffs):
        self.coeffs = coeffs # [a, b, c] for x = ay^2 + by + c (Vertical lanes)
    
    def eval_at(self, y):
        # x = ay^2 + by + c
        return self.coeffs[0]*y**2 + self.coeffs[1]*y + self.coeffs[2]

class WorldModel:
    def __init__(self):
        self.lock = threading.Lock()
        self.ego_lane_left = None
        self.ego_lane_right = None
        self.objects = []
        self.timestamp = 0.0

    def update(self, left, right, objs, ts):
        with self.lock:
            self.ego_lane_left = left
            self.ego_lane_right = right
            self.objects = objs # List of DetectedObject
            self.timestamp = ts
            
    def snapshot(self):
        with self.lock:
            # Return a deep copy or safe struct
            return {
                'ts': self.timestamp,
                'left_poly': self.ego_lane_left.coeffs if self.ego_lane_left else None,
                'objects': [(o.id, o.class_name, o.lane_assignment) for o in self.objects]
            }

# --- MODULES ---

class LaneDetector(threading.Thread):
    def __init__(self, world_model):
        super().__init__()
        self.wm = world_model
        self.running = True
        
    def run(self):
        print("[LANE] Started.")
        while self.running:
            # 1. Simulate finding lines
            # y = dist ahead. x = lateral (0 center).
            # Left lane roughly x = -2. Right roughly x = +2.
            # Curving slightly left: a = -0.001
            left = LanePoly([-0.0001, 0.0, -2.0])
            right = LanePoly([-0.0001, 0.0, 2.0])
            
            # (In real life, we'd process self.wm.latest_image)
            
            time.sleep(0.03) # 33ms (30 FPS)
            
            # Post to shared state (Partial update? No, usually main loop fuses)
            # Let's say this thread creates the 'Raw Lane' data.
            # For simplicity, we'll let the Main Loop do the Fusion/Arg Update
            self.latest_lanes = (left, right)

class ObjectDetector(threading.Thread):
    def __init__(self, world_model):
        super().__init__()
        self.wm = world_model
        self.running = True
        self.latest_objects = []
        
    def run(self):
        print("[YOLO] Started.")
        id_counter = 0
        while self.running:
            # Simulate YOLO inference time
            time.sleep(0.1) # 10 FPS
            
            # Generate fake cars
            objs = []
            # Car A: 20m ahead, 0m lateral (In Ego Lane)
            objs.append(DetectedObject(1, 0.0, 20.0, 1.8, "Car"))
            
            # Car B: 10m ahead, -4m lateral (Left Lane)
            objs.append(DetectedObject(2, -4.0, 10.0, 1.8, "Truck"))
            
            self.latest_objects = objs

class PerceptionCore:
    def __init__(self):
        self.world = WorldModel()
        self.lane_thread = LaneDetector(self.world)
        self.obj_thread = ObjectDetector(self.world)
        
    def start(self):
        self.lane_thread.start()
        self.obj_thread.start()
        
    def stop(self):
        self.lane_thread.running = False
        self.obj_thread.running = False
        self.lane_thread.join()
        self.obj_thread.join()
        
    def fuse_step(self):
        # 1. Get Latest Raw Data
        lanes = getattr(self.lane_thread, 'latest_lanes', (None, None))
        objs = getattr(self.obj_thread, 'latest_objects', [])
        
        left_poly, right_poly = lanes
        
        # 2. Association Logic
        for o in objs:
            if left_poly and right_poly:
                # Calculate lane bounds at object's longitudinal distance (y)
                # Note: Our poly is x = f(y)
                l_x = left_poly.eval_at(o.rel_pos[1])
                r_x = right_poly.eval_at(o.rel_pos[1])
                
                # Check Center
                c_x = o.rel_pos[0]
                
                if l_x < c_x < r_x:
                    o.lane_assignment = "EGO_LANE"
                elif c_x < l_x:
                     o.lane_assignment = "LEFT_LANE"
                elif c_x > r_x:
                     o.lane_assignment = "RIGHT_LANE"
            else:
                o.lane_assignment = "UNKNOWN_NO_LANES"
                
        # 3. Update World Model
        self.world.update(left_poly, right_poly, objs, time.time())

def main():
    core = PerceptionCore()
    core.start()
    
    print("running perception stack for 5 seconds...")
    log = []
    
    for _ in range(50): # 5 seconds at 10Hz
        core.fuse_step()
        
        snap = core.world.snapshot()
        print(f"[{snap['ts']:.2f}] Objects: {snap['objects']}")
        log.append(snap)
        
        time.sleep(0.1)
        
    core.stop()
    print("Done.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Cut-In"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** Car 1 is EGO_LANE. Truck 2 is LEFT_LANE.
- **Modify:** Change Car B coords. Assume it merges.
- **Loop:** `x` goes from -4.0 to 0.0 over 5 seconds.
- **Observe:** Lane assignment switches LEFT -> EGO.
- **Planner Reaction:** Once it becomes EGO, the Planner must react (ACC Slow Down). Before that, it ignores it.
- **Hysteresis:** Add a buffer. Don't flip assignment if it just touches line. Require 50% overlap.

---

## 🚀 Project: "Traffic Light State"

**Goal:** Traffic Light Classification.
1.  **Input:** Bounding Box of Traffic Light.
2.  **Crop:** Extract ROI.
3.  **Color:** Auto-White Balance + HSV Thresholding (Red/Yellow/Green).
4.  **Assocation:** Which lane does this light control? (Hard! Need Maps).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Coordinate Hell"
*   **Cause:** Box is in Image Coords (pixels). Lane is in BEV (meters).
*   **Fix:** Inverse Perspective Mapping (IPM) or Camera Matrix PnP to convert everything to "Vehicle Frame (Meters)" before Fusion.

#### 2. "Flickering ID"
*   **Cause:** YOLO misses frames. Tracker resets ID.
*   **Fix:** SORT/DeepSORT (Day 151). Maintain ID across missed frames (Kalman Prediction).

---

## ⚡ Optimization: ROI Processing

Don't run Lane detection on the sky.
*   **ROI:** Crop bottom half.
*   **Tiling:** For 4K cameras, split into tiles. Run YOLO only on tiles near horizon if looking for far cars.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why run Lanes and Objects in different threads?
    *   **A:** Lane algo (Polyfit) is CPU bound. Object (CNN) is GPU bound. Parallelize to maximize throughput.
2.  **Q:** What is "Lane Association"?
    *   **A:** Determining which lane an object occupies. Critical for ACC/AEB decisions.
3.  **Q:** Why use locks?
    *   **A:** To ensure the Planner gets a consistent snapshot (Lane info matches Object info timestamp).

### Challenge Task
> **Task:** Static Obstacles using Lanes.
> 1. Detect Lane boundaries.
> 2. If valid Lane Line ends abruptly ($y < 50m$) and resumes later ($y > 60m$), implies occlusion or road work.
> 3. Mark area as "Suspicious".

---

## 📚 Further Reading
- **NVIDIA DriveWorks:** Perception Pipeline architecture.
- **Apollo Auto:** Perception modules docs.

---

**Day 172 Complete**
