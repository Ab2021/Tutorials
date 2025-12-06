# Day 156: Cooperative Perception (Sharing Lidar)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 23: V2X & Swarm Intelligence

---

> **📝 Content Creator Instructions:**
> My eyes are your eyes.
> - **Focus:** Limits of Raw Lidar sharing (Bandwidth), CPM (Collective Perception Message), ETSI standards, Object-Level Fusion vs Raw-Data Level Fusion.
> - **Code:** A Python script `coop_fusion.py` where Car A detects an object (invisible to Car B) and sends the coordinate list to Car B. Car B transforms these points from "Frame A" to "Frame B" and visualizes the "Ghost Object".

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** why we cannot send raw Lidar point clouds over V2V (100 Mbps vs 6 Mbps).
2.  **Construct** a CPM (Collective Perception Message) containing a list of detected objects.
3.  **Perform** Coordinate Transformation between two dynamic vehicles using GPS/Compass data.
4.  **Implement** Object Fusion: Merging Local and Remote detections of the same object.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Homogeneous Transforms.
- GPS Coordinates (Lat/Lon to XY).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Bandwidth Bottleneck

*   **Raw Lidar:** 100k points/frame $\times$ 10Hz $\times$ 16 bytes = 16 MB/s (128 Mbps).
*   **DSRC Bandwidth:** ~6 Mbps effective.
*   **Conclusion:** Impossible to stream raw Lidar.
*   **Solution:** Compute Locally, Share Globally. Send **Object Lists** (Box center, dims, class). Bandwidth ~50 kbps.

### 🔹 Part 2: CPM (Collective Perception Message)

ETSI Standard.
Contains:
1.  **Origin:** Position of Sensing Station.
2.  **Sensor Info:** Sensor Range / FOV (Shadowing).
3.  **Objects:** List of Tracked Objects (ID, Pos, Vel, Conf).

### 🔹 Part 3: Map Merging

Car A sees Object at $(10, 0)$ relative to A.
Car B is at $(50, 0)$ relative to A, facing opposite.
How does B know where the object is?
$$ P_{obj\_in\_B} = T_{World \to B} \cdot T_{A \to World} \cdot P_{obj\_in\_A} $$
*   Requires high-precision GPS (RTK) on both cars.

---

## 💻 Implementation: Seeing Around Corners

Car A is at an intersection. It sees a Pedestrian.
Car B is approaching but blinded by a Building.
Car A sends the Pedestrian location to Car B.

### 🛠️ Project Structure
```text
day156_coop/
├── src/
│   ├── coop_fusion.py
└── output/
    ├── shared_view.png
```

### 👨‍💻 Cooperative Fusion (`src/coop_fusion.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class Vehicle:
    def __init__(self, name, x, y, yaw):
        self.name = name
        self.x = x
        self.y = y
        self.yaw = yaw # Radians
        self.local_objects = []
        self.remote_objects = [] # Ghosts
        
    def get_transform_matrix(self):
        # T_map_vehicle
        s = np.sin(self.yaw)
        c = np.cos(self.yaw)
        T = np.eye(3)
        T[0, 0] = c
        T[0, 1] = -s
        T[0, 2] = self.x
        T[1, 0] = s
        T[1, 1] = c
        T[1, 2] = self.y
        return T

    def detect(self, gt_objects):
        # Simulator: Check which objects are in FOV
        self.local_objects = []
        T_mv = self.get_transform_matrix()
        T_vm = np.linalg.inv(T_mv)
        
        for obj in gt_objects:
            # Global to Local
            p_global = np.array([obj['x'], obj['y'], 1.0])
            p_local = T_vm @ p_global
            
            # Simple FOV check (Circle radius 30m, +/- 60 deg)
            dist = np.linalg.norm(p_local[:2])
            angle = np.arctan2(p_local[1], p_local[0])
            
            if dist < 30.0 and abs(angle) < np.radians(60):
                # Check occlusion? (Simple: All visible)
                # Store as Local Coordinates
                self.local_objects.append({'id': obj['id'], 'x': p_local[0], 'y': p_local[1]})
                
    def receive_cpm(self, sender_vehicle, objects):
        # Transform Remote Objects (Sender Frame) to Local Frame (My Frame)
        # P_local = T_vm_me * T_mv_sender * P_remote
        
        T_mv_sender = sender_vehicle.get_transform_matrix()
        T_mv_me = self.get_transform_matrix()
        T_vm_me = np.linalg.inv(T_mv_me)
        
        T_sender_to_me = T_vm_me @ T_mv_sender
        
        self.remote_objects = []
        for obj in objects:
            p_remote = np.array([obj['x'], obj['y'], 1.0])
            p_local = T_sender_to_me @ p_remote
            
            self.remote_objects.append({'id': obj['id'], 'x': p_local[0], 'y': p_local[1]})

def main():
    # 1. Setup Scene
    # Car A: At (0,0), Facing East (0). Sees Pedestrian at (20, 5).
    car_a = Vehicle("Car A", 0, 0, 0)
    
    # Car B: At (20, -20), Facing North (90). Approaching intersection.
    # Blind to Pedestrian due to wall (not simulated physically, just logically)
    car_b = Vehicle("Car B", 20, -20, np.radians(90))
    
    # Ground Truth Objects
    gt_objects = [
        {'id': 1, 'x': 20, 'y': 5, 'label': 'Pedestrian'}
    ]
    
    # 2. Sensing
    # A sees it
    car_a.detect(gt_objects)
    print(f"Car A detects: {len(car_a.local_objects)} objects.")
    
    # B does NOT see it (Simulate building occlusion)
    car_b.detect([]) 
    print(f"Car B detects: {len(car_b.local_objects)} objects.")
    
    # 3. Communications (A sends CPM to B)
    print("--- Transmitting CPM (A -> B) ---")
    car_b.receive_cpm(car_a, car_a.local_objects)
    
    # 4. Visualization from Car B's perspective
    print(f"Car B now knows about {len(car_b.remote_objects)} remote objects.")
    
    plt.figure(figsize=(8, 8))
    
    # Plot Car B (Origin of plot)
    plt.plot(0, 0, 'bs', markersize=15, label='Me (Car B)')
    plt.arrow(0, 0, 0, 5, head_width=2, color='blue')
    
    # Plot Remote Objects
    for obj in car_b.remote_objects:
        plt.plot(obj['x'], obj['y'], 'ro', markersize=10, label=f"Ghost {obj['id']}")
        
        # Verify Coordinate Logic
        # Pedestrian Global (20, 5). 
        # Car B Global (20, -20), Heading 90 (N).
        # Pedestrian relative to B:
        # B's X-axis is Global Y. B's Y-axis is Global -X.
        # Wait, Heading 90 (North).
        # B Forward is Global +Y.
        # B Left is Global -X.
        # Pedestrian is at Global Y=5. B is at Y=-20. Diff = +25 (Forward).
        # Pedestrian is at Global X=20. B is at X=20. Diff = 0 (Lateral).
        # So in B Frame: X=25 (Forward), Y=0.
        pass
        
    plt.title("Car B's World Model (Fused)")
    plt.xlabel("Local X (Forward) [m]")
    plt.ylabel("Local Y (Left) [m]")
    plt.xlim(-10, 50)
    plt.ylim(-30, 30)
    plt.grid()
    plt.legend()
    plt.savefig("output/shared_view.png")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Ghost"

### 1. Lab Objectives
- **Run:** Sim.
- **Coords:** Check the plot. The "Ghost" should be at X=25 (Forward) for Car B.
- **Error:** Introduce GPS Error. Add `car_a.x += 2.0` noise.
- **Fail:** Car B sees the ghost shift. If the error is large (>5m), Car B might brake for a phantom pedestrian on the sidewalk, thinking it's on the road.
- **Fix:** Covariance. BSM/CPM includes "Position Confidence". If Confidence is low, inflate the Ghost Box size (Uncertainty area).

---

## 🚀 Project: "Distributed Data Association"

**Goal:** Deduplicate.
1.  **Duplicate:** Car A sees P1. Car B sees P1.
2.  **Share:** A tells B about P1.
3.  **Result:** Car B sees "Local P1" and "Remote P1".
4.  **Task:** Implement IoU check. If Local Box overlaps Remote Box, merge them. Use a Weighted Average (weighted by distance/sensor uncertainty).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Latency Jitter"
*   **Cause:** Object moving fast. Packet takes 100ms.
*   **Result:** Remote object lags behind detection.
*   **Fix:** Extrapolate. $P_{now} = P_{msg} + V_{msg} \times (t_{now} - t_{msg})$.

#### 2. "Circular Loops"
*   **Cause:** A sends to B. B sends to C. C sends back to A.
*   **Result:** A thinks there are 2 objects.
*   **Fix:** Unique Object IDs (UUID) maintained across the swarm? Hard. Or TTL (Time To Live) on hops.

---

## ⚡ Optimization: Value of Information

Don't send everything.
*   **Policy:** Only send objects that the receiver *cannot* see.
*   **How?** A knows B's position. A calculates B's FOV. If Object is in B's blind spot, Send it. Bandwidth saving.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** CPM vs DENM?
    *   **A:** CPM (Collective Perception) is continuous stream of objects. DENM (Decentralized Environmental Notification) is Event-based (Ice on road, Accident).
2.  **Q:** Why not just share video?
    *   **A:** 4K video is 25 Mbps compressed. V2X is 6 Mbps. Also privacy (Faces).
3.  **Q:** What is "Sensor Shadowing"?
    *   **A:** When a large vehicle (Truck) blocks the view. Cooperative perception allows the Truck to tell the car behind what's ahead.

### Challenge Task
> **Task:** Overtaking Assist.
> 1. Car A wants to overtake Truck.
> 2. Truck sends list of objects in front.
> 3. If "Oncoming Car" detected by Truck, Car A aborts overtake.

---

## 📚 Further Reading
- **ETSI TR 103 562:** Collective Perception Service analysis.
- **Autoware:** V2X modules.

---

**Day 156 Complete**
