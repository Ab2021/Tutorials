# Day 88: Cooperative Perception (CPM)
## Phase 4: ADAS & Robotics Systems | Week 13: V2X Communication

---

> **📝 Day 88 Focus:**
> CAM/BSM tells you about the *sender*. **CPM (Cooperative Perception Message)** tells you about *what the sender sees*. If Car A sees a pedestrian that Car B cannot see (due to a truck), Car A shares the pedestrian's location with Car B. This is "Collective Intelligence".

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the concept of Cooperative Perception (Sensor Sharing).
2.  **Analyze** the CPM structure (Sensor Info, Object List, Free Space).
3.  **Implement** an Object List Fusion algorithm (Local + Remote objects).
4.  **Solve** the Coordinate Transformation problem (Sender Frame -> Receiver Frame).
5.  **Simulate** an occlusion scenario where CPM prevents a collision.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 66:** Object Tracking (SORT).
-   **Day 85:** V2X Basics.
-   **Geometry:** Homogeneous Transformations.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Occlusion Problem

Sensors (LiDAR/Camera) are Line-of-Sight.
-   **Scenario:** A pedestrian steps out from behind a parked truck.
-   **Ego Vehicle:** Cannot see the pedestrian.
-   **V2X Solution:** Another car (or Roadside Unit) has a clear view. It detects the pedestrian and broadcasts a CPM.

### 🔹 Part 2: CPM Structure (ETSI TS 103 324)

-   **Management Container:** Position of the Sender (Reference Point).
-   **Sensor Information:** "I have a LiDAR with 100m range."
-   **Perception Data:** List of Objects.
    -   `ObjectID`: ID of the object.
    -   `Distance`: X, Y relative to Sender.
    -   `Speed`: VX, VY relative to Sender.
    -   `Class`: Pedestrian, Vehicle, Animal.
    -   `Confidence`: 0-100%.

### 🔹 Part 3: Fusion Challenges

1.  **Latency:** By the time you receive the CPM, the object has moved. (Solution: Prediction).
2.  **Coordinate Alignment:** Sender says "Object is at (10, 5)". Receiver needs to know "Where is (10, 5) relative to ME?".
3.  **Data Association:** Is the object in the CPM the same as the one I see with my own radar? (Solution: Global ID or IoU matching).

---

## 💻 Implementation: CPM Fusion

**Scenario:**
-   **Car A (Sender):** At (0, 0). Sees Pedestrian at (10, 10). Broadcasts CPM.
-   **Car B (Receiver):** At (20, 0). Cannot see Pedestrian. Receives CPM.
-   **Task:** Car B calculates Pedestrian position relative to itself.

### 🛠️ Setup
Create `week13_day88` and `cpm_fusion.py`.

```bash
mkdir -p ~/ros2_ws/src/week13_day88
cd ~/ros2_ws/src/week13_day88
touch cpm_fusion.py
```

### 👨‍💻 Code: CPM Fusion Logic

```python
import numpy as np
import matplotlib.pyplot as plt

class Object:
    def __init__(self, id, x, y, type):
        self.id = id
        self.x = x
        self.y = y
        self.type = type

class CPM:
    def __init__(self, sender_id, sender_pos, sender_heading, objects):
        self.sender_id = sender_id
        self.sender_pos = np.array(sender_pos) # [x, y]
        self.sender_heading = sender_heading # Degrees
        self.objects = objects # List of Object (Relative to Sender)

class EgoVehicle:
    def __init__(self, x, y, heading):
        self.pos = np.array([x, y])
        self.heading = heading
        self.local_objects = []
        self.remote_objects = []
        
    def receive_cpm(self, cpm):
        # Transform CPM objects from Sender Frame to Global Frame, then to Ego Frame
        # For simplicity, we'll just go to Global Frame for visualization
        
        # 1. Rotation Matrix (Sender -> Global)
        theta = np.radians(cpm.sender_heading)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        
        for obj in cpm.objects:
            # P_global = P_sender + R * P_relative
            p_rel = np.array([obj.x, obj.y])
            p_global = cpm.sender_pos + R @ p_rel
            
            # Store as remote object (Global Coords for this demo)
            self.remote_objects.append({
                'id': f"{cpm.sender_id}_{obj.id}",
                'pos': p_global,
                'type': obj.type
            })
            print(f"CPM Rx: Object {obj.type} at Global {p_global}")

def main():
    # 1. Setup Scenario
    # Car A (Sender) at Origin, facing East (0 deg)
    car_a_pos = [0, 0]
    car_a_hdg = 0.0
    
    # Car B (Receiver) at (20, -5), facing North (90 deg)
    car_b = EgoVehicle(20, -5, 90.0)
    
    # 2. Car A detects objects
    # Pedestrian at (10, 10) relative to A
    # Truck at (20, 0) relative to A
    objs_a = [
        Object(1, 10, 10, "PEDESTRIAN"),
        Object(2, 20, 0, "TRUCK")
    ]
    
    # 3. Generate CPM
    cpm = CPM("Car_A", car_a_pos, car_a_hdg, objs_a)
    
    # 4. Car B receives CPM
    car_b.receive_cpm(cpm)
    
    # 5. Visualization
    plt.figure(figsize=(8, 8))
    
    # Draw Car A
    plt.plot(car_a_pos[0], car_a_pos[1], 'bs', label='Car A (Sender)')
    plt.arrow(car_a_pos[0], car_a_pos[1], 5, 0, head_width=2, color='b')
    
    # Draw Car B
    plt.plot(car_b.pos[0], car_b.pos[1], 'rs', label='Car B (Receiver)')
    plt.arrow(car_b.pos[0], car_b.pos[1], 0, 5, head_width=2, color='r')
    
    # Draw Remote Objects
    for obj in car_b.remote_objects:
        pos = obj['pos']
        marker = 'k*' if obj['type'] == 'PEDESTRIAN' else 'kd'
        plt.plot(pos[0], pos[1], marker, markersize=10, label=f"Remote {obj['type']}")
        plt.text(pos[0]+1, pos[1], obj['type'])
        
        # Draw Line of Sight from A
        plt.plot([car_a_pos[0], pos[0]], [car_a_pos[1], pos[1]], 'b--', alpha=0.3)
        
    # Draw Occlusion (Wall) that blocks B from seeing Pedestrian
    plt.plot([15, 15], [-10, 20], 'k-', linewidth=5, label='Wall')
    
    plt.xlim(-10, 40)
    plt.ylim(-10, 40)
    plt.legend()
    plt.title("Cooperative Perception (CPM)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Coordinate Transformation

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   Car A is at (0,0). Pedestrian is at (10,10).
    -   Car B is at (20,-5).
    -   Car B correctly plots the Pedestrian at Global (10,10).
3.  **Experiment:**
    -   Change Car A heading to 45 degrees.
    -   Update `objs_a` relative positions to match the new heading (or keep them same to see rotation).
    -   **Result:** The Global position of the pedestrian rotates. This proves the importance of `sender_heading` in the CPM. If A's compass is wrong, B puts the ghost pedestrian in the wrong place!

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Circular Dependencies
**Symptom:** Ghost objects multiplying.
**Cause:** Car A sends CPM to B. B fuses it and sends CPM to A. A thinks it's a new object.
**Solution:** Track `SourceID`. Never re-broadcast an object back to its source.

#### 2. Bandwidth Saturation
**Symptom:** Network jam.
**Cause:** Sending raw object lists (100 objects) at 10Hz.
**Solution:**
    -   **Redundancy Check:** Don't send objects that are already seen by everyone (e.g., if another car already broadcasted it).
    -   **Delta Updates:** Only send moving objects.

---

## ⚡ Optimization & Best Practices

### 1. Object Inclusion Rules
Don't send everything.
-   **Rule:** Only send objects that are *dynamic* and *relevant* (e.g., on the road).
-   Static objects (trees) are in the HD Map. Don't waste bandwidth on them.

### 2. Confidence Score
Always include confidence.
-   If Car A is 90% sure, and Car B is 20% sure (glitchy radar), Car B should trust Car A.
-   Fusion: $P_{fused} = 1 - (1-P_A)(1-P_B)$.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the "Reference Point" in a CPM?
    *   **A:** The location of the sender (usually the center of the rear axle). All object coordinates are relative to this point.
2.  **Q:** How does CPM help with "Left Turn Assist"?
    *   **A:** A car waiting to turn left can't see oncoming traffic blocked by a truck. The truck (or RSU) sends a CPM showing the oncoming cars.
3.  **Q:** What is the main risk of CPM?
    *   **A:** **Position Error Propagation.** If the Sender has bad GPS, the Receiver puts the objects in the wrong place.

### Challenge Task
**Task:** Ego-Frame Transformation.
1.  Modify `receive_cpm` to transform objects into Car B's *local* frame.
2.  $P_{local\_B} = R_B^T (P_{global} - P_B)$.
3.  Print the distance from Car B to the Pedestrian.

---

## 📚 Further Reading & References
-   [ETSI TR 103 562 (CPM Study)](https://www.etsi.org/deliver/etsi_tr/103500_103599/103562/02.01.01_60/tr_103562v020101p.pdf)
-   [Arriver (V2X Software)](https://arriver.com/)

---

**Day 88 Complete** | Phase 4: ADAS & Robotics Systems | Week 13: V2X Communication
