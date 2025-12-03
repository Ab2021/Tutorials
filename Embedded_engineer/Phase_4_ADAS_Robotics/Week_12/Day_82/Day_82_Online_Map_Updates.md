# Day 82: Online Map Updates (Crowdsourcing)
## Phase 4: ADAS & Robotics Systems | Week 12: HD Maps & Map Matching

---

> **📝 Day 82 Focus:**
> The world changes. Construction zones appear, lanes are repainted, and signs fall down. A static HD Map becomes dangerous if it's outdated. **Online Map Updating** uses the fleet of cars as "Surveyors" to detect changes and update the map in the cloud.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the concept of Crowdsourced Mapping (Fleet Learning).
2.  **Design** a Change Detection algorithm (Map vs Perception).
3.  **Implement** a "Map Diff" system to identify discrepancies.
4.  **Understand** the Map Versioning problem (Tile-based updates).
5.  **Simulate** a scenario where a car detects a new Stop Sign and uploads it.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 79:** Map Layers (Signs).
-   **Day 81:** Semantic Mapping.
-   **Cloud:** Basic client-server concept.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `json`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Stale Map Problem

HD Maps are expensive to create (Survey vehicles + Manual annotation).
If a speed limit changes from 50 to 30, and the map says 50, the AV will speed.
**Solution:** Use consumer cars (equipped with cameras/Mobileye) to verify the map.
-   **Verification:** "I see a 50 sign. Map says 50. Good."
-   **Discrepancy:** "I see a 30 sign. Map says 50. Flag it!"

### 🔹 Part 2: Change Detection Logic

1.  **Geometric Change:**
    -   Lane markings shifted (Construction).
    -   Detected by comparing Lane Detector Polynomial vs Map Lane Polynomial.
2.  **Semantic Change:**
    -   New/Missing Sign.
    -   Detected by comparing Object Detection List vs Map Landmark List.
3.  **Confidence Threshold:**
    -   One car reporting a change might be noise (sensor error).
    -   **N cars** reporting the same change = High Confidence Update.

### 🔹 Part 3: Map Tiles and Versioning

Maps are tiled (e.g., 1km x 1km squares).
-   **Versioning:** Each tile has a hash/version.
-   **OTA Update:** The car only downloads tiles that have changed since its last update.
-   **NDS (Navigation Data Standard):** Industry standard for map tiling and updates.

---

## 💻 Implementation: Map Diff Generator

**Scenario:**
-   **Server:** Holds the "Golden Map" (List of Signs).
-   **Car:** Drives through, detects signs, and compares with Map.
-   **Output:** A "Diff" report (Add/Remove/Modify).

### 🛠️ Setup
Create `week12_day82` and `map_updater.py`.

```bash
mkdir -p ~/ros2_ws/src/week12_day82
cd ~/ros2_ws/src/week12_day82
touch map_updater.py
```

### 👨‍💻 Code: Map Diff Logic

```python
import numpy as np
import json
import matplotlib.pyplot as plt

# --- Data Structures ---
class TrafficSign:
    def __init__(self, id, x, y, type, value=None):
        self.id = id # Map ID (None if new)
        self.x = x
        self.y = y
        self.type = type # SPEED_LIMIT, STOP
        self.value = value # e.g., 50
        
    def to_dict(self):
        return {'id': self.id, 'x': self.x, 'y': self.y, 'type': self.type, 'value': self.value}
        
    def distance(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class MapServer:
    def __init__(self):
        # Golden Map
        self.signs = [
            TrafficSign(1, 10, 10, "STOP"),
            TrafficSign(2, 50, 10, "SPEED_LIMIT", 50),
            TrafficSign(3, 90, 10, "YIELD")
        ]
        
    def get_map_in_roi(self, x_min, x_max):
        return [s for s in self.signs if x_min <= s.x <= x_max]
        
    def apply_update(self, diff):
        print("\n--- Server: Applying Update ---")
        for item in diff['added']:
            new_id = len(self.signs) + 100
            print(f"Adding Sign ID {new_id}: {item['type']} at ({item['x']:.1f}, {item['y']:.1f})")
            self.signs.append(TrafficSign(new_id, item['x'], item['y'], item['type'], item['value']))
            
        for item in diff['removed']:
            print(f"Removing Sign ID {item['id']}")
            self.signs = [s for s in self.signs if s.id != item['id']]
            
        for item in diff['modified']:
            print(f"Modifying Sign ID {item['id']}: Value {item['old_value']} -> {item['new_value']}")
            for s in self.signs:
                if s.id == item['id']:
                    s.value = item['new_value']

class AutonomousVehicle:
    def __init__(self, server):
        self.server = server
        self.pos_x = 0
        self.detected_signs = []
        
    def drive_and_scan(self):
        # Simulation: Car drives from 0 to 100
        # Scenario:
        # 1. Sign 1 (STOP) is there. (Match)
        # 2. Sign 2 (SPEED 50) is changed to SPEED 30. (Modify)
        # 3. Sign 3 (YIELD) is missing/fallen. (Remove)
        # 4. Sign 4 (New STOP) appears at 70, 10. (Add)
        
        print("Car: Driving and Scanning...")
        self.detected_signs = [
            TrafficSign(None, 10.1, 9.9, "STOP"),        # Matches ID 1 (with noise)
            TrafficSign(None, 50.2, 10.1, "SPEED_LIMIT", 30), # ID 2 Changed value
            # ID 3 Missing
            TrafficSign(None, 70.0, 10.0, "STOP")        # New Sign
        ]
        
    def generate_diff(self):
        # Download local map
        local_map = self.server.get_map_in_roi(0, 100)
        
        diff = {'added': [], 'removed': [], 'modified': []}
        
        # Matching Threshold
        MATCH_DIST = 2.0
        
        # 1. Check for Matches and Modifications
        matched_map_ids = set()
        
        for det in self.detected_signs:
            best_match = None
            min_dist = float('inf')
            
            for map_sign in local_map:
                d = det.distance(map_sign)
                if d < MATCH_DIST and d < min_dist:
                    min_dist = d
                    best_match = map_sign
            
            if best_match:
                matched_map_ids.add(best_match.id)
                # Check Attributes
                if det.type != best_match.type:
                    # Type mismatch usually means new sign replacing old, or error
                    pass 
                elif det.value != best_match.value:
                    diff['modified'].append({
                        'id': best_match.id,
                        'old_value': best_match.value,
                        'new_value': det.value
                    })
            else:
                # No match in map -> New Sign
                diff['added'].append(det.to_dict())
                
        # 2. Check for Removals
        for map_sign in local_map:
            if map_sign.id not in matched_map_ids:
                diff['removed'].append({'id': map_sign.id})
                
        return diff

def main():
    server = MapServer()
    car = AutonomousVehicle(server)
    
    # 1. Car drives
    car.drive_and_scan()
    
    # 2. Car computes diff
    diff = car.generate_diff()
    
    print("\n--- Diff Report Generated by Car ---")
    print(json.dumps(diff, indent=2))
    
    # 3. Server applies update (Simulated Crowdsourcing)
    # In reality, Server waits for N confirmations
    server.apply_update(diff)
    
    # 4. Verify Server State
    print("\n--- Updated Server Map ---")
    for s in server.signs:
        val = f" {s.value}" if s.value else ""
        print(f"ID {s.id}: {s.type}{val} at ({s.x}, {s.y})")

    # Visualization
    plt.figure(figsize=(10, 2))
    # Draw Road
    plt.plot([0, 100], [10, 10], 'k-', lw=10, alpha=0.2)
    
    # Plot Final Map
    for s in server.signs:
        color = 'r' if s.type == 'STOP' else 'b'
        if s.type == 'SPEED_LIMIT': color = 'g'
        plt.plot(s.x, s.y, 'o', color=color, markersize=15)
        plt.text(s.x, s.y+2, f"{s.type}\n{s.value if s.value else ''}", ha='center')
        
    plt.xlim(0, 110)
    plt.ylim(0, 20)
    plt.title("Updated Map State")
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The False Alarm

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   ID 2 changed to 30.
    -   ID 3 removed.
    -   New STOP sign added.
3.  **Experiment:**
    -   Simulate a "False Negative": The car misses Sign 1 (Occlusion).
    -   **Result:** The Diff reports "Remove ID 1".
    -   **Server Logic:** If the server applies this immediately, the map breaks.
    -   **Fix:** The server needs a **Trust Score**. "Remove ID 1" gets +1 vote. Only remove if votes > 10.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. GPS Drift
**Symptom:** All signs are detected as "New" and all map signs as "Removed".
**Cause:** The car's position is shifted by 5m. Nothing matches.
**Solution:**
    -   **Relative Matching:** Match the *pattern* of signs, not absolute coordinates.
    -   **Map Matching:** Snap the car to the map first (Day 80) before checking signs.

#### 2. Sign Classification Error
**Symptom:** "SPEED 50" detected as "SPEED 60".
**Cause:** Vision error.
**Solution:** Bayesian Fusion. Combine multiple detections from multiple frames/cars.

---

## ⚡ Optimization & Best Practices

### 1. Edge Computing
Don't upload video. Don't even upload all detections.
-   Compute the Diff **on the car**.
-   Only upload the Diff (KB of data).
-   Saves bandwidth (LTE/5G).

### 2. Privacy
Crowdsourcing tracks people's location.
-   **Anonymization:** Strip Vehicle ID.
-   **Spatial Obfuscation:** Only upload map updates, not the trajectory.
-   **Aggregation:** Server only stores "A car saw this", not "User X saw this".

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is "Fleet Learning"?
    *   **A:** Using the combined data from thousands of customer vehicles to improve the system (Map, Perception) for everyone.
2.  **Q:** Why is "Geometric Change" harder than "Semantic Change"?
    *   **A:** Lane lines are continuous and subtle. Signs are discrete and distinct. Detecting a 10cm shift in a lane requires high-precision localization.
3.  **Q:** What is NDS?
    *   **A:** Navigation Data Standard. A database format for maps that supports tiling and incremental updates.

### Challenge Task
**Task:** Confidence Accumulator.
1.  Modify `MapServer`. Add a `confidence` field to each sign (Default 10).
2.  If `added`: New sign starts with confidence 1. (Not published to fleet yet).
3.  If `confirmed`: Confidence += 1.
4.  If `removed`: Confidence -= 1.
5.  Publish only if `confidence > 5`.

---

## 📚 Further Reading & References
-   [Mobileye REM (Road Experience Management)](https://www.mobileye.com/our-technology/rem/)
-   [TomTom HD Map](https://www.tomtom.com/products/hd-map/)

---

**Day 82 Complete** | Phase 4: ADAS & Robotics Systems | Week 12: HD Maps & Map Matching
