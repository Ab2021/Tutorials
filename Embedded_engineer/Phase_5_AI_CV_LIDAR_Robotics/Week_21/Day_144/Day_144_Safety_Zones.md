# Day 144: Safety Zones (Speed & Separation Monitoring)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 21: Collaborative Robotics (Cobots)

---

> **📝 Content Creator Instructions:**
> The closer you get, the slower I go.
> - **Focus:** Safety Laser Scanners (SICK/Keyence), Zone switching (Warning vs Stop), ISO 13855 Stopping Distance calculation ($S = K \times T + C$).
> - **Code:** A Python script `ssm_monitor.py` that reads a Lidar scan, checks integration with polygonal safety zones (Shapely), and scales the robot's velocity override percentage.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Calculate** the required Safety Zone size using the Stopping Distance Formula.
2.  **Implement** SSM (Speed and Separation Monitoring): $V_{robot} \propto Distance_{human}$.
3.  **Define** Warning Fields (Yellow) vs Protective Fields (Red).
4.  **Process** Lidar data to find the nearest intruder effectively.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Lidar (2D) or Simulation.

### Software Environment
```bash
pip install numpy shapely matplotlib
```

### Prior Knowledge
- Lidar Basics.
- Coordinate Transforms.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Safety Formula (ISO 13855)

How much space does a robot need to stop?
$$ S = (K \times T) + C $$
*   **$S$:** Minimum Distance (Protective Field Size).
*   **$K$:** Approach Speed of Human (Standard: $1600 mm/s$).
*   **$T$:** Total Response Time ($T_{scanner} + T_{plc} + T_{brake}$).
*   **$C$:** Intrusion Distance (Sensor resolution error).
*   *Example:* If Robot needs 0.5s to stop, Human needs $1.6m/s \times 0.5s = 0.8m$ buffer.

### 🔹 Part 2: SSM (Speed & Separation)

Instead of a fixed Stop, we scale speed.
*   **Zone 3 (Green):** $> 2.0m$. Full Speed (100%).
*   **Zone 2 (Yellow):** $1.0m - 2.0m$. Reduced Speed (30%).
*   **Zone 1 (Red):** $< 1.0m$. Stop (0%).
This increases productivity (robot doesn't stop unnecessarily).

### 🔹 Part 3: Muting

Sometimes the robot needs to enter a "Bad Zone" (e.g., to load a pallet).
*   **Muting:** Temporarily bypassing sensors *if* the process sequence guarantees safety (e.g., Robot is moving AWAY from the danger).

---

## 💻 Implementation: The Zone Monitor

We simulate a 2D Lidar and a Human walking into the zones.

### 🛠️ Project Structure
```text
day144_safety/
├── src/
│   ├── ssm_monitor.py
└── output/
    ├── zone_viz.png
```

### 👨‍💻 SSM Monitor (`src/ssm_monitor.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from matplotlib.patches import Polygon as PlotPolygon

class SafetyMonitor:
    def __init__(self):
        # Define Zones (as Polygons relative to Robot at 0,0)
        # Red: 1m radius box
        self.zone_red = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
        # Yellow: 2m radius box
        self.zone_yellow = Polygon([(-2.5, -2.5), (2.5, -2.5), (2.5, 2.5), (-2.5, 2.5)])
        
        self.max_speed = 1.0 # m/s
        self.current_speed_limit = 1.0

    def process_scan(self, scan_points):
        # scan_points: List of (x, y) tuples representing obstacles
        
        violation_level = "GREEN" # Default
        min_dist = 100.0
        
        for p in scan_points:
            point = Point(p[0], p[1])
            dist = point.distance(Point(0,0)) # Dist to robot center
            if dist < min_dist: min_dist = dist
            
            # Hierarchical Check (Inner first)
            if self.zone_red.contains(point):
                violation_level = "RED"
                break # Worst case found
            elif self.zone_yellow.contains(point):
                if violation_level != "RED":
                    violation_level = "YELLOW"
        
        # Apply Logic
        if violation_level == "RED":
            self.current_speed_limit = 0.0
            print(f"!!! RED ZONE VIOLATION at {min_dist:.2f}m !!! STOPPING.")
        elif violation_level == "YELLOW":
            # Linear scaling? Or Fixed Step? ISO usually prefers Step or calculated curve.
            # Let's use simple Step.
            self.current_speed_limit = 0.3 # 30%
            print(f"Warning: Yellow Zone at {min_dist:.2f}m. Slowing down.")
        else:
            self.current_speed_limit = 1.0
            print("Zone Clear. Full Speed.")
            
        return self.current_speed_limit, violation_level

def main():
    monitor = SafetyMonitor()
    
    # Simulate Human walking in
    # Path: from (4,0) to (0,0)
    human_path = [ (x, 0) for x in np.linspace(4, 0.5, 20) ]
    
    speeds = []
    dists = []
    
    for h_pos in human_path:
        # Lidar sees Human + some random noise points
        scan = [h_pos] 
        # Add walls (far away)
        scan.append((5, 5))
        scan.append((-5, 5))
        
        speed_limit, status = monitor.process_scan(scan)
        
        speeds.append(speed_limit)
        dists.append(h_pos[0])
        
    # Visualization
    fig, ax = plt.subplots(figsize=(8,8))
    
    # Draw Zones
    x,y = monitor.zone_yellow.exterior.xy
    ax.add_patch(PlotPolygon(np.column_stack((x,y)), alpha=0.3, color='yellow', label='Warning'))
    
    x,y = monitor.zone_red.exterior.xy
    ax.add_patch(PlotPolygon(np.column_stack((x,y)), alpha=0.3, color='red', label='Protective'))
    
    # Draw Path
    path_x = [p[0] for p in human_path]
    path_y = [p[1] for p in human_path]
    ax.plot(path_x, path_y, 'b-o', label='Human Path')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.legend()
    plt.savefig("output/zone_viz.png")
    
    # Speed Plot
    plt.figure()
    plt.plot(dists, speeds, 'k-')
    plt.xlabel("Human Distance (m)")
    plt.ylabel("Robot Speed Limit (m/s)")
    plt.gca().invert_xaxis() # Right to Left approach
    plt.grid()
    plt.title("SSM Response")
    plt.savefig("output/ssm_response.png")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Blind Spot"

### 1. Lab Objectives
- **Run:** Sim.
- **Fail:** Why is there a blind spot behind the robot? (Lidar field of view often $270^\circ$).
- **Scenario:** Human approaches from angle $180^\circ$ (Directly behind). Dist = 0.5m.
- **Result:** Robot does not stop. $V=100\%$. Dangerous.
- **Fix:** Add a second Lidar on the back (Series connection) or use $360^\circ$ setup.

---

## 🚀 Project: "Dynamic Zone Switching"

**Goal:** Zones that change shape based on Robot Velocity.
1.  **State:** Robot moves Forward ($+X$).
2.  **Zone:** Elongate Red Zone in $+X$ direction ($S = V \cdot T$).
3.  **State:** Robot turns Left.
4.  **Zone:** Bulge Zone to Left.
5.  **Benefit:** Don't stop for obstacles that are *not in the path*.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Dust/Fog Triggers"
*   **Cause:** Lidar reflects off dust particles.
*   **Result:** False Stop. Productivity loss.
*   **Fix:** Multi-echo evaluation (Safety Scanners do this). Or filter isolated single pixels.

#### 2. "Latency"
*   **Cause:** Python is slow.
*   **Fix:** Safety logic MUST run on Safety Rated PLC or dedicated Controller (C++). Python is for understanding logic only. **NEVER use Python for real safety functions.**

---

## ⚡ Optimization: 3D Safety

2D Lidar misses an arm reaching *over* the zone.
*   **Solution:** 3D ToF Safety Cameras (e.g., Pilz SafetyEYE).
*   **Zones:** 3D Volumes (Virtual Cages).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Response Time" ($T$)?
    *   **A:** Time from "Event happens" (Light break) to "Robot actually stops moving". Includes processing + mechanical brake engagement.
2.  **Q:** Why not just calculate Distance = Norm(Robot - Human)?
    *   **A:** Robots are distinct shapes. An arm extended 2m needs a larger zone than a retracted arm. Polygon clipping is safer than centroid distance.
3.  **Q:** Can I use a generic webcam for safety?
    *   **A:** No. Safety sensors must be SIL3/PLe rated (Dual channel, Fail-safe). Webcams hang, focus hunt, and crash.

### Challenge Task
> **Task:** Latency Impact.
> 1. Set Robot Speed $V = 2m/s$.
> 2. Set Response Time $T = 0.5s$.
> 3. Calculate Stop Dist = $1m$.
> 4. Change Response Time to $0.1s$. Stop Dist = $0.2m$.
> 5. **Lesson:** Low latency hardware = Smaller zones = Less floor space usage.

---

## 📚 Further Reading
- **SICK Sensor Intelligence:** "Guide to Safe Machinery".
- **ISO 13855:** Standard for positioning safeguards.

---

**Day 144 Complete**
