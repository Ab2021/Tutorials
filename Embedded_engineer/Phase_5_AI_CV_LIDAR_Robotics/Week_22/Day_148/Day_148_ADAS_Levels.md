# Day 148: ADAS Levels & Sensor Sets
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 22: Autonomous Driving Stack

---

> **📝 Content Creator Instructions:**
> From Cruise Control to Robotaxis.
> - **Focus:** SAE Levels 0-5, Sensor Architectures (Camera-First vs Lidar-First), Coordinate Frames (Ego, Map, Sensor), and Extrinsic Calibration.
> - **Code:** A Python script `ego_vehicle.py` that defines a Robot Class with multiple sensors, managing their Rigid Body Transforms (Extrinsics) and simulating data synchronization.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** SAE Levels 2 (Tesla Autopilot), 3 (Traffic Jam Pilot), and 4 (Waymo).
2.  **Design** a Sensor Suite: Where to place Lidars/Cameras for $360^\circ$ coverage.
3.  **Implement** a TF Tree for an Ego Vehicle (Base Link $\to$ Camera Link).
4.  **Simulate** Time Synchronization (PTP) issues between sensors.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy scipy matplotlib
```

### Prior Knowledge
- Homogeneous Transformation Matrices (Day 15).
- Reference Frames.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Levels of Autonomy (SAE J3016)

*   **L0:** No automation (Anti-lock brakes don't count).
*   **L1:** Driver Assistance (ACC **OR** Lane Keep).
*   **L2:** Partial Automation (ACC **AND** Lane Keep). Driver hands on wheel (or eyes on road).
*   **L3:** Conditional Automation. Car drives self in traffic jam. Driver can watch Netflix but must take over in 10s if requested. (Liability shifts to Car).
*   **L4:** High Automation. No driver needed in Geo-fenced area (Robotaxi in SF).
*   **L5:** Full Automation. Anywhere, anytime (Dirt road in blizzard). Does not exist yet.

### 🔹 Part 2: Sensor Suites

*   **Camera-First (Tesla):** Cheap. High Res. Needs massive AI. Fails in fog/darkness?
*   **Lidar-First (Waymo/Cruise/Zoox):** Expensive ($100k+). Accurate Depth. Geometric truth. "Ugly" roof racks.
*   **Radar:** Best for speed (Doppler) and weather. Low resolution.

### 🔹 Part 3: Calibration (Extrinsics)

Where is the camera relative to the wheel center?
$$ T_{base \to cam} = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} $$
*   If this is off by $1^\circ$, a car at 100m looks like it's in the next lane.
*   **Target:** $< 0.1^\circ$ rotational error.

---

## 💻 Implementation: The Virtual Car

We define a class that manages the TF tree of a car.

### 🛠️ Project Structure
```text
day148_adas/
├── src/
│   ├── ego_vehicle.py
└── output/
    ├── sensor_viz.png
```

### 👨‍💻 Ego Vehicle Model (`src/ego_vehicle.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class Sensor:
    def __init__(self, name, x, y, z, roll, pitch, yaw, fov, max_range):
        self.name = name
        # Extrinsic: Transform from Base_Link to Sensor
        self.pos = np.array([x, y, z])
        self.rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        self.fov = fov
        self.max_range = max_range
        
    def get_transform(self):
        # 4x4 Homogeneous Matrix
        T = np.eye(4)
        T[:3, :3] = self.rot.as_matrix()
        T[:3, 3] = self.pos
        return T

    def plot_fov(self, ax, vehicle_pose):
        # Draw sensor origin in Global Frame
        # vehicle_pose: T_map_base
        # T_map_sensor = T_map_base * T_base_sensor
        
        T_base_sensor = self.get_transform()
        T_map_sensor = vehicle_pose @ T_base_sensor
        
        origin = T_map_sensor[:3, 3]
        
        # Draw FOV Cone (Simplified 2D Triangle on Ground)
        # Heading vector in Sensor Frame: X axis [1, 0, 0]
        # FOV vectors: Rotated +/- FOV/2
        
        yaw_sensor = R.from_matrix(T_map_sensor[:3, :3]).as_euler('xyz')[2]
        
        wedge_len = self.max_range
        angle_left = yaw_sensor + np.radians(self.fov/2)
        angle_right = yaw_sensor - np.radians(self.fov/2)
        
        p_left = origin + wedge_len * np.array([np.cos(angle_left), np.sin(angle_left), 0])
        p_right = origin + wedge_len * np.array([np.cos(angle_right), np.sin(angle_right), 0])
        
        # Plot triangle
        pts = np.array([origin, p_left, p_right])
        ax.fill(pts[:,0], pts[:,1], alpha=0.2, label=self.name)
        ax.plot(origin[0], origin[1], 'ko', markersize=3)

class EgoVehicle:
    def __init__(self):
        # Waymo-style config
        self.sensors = [
            Sensor("Top Lidar", 0.0, 0.0, 2.0, 0, 0, 0, 360, 50),
            Sensor("Front Camera", 1.5, 0.0, 1.4, 0, 0, 0, 60, 40),
            Sensor("Side Left Cam", 1.0, 0.9, 1.4, 0, 0, 90, 80, 30),
            Sensor("Side Right Cam", 1.0, -0.9, 1.4, 0, 0, -90, 80, 30),
            Sensor("Rear Radar", -1.0, 0.0, 0.5, 0, 0, 180, 90, 60)
        ]
        
        # Vehicle State
        self.pose = np.eye(4) # At 0,0,0
        
    def drive(self, x, y, yaw_deg):
        self.pose[:3, 3] = [x, y, 0]
        self.pose[:3, :3] = R.from_euler('z', yaw_deg, degrees=True).as_matrix()

def main():
    car = EgoVehicle()
    
    # Scene
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Plot Car at Origin
    car.drive(0, 0, 0)
    
    # Draw Car Body (Rectangle)
    rect = plt.Rectangle((-2, -1), 4, 2, color='gray', alpha=0.5)
    ax.add_patch(rect)
    
    for s in car.sensors:
        s.plot_fov(ax, car.pose)
        T = s.get_transform()
        print(f"{s.name}: Pos={T[:3,3]}")
    
    # 2. Add an Obstacle
    obs = np.array([30, 10, 0])
    ax.plot(obs[0], obs[1], 'rx', markersize=10, label="Pedestrian")
    
    # Check Logic: Who sees the pedestrian?
    # (Simplified Distance/Angle check)
    print("\n--- Detection Report ---")
    for s in car.sensors:
        # Transform Obs to Sensor Frame
        T_bs = s.get_transform()
        T_sb = np.linalg.inv(T_bs)
        
        obs_local = T_sb @ np.append(obs, 1)
        dist = np.linalg.norm(obs_local[:3])
        angle = np.degrees(np.arctan2(obs_local[1], obs_local[0]))
        
        if dist < s.max_range and abs(angle) < s.fov/2:
            print(f"✅ {s.name} detects Pedestrian! (Dist={dist:.1f}m, Ang={angle:.1f}deg)")
        else:
            print(f"❌ {s.name} missed.")
            
    ax.set_xlim(-20, 60)
    ax.set_ylim(-40, 40)
    ax.grid()
    ax.legend()
    ax.set_title("Ego Vehicle Sensor Coverage")
    plt.savefig("output/sensor_viz.png")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Blind Spot"

### 1. Lab Objectives
- **Run:** The script.
- **Observe:** Front Cam and Lidar see the pedestrian. Rear Radar misses.
- **Task:** Move the pedestrian to `(-10, 10, 0)` (Rear Left).
- **Result:** Depending on the setup, *No sensor* might see them. This is a blind spot.
- **Action:** Add a "Blind Spot Radar" at the rear bumper corners ($x=-1, y=1, yaw=135$).

---

## 🚀 Project: "Multi-Sensor Sync"

**Goal:** Simulate Timestamp jitter.
1.  **Lidar:** 10Hz (every 100ms).
2.  **Camera:** 30Hz (every 33ms).
3.  **Problem:** Camera image at $t=33ms$, Lidar at $t=100ms$.
4.  **Fusion:** You must interpolate the Vehicle Pose to match timestamps.
5.  **Task:** Write a queue that buffers Camera images until the next Lidar scan arrives (or vice versa).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Coordinate Hell"
*   **Cause:** Standard Camera Frame is Z-forward (Optical). Standard Lidar/Car Frame is X-forward (ISO 8855).
*   **Result:** The ground plane appears as a vertical wall in the camera projection.
*   **Fix:** Always apply the Optical-to-Base static transform.
    $$ R_{opt \to base} = \begin{bmatrix} 0 & -1 & 0 \\ 0 & 0 & -1 \\ 1 & 0 & 0 \end{bmatrix} $$

#### 2. "Rolling Shutter Distortion"
*   **Cause:** Car moving fast. Top of image taken at $t$, bottom at $t + 10ms$. Vertical lines slant.
*   **Fix:** Use Global Shutter cameras for robotics.

---

## ⚡ Optimization: Hardware Synchronization

Software sync (matching timestamps) is usually enough for slow speeds.
For $100km/h$ on Highway:
*   **PTP (Precision Time Protocol):** Sub-microsecond clock sync over Ethernet.
*   **Hardware Trigger:** Lidar fires a pulse. Camera opens shutter *exactly* when Lidar fires.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why did Tesla remove Radar?
    *   **A:** "Phantom Braking". Radar sees a bridge as a wall (due to low elevation resolution). Camera sees bridge. Disagreement $\to$ Confusion. Tesla bet on Camera being the "Single Source of Truth".
2.  **Q:** What is the "Long Tail"?
    *   **A:** The infinite number of weird edge cases (Unicyclist in a chicken suit). L5 autonomy requires solving the Long Tail.
3.  **Q:** Why not use GPS for everything?
    *   **A:** Tunnels. Urban Canyons. GPS accuracy is ~3m. Lanes are 3.5m wide. You need cm-level relative accuracy (Lidar/Cam).

### Challenge Task
> **Task:** Pitch Compensation.
> 1. Car brakes hard $\to$ Nose dives (Pitch changes).
> 2. Camera FOV shifts down. Horizon moves up.
> 3. Simulate this effect in detection logic. Does detecting a traffic light at 100m fail during braking?

---

## 📚 Further Reading
- **SAE J3016:** Official Levels of Autonomy.
- **Waymo Safety Report:** Detailed breakdown of their sensor suite.

---

**Day 148 Complete**
