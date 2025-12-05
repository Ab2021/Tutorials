# Day 116: Multi-Robot SLAM (Map Merging)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 17: Swarm Robotics

---

> **📝 Content Creator Instructions:**
> Two sets of eyes are better than one, but only if they agree on what they see.
> - **Focus:** TF Trees for Multi-Robot (Namespaces), Map Stitching mechanics (Occupancy Grid Merging), and Known vs Unknown Initial Poses.
> - **Code:** A `map_merger_node` that takes two local maps (with known initial offsets) and produces a fused `/global_map`.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Structure** TF trees for multiple robots (Preventing `map` frame conflicts).
2.  **Merge** two `OccupancyGrid` messages into one, resolving overlaps (Max Probability).
3.  **Handle** the "Initial Pose" problem (Start locations must be known or estimated).
4.  **Visualize** the merged map in Rviz while showing both robots moving live.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Simulation (2 Robots mapping different rooms).

### Software Environment
```bash
sudo apt install ros-humble-slam-toolbox ros-humble-nav2-map-server
pip install numpy
```

### Prior Knowledge
- SLAM (GMapping/SLAM Toolbox).
- Occupancy Grids (0-100, -1).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The TF Tree Problem

*   **Single Robot:** `map` $\to$ `odom` $\to$ `base_link`.
*   **Multi Robot:**
    *   **Bad:** `map` $\to$ `robot1/odom` AND `map` $\to$ `robot2/odom`? (Only if strictly localized).
    *   **SLAM:** Each robot runs its own SLAM.
    *   **Result:** `robot1/map` $\to$ `robot1/odom` ... and `robot2/map` $\to$ `robot2/odom`.
    *   **Fusion:** We need a `world` frame connecting `robot1/map` and `robot2/map`.

### 🔹 Part 2: Map Merging Logic

1.  **Grid Alignment:** Robot 1 map is $2000 \times 2000$. Robot 2 map is $500 \times 500$ at offset $(50, 50)$.
2.  **Probability Update:**
    *   If Map A says FREE and Map B says OCCUPIED $\to$ OCCUPIED (usually).
    *   **Bayesian:** $Odds_{new} = Odds_A \times Odds_B$.
    *   **Simplified (Max):** $M_{fused}(x,y) = \max(M_A(x,y), M_B(x,y))$.

### 🔹 Part 3: Bandwidth

Sharing full maps (10MB) over WiFi is bad.
*   **Solution:** Submap transmission. Only send updates (Changed pixels).
*   **Solution:** Graph transmission. Send Pose Graph nodes/edges, re-render map centrally.

---

## 💻 Implementation: Simple Map Merger

We assume we know the transform between Robot 1's Start and Robot 2's Start.

### 🛠️ Project Structure
```text
day116_multislam/
├── src/
│   ├── map_merger.py
└── launch/
    ├── multi_slam.launch.py
```

### 👨‍💻 Merger Node (`src/map_merger.py`)

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import tf_transformations

class MapMerger(Node):
    def __init__(self):
        super().__init__('map_merger')
        self.pub = self.create_publisher(OccupancyGrid, '/world/map', 10) # Fused Frame
        
        # Subscriptions
        self.create_subscription(OccupancyGrid, '/robot_1/map', self.map1_cb, 10)
        self.create_subscription(OccupancyGrid, '/robot_2/map', self.map2_cb, 10)
        
        # Storage
        self.map1 = None
        self.map2 = None
        
        # Known Offsets (Robot 2 starts at x=10.0 relative to Robot 1)
        # In reality, obtain this via TF or measuring tape
        self.r2_offset_x = 10.0 
        self.r2_offset_y = 0.0
        
        self.create_timer(2.0, self.merge) # Merge every 2s

    def map1_cb(self, msg): self.map1 = msg
    def map2_cb(self, msg): self.map2 = msg

    def merge(self):
        if not self.map1 or not self.map2: return
        
        # 1. Determine Global Grid Bounds
        # Simplified: Assume Resolution is same (0.05)
        res = self.map1.info.resolution
        w1, h1 = self.map1.info.width, self.map1.info.height
        w2, h2 = self.map2.info.width, self.map2.info.height
        
        # Offsets in pixels
        off_x_px = int(self.r2_offset_x / res)
        # Note: Origin of local maps (msg.info.origin) also matters!
        # This implementation assumes origins are (0,0) relative to start pose for simplicity
        
        # Create Giant Canvas
        global_w = max(w1, w2 + off_x_px)
        global_h = max(h1, h2)
        
        grid = np.full((global_h, global_w), -1, dtype=np.int8) # -1 = Unknown
        
        # 2. Paste Map 1
        # Convert flat data to 2D
        d1 = np.array(self.map1.data).reshape((h1, w1))
        # Logic: If known, overwrite.
        grid[0:h1, 0:w1] = d1
        
        # 3. Paste Map 2
        d2 = np.array(self.map2.data).reshape((h2, w2))
        
        # Overlap Region handling
        # Define ROI
        y_start, y_end = 0, h2
        x_start, x_end = off_x_px, off_x_px + w2
        
        # Slice
        current_roi = grid[y_start:y_end, x_start:x_end]
        
        # Fusion Logic (Max pooling for Obstacles)
        # Where Robot 2 sees known space, we take max(R1, R2)
        # 0 (Free) vs 100 (Occupied) -> 100
        # -1 (Unknown) vs 0 -> 0 
        
        # Numpy magic:
        # Create masks where data is valid
        valid_d2 = (d2 != -1)
        
        # If ROI -1, take d2. If d2 -1, keep ROI. If both valid, take MAX.
        merged_roi = np.maximum(current_roi, d2)
        
        # Problem: -1 is "Stored" as high number if interpreted unsigned? No, int8 signed.
        # But max(-1, 0) = 0. Correct.
        # max(-1, 100) = 100. Correct.
        # But max(0, 100) = 100. Correct.
        # Edge case: max(0, 0) = 0.
        
        # Update Canvas
        # Only overwrite where D2 has info? Or merge?
        # The np.maximum works well for Occupancy.
        grid[y_start:y_end, x_start:x_end] = merged_roi

        # 4. Publish
        out = OccupancyGrid()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "world" # The Global Frame
        out.info.resolution = res
        out.info.width = global_w
        out.info.height = global_h
        out.info.origin.position.x = self.map1.info.origin.position.x # Approx
        out.info.origin.position.y = self.map1.info.origin.position.y
        out.data = grid.flatten().tolist()
        
        self.pub.publish(out)

def main():
    rclpy.init()
    rclpy.spin(MapMerger())
```

---

## 🔬 Lab Exercise: "Double Agent"

### 1. Lab Objectives
- **Launch:** Two Turtlebots in Gazebo (House World).
- **Separation:** Spawn `robot2` at `(x=5, y=0)`.
- **Run:** SLAM Toolbox on both namespaces.
- **Drive:** Move `robot1` to map left side. Move `robot2` to map right side.
- **Merge:** Run `map_merger`.
- **Result:** A single continuous map of the house, appearing in Rviz under `/world/map`.

---

## 🚀 Project: "Unknown Initial Pose"

**Goal:** Merge maps without knowing `r2_offset`.
1.  **Requirement:** Robots must overlap.
2.  **Algorithm:** Feature Matching (SIFT/ORB) on the Map Image.
3.  **Process:**
    *   Convert OccupancyGrid $\to$ Image (cv2).
    *   Detect corners/features.
    *   Estimate Affine Transform (RANSAC).
    *   Calculate relative offset $(dx, dy, d\theta)$.
4.  **Result:** "Loop Closure" between two different robots.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Map Flickering"
*   **Cause:** Slight drift in Odometry causes the merge to jitter.
*   **Fix:** Only accept *new* transforms if confidence is high, or lock the transform once established.

#### 2. "Memory OOM"
*   **Cause:** Numpy array for 4000x4000 map is small (16MB). But ROS message serialization + List conversion in Python can balloon RAM.
*   **Fix:** Use C++ for map merging. Python `tolist()` on large arrays is slow and memory hungry.

---

## ⚡ Optimization: 3D Map Merging (Octomap)

2D is limited. 3D is better.
*   **OctoMap:** Probabilistic 3D Voxel Grid.
*   **Merging:** Simply add point clouds from Robot 2 (transformed to World) into Robot 1's Octree.
*   OctoMap handles the probablistic update natively.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the `frame_id` of the merged map?
    *   **A:** Usually `world` or `earth`. It is the parent of `robot1/map` and `robot2/map`.
2.  **Q:** Why not just have one SLAM node subscribed to 2 Lidars?
    *   **A:** Network Lag. Transmitting raw Lidar scan (High Bandwidth) over WiFi to a central server is risky. Mapping locally and sending Maps (Low Bandwidth) is robust.

### Challenge Task
> **Task:** Multi-Robot Exploration.
> 1. Use `nav2_simple_commander`.
> 2. Robot 1 goes to `(2, 2)`. Robot 2 goes to `(12, 2)`.
> 3. Verify the merged map grows in two directions simultaneously.

---

## 📚 Further Reading
- **map_merge_2d:** Existing ROS package for this.
- **SLAM Toolbox:** Supports multi-session mapping (similar concept).

---

**Day 116 Complete**
