# Day 132: HD Maps & Localization
## Phase 4: ADAS & Robotics Systems | Week 19: Localization & SLAM

---

> **📝 Day 132 Focus:**
> SLAM builds a map. But in production (Waymo/Cruise), we often drive in areas we have *already* mapped. **HD Map Localization** uses a pre-built, centimeter-accurate map to find the car's position. It's robust and drift-free.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Explain** the structure of an HD Map (Lanelet2 format).
2.  **Differentiate** between SLAM (Mapping) and Localization (Matching).
3.  **Implement** NDT Localization against a global Point Cloud Map.
4.  **Perform** Map Matching (snapping GPS to Lane Centerlines).
5.  **Visualize** the car driving on an HD Map.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 130:** NDT/ICP.
-   **Day 127:** GNSS coordinates.

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `open3d`, `xml.etree.ElementTree` (for parsing OSM/Lanelet2).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The HD Map

Standard maps (Google Maps) have 5m accuracy. HD Maps have **10cm** accuracy.
Layers:
1.  **Geometric:** Point Cloud (PCD) of the environment (Buildings, Poles).
2.  **Semantic:** Vector Map (Lanelet2). Lane lines, Stop lines, Traffic lights.
3.  **Dynamic:** Real-time traffic info (V2X).

### 🔹 Part 2: Map-Based Localization

Instead of matching Scan $t$ to Scan $t-1$ (Odometry), we match Scan $t$ to the **Global Map**.
-   **Global Map:** Huge point cloud ($10^9$ points).
-   **Local Scan:** Small point cloud ($10^5$ points).
-   **Algorithm:** NDT (Normal Distributions Transform).
    -   Initialize with GPS guess.
    -   Align Local Scan to Global Map.
    -   Result: Absolute Pose in Map Frame.

### 🔹 Part 3: Lanelet2 Format

Based on OpenStreetMap (OSM).
-   **Points:** Lat/Lon.
-   **Linestrings:** Connected points (Lane boundaries).
-   **Lanelets:** Atomic lane segments (Left bound + Right bound).
-   **Relations:** Connectivity (Successor, Predecessor, Adjacent).

---

## 💻 Implementation: NDT Localization

**Scenario:**
-   Global Map: A pre-scanned street (Synthetic).
-   Car: Moves through the street.
-   Task: Localize the car using NDT.

### 🛠️ Setup
Create `week19_day132` and `ndt_localizer.py`.

```bash
mkdir -p ~/ros2_ws/src/week19_day132
cd ~/ros2_ws/src/week19_day132
touch ndt_localizer.py
```

### 👨‍💻 Code: Localizing in a Map

```python
import open3d as o3d
import numpy as np
import copy
import time

def create_global_map():
    # Create a "Street" with poles and walls
    pcd = o3d.geometry.PointCloud()
    points = []
    
    # Ground
    for x in range(0, 100, 1):
        for y in range(-10, 10, 1):
            points.append([x, y, 0])
            
    # Walls
    for x in range(0, 100, 1):
        for z in range(0, 5, 1):
            points.append([x, -10, z]) # Left Wall
            points.append([x, 10, z])  # Right Wall
            
    # Poles every 20m
    for x in range(10, 100, 20):
        for z in range(0, 8, 1):
            points.append([x, -8, z])
            
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    # Downsample for NDT grid
    pcd = pcd.voxel_down_sample(voxel_size=0.5)
    return pcd

def simulate_lidar_scan(pose, global_map):
    # Crop global map to simulate local view
    # In real life, this is ray casting. Here, simple cropping.
    
    # Transform map to robot frame (Inverse of pose)
    inv_pose = np.linalg.inv(pose)
    local_map = copy.deepcopy(global_map)
    local_map.transform(inv_pose)
    
    # Crop box (-20 to 20m)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=[-20, -20, -2], max_bound=[20, 20, 10])
    scan = local_map.crop(bbox)
    
    # Add noise
    pts = np.asarray(scan.points)
    pts += np.random.normal(0, 0.05, pts.shape)
    scan.points = o3d.utility.Vector3dVector(pts)
    
    return scan

def main():
    # 1. Load Map
    print("Loading Global Map...")
    global_map = create_global_map()
    
    # 2. Simulate Robot Motion
    # Robot moves at x=0 to x=50
    true_poses = []
    for x in range(0, 50, 2):
        T = np.eye(4)
        T[0, 3] = x
        true_poses.append(T)
        
    # 3. Localization Loop
    # Initial Guess (GPS) - Add some error
    curr_pose = np.eye(4)
    curr_pose[0, 3] = -2.0 # 2m error
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(global_map)
    
    # Robot geometry for visualization
    robot_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(robot_mesh)
    
    print("Starting Localization...")
    
    for i, true_pose in enumerate(true_poses):
        # A. Get Sensor Data
        scan = simulate_lidar_scan(true_pose, global_map)
        
        # B. Prediction Step (Motion Model)
        # Assume constant velocity 2m/s
        motion = np.eye(4)
        motion[0, 3] = 2.0
        curr_pose = curr_pose @ motion
        
        # C. Update Step (NDT Registration)
        # Align Scan to Map using curr_pose as initial guess
        
        # Note: Open3D ICP is used here as a proxy for NDT
        # (Open3D's NDT implementation is less standard, ICP Point-to-Plane is similar)
        
        # Transform scan to initial guess
        scan_in_map = copy.deepcopy(scan)
        scan_in_map.transform(curr_pose)
        
        reg = o3d.pipelines.registration.registration_icp(
            scan_in_map, global_map, 1.0, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        # Refine Pose
        # reg.transformation is the correction matrix
        curr_pose = reg.transformation @ curr_pose
        
        # Error Check
        error = np.linalg.norm(curr_pose[:3, 3] - true_pose[:3, 3])
        print(f"Step {i}: Error = {error:.3f} m")
        
        # Visualize
        robot_mesh.transform(curr_pose) # This is wrong (accumulates). Need to reset.
        # Reset mesh
        robot_mesh.vertices = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0).vertices
        robot_mesh.transform(curr_pose)
        
        vis.update_geometry(robot_mesh)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)
        
    vis.destroy_window()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Kidnapped Robot (Again)

### Lab Objectives
1.  Run the script.
2.  **Observation:** The coordinate frame (Robot) tracks the path along the street. The error stays low (< 10cm).
3.  **Experiment:**
    -   Set `curr_pose[0, 3] = 10.0` (10m error) at start.
    -   **Result:** ICP/NDT might fail to converge or snap to the wrong pole (Aliasing).
    -   **Lesson:** Map-based localization needs a good initial guess (from GPS/Particle Filter).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Map Changes
**Symptom:** Localization fails in a construction zone.
**Cause:** The map is outdated. The scan sees a fence, the map sees an empty road.
**Solution:** Probabilistic Map Update (SLAM) or Robust Kernels (ignore outliers).

#### 2. Z-Drift
**Symptom:** Car sinks into the ground.
**Cause:** Flat ground (highway) has no Z-features. NDT slides up/down.
**Solution:** Fuse with IMU (Gravity vector) or clamp Z to the map surface.

---

## ⚡ Optimization & Best Practices

### 1. Multi-Resolution Maps
-   Use a coarse grid (1m) for initial alignment.
-   Use a fine grid (10cm) for final refinement.
-   Speeds up convergence.

### 2. Keyframe-Based Map
Don't load the whole 100GB map.
-   Stream map tiles based on GPS position.
-   Keep only the local 200m radius in memory.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between a Point Cloud Map and a Vector Map?
    *   **A:** Point Cloud is raw geometry (dots). Vector Map is semantic meaning (lines, lanes, rules). Localization uses Point Cloud. Planning uses Vector Map.
2.  **Q:** Why is NDT preferred over ICP for Map Localization?
    *   **A:** NDT is faster (grid lookup vs KD-tree search) and handles density variations better.
3.  **Q:** What is "Map Matching"?
    *   **A:** Snapping the estimated position to the nearest logical lane center. "I am 10cm left of the center line".

### Challenge Task
**Task:** Lanelet2 Parsing.
1.  Download a sample `.osm` file (Lanelet2).
2.  Use Python to parse the XML.
3.  Extract all nodes with tag `type=line_thin`.
4.  Plot the lane lines in Matplotlib.

---

## 📚 Further Reading & References
-   [Lanelet2 Paper](https://ieeexplore.ieee.org/document/8569077)
-   [Autoware Localization Documentation](https://autowarefoundation.github.io/autoware-documentation/)

---

**Day 132 Complete** | Phase 4: ADAS & Robotics Systems | Week 19: Localization & SLAM
