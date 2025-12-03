# Day 81: Semantic Mapping
## Phase 4: ADAS & Robotics Systems | Week 12: HD Maps & Map Matching

---

> **📝 Day 81 Focus:**
> A geometric map tells you *where* things are. A **Semantic Map** tells you *what* things are. Is that obstacle a wall (permanent) or a parked truck (temporary)? Is that surface a road (drivable) or a sidewalk (forbidden)? Today, we add meaning to the map.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** Semantic Mapping and its role in Autonomous Driving (Drivable Area vs Obstacles).
2.  **Explain** the pipeline: LiDAR Point Cloud -> Semantic Segmentation -> Map Projection.
3.  **Implement** a simple Semantic Map Builder using labeled point clouds.
4.  **Distinguish** between Static (Buildings, Roads) and Dynamic (Cars, Pedestrians) elements during mapping.
5.  **Visualize** a colored semantic point cloud map.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 47:** Point Cloud Processing (PCL).
-   **Day 51:** Deep Learning for Point Clouds (PointNet/RandLA-Net).
-   **Day 71:** Coordinate Transformations.

### Hardware Requirements
-   **LiDAR:** (Optional) Velodyne/Ouster or Simulated data.

### Software Stack
-   **Python:** `numpy`, `open3d`, `sklearn`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is a Semantic Map?

A traditional Occupancy Grid has binary states: Free (0) or Occupied (1).
A **Semantic Map** stores a class label for every voxel/point:
-   **Label 0:** Unlabeled
-   **Label 1:** Road (Drivable)
-   **Label 2:** Sidewalk (Drivable but forbidden)
-   **Label 3:** Building (Static Obstacle)
-   **Label 4:** Vegetation (Semi-static)
-   **Label 5:** Car (Dynamic - should be removed from map)

### 🔹 Part 2: The Mapping Pipeline

1.  **Input:** Sequence of LiDAR scans + Pose (from SLAM/Localization).
2.  **Segmentation:** Run a Deep Neural Network (e.g., RangeNet++, Cylinder3D) on each scan to assign labels.
3.  **Dynamic Removal:** Filter out points labeled as "Car", "Pedestrian", "Cyclist". We only want the *background* map.
4.  **Transformation:** Transform points from Sensor Frame to Global Map Frame using the Pose.
5.  **Aggregation:** Add points to the global map (Voxel Grid or Point Cloud).
6.  **Refinement:** Ray casting to clear dynamic objects that moved (Ray tracing free space).

### 🔹 Part 3: Dynamic vs Static

-   **Static:** Walls, Poles, Traffic Signs. (Keep in Map).
-   **Dynamic:** Cars, People. (Remove).
-   **Semi-Static:** Parked Cars, Construction Cones. (Hard case! Usually keep as temporary obstacles or maintain a separate "Live" layer).

---

## 💻 Implementation: Semantic Map Builder

**Scenario:**
-   **Input:** Simulated LiDAR scans with ground truth labels (SemanticKITTI format).
-   **Task:** Aggregate them into a global semantic map, filtering out dynamic objects.

### 🛠️ Setup
Create `week12_day81` and `semantic_mapper.py`.

```bash
mkdir -p ~/ros2_ws/src/week12_day81
cd ~/ros2_ws/src/week12_day81
touch semantic_mapper.py
```

### 👨‍💻 Code: Semantic Mapper

```python
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# --- Constants ---
# SemanticKITTI Color Map (Simplified)
COLOR_MAP = {
    0: [0, 0, 0],       # Unlabeled (Black)
    1: [0.5, 0.5, 0.5], # Road (Gray)
    2: [0.2, 0.2, 0.2], # Sidewalk (Dark Gray)
    3: [1.0, 0.0, 0.0], # Building (Red)
    4: [0.0, 1.0, 0.0], # Vegetation (Green)
    5: [0.0, 0.0, 1.0], # Car (Blue) - Dynamic
    6: [1.0, 1.0, 0.0], # Pole (Yellow)
}

class SemanticScan:
    def __init__(self, points, labels, pose):
        self.points = points # Nx3
        self.labels = labels # Nx1
        self.pose = pose     # 4x4 Transformation Matrix

class SemanticMapper:
    def __init__(self, voxel_size=0.2):
        self.voxel_size = voxel_size
        self.global_pcd = o3d.geometry.PointCloud()
        
    def process_scan(self, scan):
        # 1. Filter Dynamic Objects
        # Keep only Static (Labels 1, 2, 3, 4, 6)
        # Remove Car (5)
        static_mask = scan.labels != 5
        points = scan.points[static_mask]
        labels = scan.labels[static_mask]
        
        if len(points) == 0:
            return
            
        # 2. Transform to Global Frame
        # P_global = T * P_local
        # Homogeneous coordinates
        points_h = np.hstack((points, np.ones((len(points), 1))))
        points_global = (scan.pose @ points_h.T).T[:, :3]
        
        # 3. Colorize
        colors = np.zeros((len(points), 3))
        for i, lbl in enumerate(labels):
            if lbl in COLOR_MAP:
                colors[i] = COLOR_MAP[lbl]
            else:
                colors[i] = [1, 1, 1] # White for unknown
                
        # 4. Create Open3D PointCloud
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points_global)
        new_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 5. Aggregate
        self.global_pcd += new_pcd
        
        # 6. Downsample (Voxel Grid Filter) to keep map size manageable
        self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=self.voxel_size)

    def visualize(self):
        o3d.visualization.draw_geometries([self.global_pcd], 
                                          window_name="Semantic Map",
                                          width=800, height=600)

def generate_synthetic_data():
    scans = []
    
    # Simulate a car driving straight along X axis
    for i in range(20):
        x_pose = i * 2.0
        pose = np.eye(4)
        pose[0, 3] = x_pose
        
        points = []
        labels = []
        
        # Ground (Road) - Plane at Z=0
        for x in np.arange(-5, 5, 0.5):
            for y in np.arange(-5, 5, 0.5):
                points.append([x, y, 0])
                labels.append(1) # Road
                
        # Building (Wall) - Plane at Y=6
        for x in np.arange(-5, 5, 0.5):
            for z in np.arange(0, 5, 0.5):
                points.append([x, 6, z])
                labels.append(3) # Building
                
        # Moving Car (Dynamic) - Box moving with ego
        # Relative position (3, 0)
        for x in np.arange(2, 4, 0.2):
            for y in np.arange(-1, 1, 0.2):
                for z in np.arange(0, 1.5, 0.2):
                    points.append([x, y, z])
                    labels.append(5) # Car
                    
        # Static Pole - At X=10, Y=-3
        # Relative X changes as we move
        pole_x_global = 10.0
        pole_x_local = pole_x_global - x_pose
        if -10 < pole_x_local < 10: # In view
            for z in np.arange(0, 4, 0.2):
                points.append([pole_x_local, -3, z])
                labels.append(6) # Pole
        
        scans.append(SemanticScan(np.array(points), np.array(labels), pose))
        
    return scans

def main():
    print("Generating Synthetic Semantic Data...")
    scans = generate_synthetic_data()
    
    mapper = SemanticMapper(voxel_size=0.1)
    
    print("Building Map...")
    for i, scan in enumerate(scans):
        mapper.process_scan(scan)
        print(f"Processed Scan {i+1}/{len(scans)}")
        
    print("Visualizing Map...")
    print("Legend: Gray=Road, Red=Building, Yellow=Pole. (Blue Car should be gone)")
    mapper.visualize()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Ghost Car

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   The **Road** (Gray) forms a continuous strip.
    -   The **Building** (Red) forms a continuous wall.
    -   The **Pole** (Yellow) appears at X=10.
    -   The **Car** (Blue) is **MISSING**.
    -   *Why?* Because we filtered `label == 5`.
3.  **Experiment:**
    -   Comment out the filtering line: `static_mask = scan.labels != 5`.
    -   **Result:** You will see a "smear" of blue points along the road. This is the "Ghost Car" effect. Since the car moves with the sensor (in this sim), it leaves a trail. In real life, moving cars leave trails if not removed.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Map Drift
**Symptom:** The wall looks double or blurry.
**Cause:** Poor Localization (Pose error).
**Solution:** Use SLAM (Scan Matching) to refine the pose before adding to the map. Semantic ICP can help (match buildings to buildings, not buildings to cars).

#### 2. Memory Explosion
**Symptom:** RAM full after a few minutes.
**Cause:** Point clouds are huge.
**Solution:** Use **Octomap** (Octree) or Voxel Hashing to store the map efficiently. Don't store raw points; store voxel centers.

---

## ⚡ Optimization & Best Practices

### 1. Ray Casting for Cleaning
Simply filtering dynamic objects isn't enough.
-   If a car drives away, the space behind it becomes free.
-   **Ray Casting:** For every "Free" ray, decrement the occupancy of voxels along the ray. This clears out "Ghost" points that might have been misclassified as static earlier.

### 2. Semantic ICP
Use semantics to improve localization.
-   When matching scan to map, only match "Road" points to "Road" map points.
-   Reduces drift significantly in dynamic environments.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is Semantic Mapping better than Geometric Mapping?
    *   **A:** It allows the planner to make smarter decisions (e.g., "Park on the shoulder", not "Park on the sidewalk").
2.  **Q:** How do we handle "Semi-Static" objects like parked cars?
    *   **A:** Usually, we map them. A separate "Live Perception" system detects if they move. Or we use a "Long-Term" map (background) and "Short-Term" map (foreground).
3.  **Q:** What is the output of a Semantic Segmentation network?
    *   **A:** A class probability vector for every pixel (camera) or point (LiDAR).

### Challenge Task
**Task:** Probabilistic Update.
1.  Instead of just adding points, maintain a probability vector for each voxel $[P_{road}, P_{building}, \dots]$.
2.  Update using Bayes Rule (Log Odds).
3.  This handles sensor noise (e.g., one scan says "Road", next says "Sidewalk").

---

## 📚 Further Reading & References
-   [SemanticKITTI Dataset](http://semantic-kitti.org/)
-   [RangeNet++ Paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf)

---

**Day 81 Complete** | Phase 4: ADAS & Robotics Systems | Week 12: HD Maps & Map Matching
