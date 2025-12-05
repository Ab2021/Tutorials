# Day 38: Multi-Robot SLAM (Map Merging)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 6: Multi-Robot Systems

---

> **📝 Content Creator Instructions:**
> Robot A maps the kitchen. Robot B maps the bedroom. How do they combine them?
> - **Focus:** Map Merging, Relative Pose Estimation, and Distributed Loop Closure.
> - **Code:** Aligning and stitching two Occupancy Grids using Feature Matching.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Understand** the Map Merging problem: Finding the transformation $T_{AB}$ that aligns Map A and Map B.
2.  **Implement** Feature-Based Alignment (ORB features on Grid Maps).
3.  **Perform** Grid Stitching using Probabilistic Fusion.
4.  **Execute** a multi-robot mapping session (Simulated) and visualize the global map.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install opencv-python numpy matplotlib scikit-image
```

### Prior Knowledge
- SLAM (Occupancy Grids).
- Rigid Body Transformations ($SE(2)$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Problem

Robot A has Map $M_A$ in frame $F_A$.
Robot B has Map $M_B$ in frame $F_B$.
Usually, $F_A \ne F_B$ (They started at different unrelated locations).
**Goal:** Find transform $T_{AB}$ such that $M_{Global} = M_A \cup (T_{AB} \cdot M_B)$.

### 🔹 Part 2: How to find $T_{AB}$?

1.  **Rendezvous:** Robots meet each other. Sensors detect relative pose. Highly accurate but requires coordination.
2.  **Map Matching:**
    *   Treat Occupancy Grid as an Image.
    *   Extract Features (corners, corridors) from $M_A$ and $M_B$.
    *   Match features -> RANSAC -> Compute $T_{AB}$.
    *   *Requirement:* Maps must overlap significantly.

### 🔹 Part 3: Distributed Loop Closure

In Single-Robot SLAM, "Loop Closure" happens when I see a place I've been before.
In Multi-Robot SLAM, **"Inter-Robot Loop Closure"** happens when *I* see a place *You* have been before.
*   Once detected, this "anchors" the two trajectories together.
*   Optimization (Pose Graph) runs over both robots' poses.

---

## 💻 Implementation: Map Stitching

We will take two partial maps (images) and merge them.

### 🛠️ Project Structure
```text
day38_multislam/
├── data/
│   ├── map_part1.png
│   ├── map_part2.png
├── src/
│   ├── map_merger.py
│   └── visualize.py
└── run_merge.py
```

### 👨‍💻 Code Implementation (`src/map_merger.py`)

```python
import cv2
import numpy as np

def merge_maps(map1_path, map2_path):
    # 1. Load Maps (GrayScale: 0=Free, 255=Unknown, 100=Obstacle)
    # Simplified: 0=Obstacle, 255=Free for CV2
    img1 = cv2.imread(map1_path, 0)
    img2 = cv2.imread(map2_path, 0)
    
    # 2. Extract ORB Features
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # 3. Match Features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Keep top 20 good matches
    good_matches = matches[:20]
    
    # 4. Find Homography (Transformation Matrix)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # We want T that moves Map2 to Map1
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    print("Found Transformation:\n", M)
    
    # 5. Warp Map2 to align with Map1
    h, w = img1.shape
    img2_aligned = cv2.warpPerspective(img2, M, (w, h))
    
    # 6. Fuse (Max Pooling logic for simplicity)
    # If Map1 has obstacle, result has obstacle.
    # Logic: Min(pixel) since 0=Obstacle usually in generated maps
    fused = np.minimum(img1, img2_aligned)
    
    return img1, img2, img2_aligned, fused, good_matches, kp1, kp2
```

### 👨‍💻 Visualization (`run_merge.py`)

```python
import matplotlib.pyplot as plt
from src.map_merger import merge_maps

m1, m2, m2_a, fused, matches, kp1, kp2 = merge_maps("data/map_part1.png", "data/map_part2.png")

# Draw Matches
img_matches = cv2.drawMatches(m1, kp1, m2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(img_matches)
plt.title("Constraint Matching")

plt.subplot(1, 3, 2)
plt.imshow(m2_a, cmap='gray')
plt.title("Map B Aligned")

plt.subplot(1, 3, 3)
plt.imshow(fused, cmap='gray')
plt.title("Global Fused Map")
plt.show()
```

---

## 🔬 Lab Exercise: The "Mystery Map"

### 1. Lab Objectives
- Given two map fragments that share a common corridor.
- Run the Merger.
- **Fail Case:** Maps have too little overlap (Matches are random noise).
- **Fix:** Robots must proactively explore boundaries (Frontier Exploration) to create overlap.

---

## 🚀 Project: "Collaborative Graph SLAM"

**Goal:** Use GTSAM (Factor Graph) for 2 Robots (Day 11).
1.  **Node:** `x_A_t` (Robot A poses), `x_B_t` (Robot B poses).
2.  **Factor:**
    *   Odometry factors (A->A, B->B).
    *   **Loop Closure Factor:** Between A and B. "I saw Robot B at relative pose $\Delta x$".
3.  **Result:** When the loop closes, both trajectories snap into the global frame.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Scale Drift"
*   **Cause:** Visual SLAM often loses scale ($x,y,z$ are arbitrary units). Merging Map A (Scale=1.0) with Map B (Scale=0.8) fails.
*   **Fix:** Use Stereo/Lidar (Metric Scale). Or use Sim(3) alignment during merging to solve for Scale $s$.

#### 2. "Bandwidth Hog"
*   **Cause:** Sending full grid maps (images) continuously.
*   **Fix:** Only send **Submaps** (small patches that changed). Or send Factor Graphs (compressed poses + features).

---

## ⚡ Optimization: Condensed Maps

Instead of grids, exchange **Topological Graphs**.
*   Node = Room Center. Edge = Corridor.
*   Map Matching = Graph Isomorphism (finding similar sub-graphs).
*   Data size: KB instead of MB.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Can we merge maps without overlap?
    *   **A:** No. Without overlap or direct robot-to-robot observation, the relative transform is unobservable.
2.  **Q:** What is "Map Consensus"?
    *   **A:** When all robots agree on the global map.
3.  **Q:** Why is RANSAC important here?
    *   **A:** Maps contain repetitive structures (corridors look alike). RANSAC filters out wrong matches (outliers) that would destroy the alignment.

### Challenge Task
> **Task:** 3-Robot Merge.
> 1. Merge A and B -> AB.
> 2. Merge AB and C -> ABC.
> 3. Does order matter? (A+B, then +C vs B+C then +A). Yes, error propagation varies.

---

## 📚 Further Reading
- **Multi-Robot SLAM:** Saeedi et al. (Survey).
- **GTSAM:** Multi-robot examples.

---

**Day 38 Complete**
