# Day 152: Structure from Motion (SfM)
## Phase 3: Camera Systems & ISP | Week 24: Camera Calibration & 3D Vision

---

## 🎯 Learning Objectives
1.  **Differentiate** between Stereo Vision (Simultaneous) and SfM (Sequential).
2.  **Understand** the SfM Pipeline: Feature Extraction -> Matching -> Bundle Adjustment.
3.  **Use** COLMAP (The Gold Standard Open Source SfM tool) to reconstruct a scene from a video.
4.  **Visualize** Sparse vs Dense Reconstruction.
5.  **Export** the camera poses ($R, t$ for every frame).

---

## 📚 Prerequisites & Preparation
*   **Software:** COLMAP (GUI or CLI), MeshLab.
*   **Data:** A video of a static object (walk around it).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Concept
*   **Stereo:** Fixed baseline, known calibration.
*   **SfM:** Moving camera, unknown baseline, unknown calibration (sometimes).
*   **Logic:** If I see a point $X$ move 10 pixels to the left, I must have moved to the right. By tracking thousands of points, I can solve for *both* the 3D structure of the scene ($X, Y, Z$) and the motion of the camera ($R, t$).

### 🔹 Part 2: The Pipeline
1.  **Feature Extraction:** SIFT (Scale-Invariant Feature Transform). Finds keypoints.
2.  **Feature Matching:** Finds the same keypoints in adjacent frames.
3.  **Geometric Verification:** RANSAC with Fundamental Matrix ($F$). Removes outliers.
4.  **Incremental Reconstruction:**
    *   Start with two frames (Initialization).
    *   Triangulate points.
    *   Add next frame (PnP).
    *   **Bundle Adjustment:** Non-linear optimization to minimize reprojection error across *all* cameras and *all* points simultaneously.

---

## 💻 Implementation Examples

### Example 1: Running COLMAP (CLI)

Automating the process.

```bash
# 1. Feature Extraction
colmap feature_extractor \
   --database_path database.db \
   --image_path images/

# 2. Exhaustive Matching (or Sequential for video)
colmap exhaustive_matcher \
   --database_path database.db

# 3. Mapper (Sparse Reconstruction)
mkdir sparse
colmap mapper \
    --database_path database.db \
    --image_path images/ \
    --output_path sparse/

# 4. Image Undistortion (Prepare for Dense)
mkdir dense
colmap image_undistorter \
    --image_path images/ \
    --input_path sparse/0 \
    --output_path dense/ \
    --output_type COLMAP \
    --max_image_size 2000

# 5. Patch Match Stereo (Dense Reconstruction)
colmap patch_match_stereo \
    --workspace_path dense/ \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

# 6. Stereo Fusion (Create PLY)
colmap stereo_fusion \
    --workspace_path dense/ \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path dense/fused.ply
```

### Example 2: Reading COLMAP Output (Python)

Extracting Camera Poses.

```python
import sqlite3
import numpy as np

# COLMAP stores data in a database or binary files
# Here we read the database for matches
db = sqlite3.connect("database.db")
cursor = db.cursor()

cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = 1;")
row = cursor.fetchone()
# Parse blob...

# Better: Use the provided `read_model.py` from COLMAP repo
from read_write_model import read_model

cameras, images, points3D = read_model(path="sparse/0", ext=".bin")

for image_id, image in images.items():
    print(f"Image {image.name}:")
    print(f"  Rotation (Quaternion): {image.qvec}")
    print(f"  Translation: {image.tvec}")
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: The "Statue" Scan

**Objective:** Reconstruct a static object.

**Steps:**
1.  Find a statue or a rock.
2.  Record a video walking in a circle around it (360 degrees).
3.  Extract frames: `ffmpeg -i video.mp4 -vf fps=2 images/%04d.jpg`.
4.  Run COLMAP (GUI is easier for first time).
5.  **Result:** You should see a ring of red camera frustums and a sparse point cloud of the object.

### Lab 2: Dense Reconstruction

**Objective:** Get a solid model.

**Steps:**
1.  Continue from Lab 1.
2.  Run "Multi-View Stereo" (Dense Reconstruction).
3.  **Result:** `fused.ply`.
4.  Open in MeshLab.
5.  **Observation:** It looks like a real 3D model, but with some noise.

### Lab 3: Export to Blender

**Objective:** Use the asset.

**Steps:**
1.  Import `.ply` into Blender.
2.  Add a texture (Vertex Colors).
3.  Render it.

---

## 🐛 Debugging SfM

### Debug 1: "Reconstruction Failed"

**Symptom:** COLMAP says "Could not initialize reconstruction".

**Cause:**
*   Not enough parallax (camera didn't move enough).
*   Pure rotation (panoramic tripod). SfM needs translation to triangulate depth!
*   **Fix:** Move the camera sideways (crab walk), don't just rotate it.

### Debug 2: "Banana" Effect

**Symptom:** A straight corridor looks curved.

**Cause:**
*   Drift. Errors accumulate over time.
*   **Fix:** Loop Closure. Return to the start position so the algorithm detects the same features and "snaps" the loop shut.

---

## ⚡ Performance Optimization

### Optimization 1: Vocabulary Tree

*   For large datasets (10,000 images), Exhaustive Matching is $O(N^2)$.
*   Use a **Vocabulary Tree** (Bag of Words) to only match images that look similar.
*   Reduces matching time from days to hours.

### Optimization 2: GPU Acceleration

*   COLMAP uses CUDA for SIFT extraction and Dense Stereo.
*   Ensure you have NVIDIA drivers installed.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why does SfM fail on a shiny car?** (Reflections move differently than the surface. Features are not stable).
2.  **What is "Bundle Adjustment"?** (Refining the 3D points and Camera Poses to minimize the reprojection error).
3.  **Difference between Sparse and Dense SfM?** (Sparse = Keypoints only. Dense = Every pixel).

### Practical Challenges

1.  **Drone Mapping:** Use drone footage (top-down) to reconstruct a building roof.
2.  **Scale Problem:** SfM is "Scale Ambiguous". The model might be 1 unit wide. Is that 1 meter or 1 km?
    *   **Task:** Place a ruler in the scene. Measure it in the 3D model. Calculate the Scale Factor.

---

## 📚 Further Reading & Resources

### Documentation
*   **COLMAP Documentation (Excellent tutorial).**
*   **"Visual SfM" (Older tool, but good concepts).**

---

## 🎓 Summary

Today we covered:
- ✅ **SfM:** Structure from Motion.
- ✅ **COLMAP:** The tool of choice.
- ✅ **Pipeline:** Feature -> Match -> Sparse -> Dense.
- ✅ **Drift:** The enemy of long sequences.
- ✅ **Scale:** The missing variable.

**Next:** Day 153 - Week 24 Review & Project (3D Scanner).

---

**Day 152 Complete** | Phase 3: Camera Systems & ISP | Week 24: Camera Calibration & 3D Vision


