# Day 49: Structure from Motion (SfM) & Photogrammetry
## Phase 3: Camera Systems & ISP | Week 8: Depth Sensing & 3D Vision

---

## 🎯 Learning Objectives
1.  **Understand** the difference between SLAM (Real-time) and SfM (Offline, High Precision).
2.  **Analyze** the SfM pipeline: Feature Extraction -> Matching -> Sparse Reconstruction -> Dense Reconstruction.
3.  **Use** COLMAP (State-of-the-art SfM) to reconstruct a 3D model from a set of photos.
4.  **Implement** Multi-View Stereo (MVS) concepts.
5.  **Debug** reconstruction failures (symmetries, lack of overlap).
6.  **Export** models for 3D printing or Game Engines.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera (Smartphone is fine).
*   **Software:** COLMAP (GUI/CLI), MeshLab.
*   **Knowledge:** Bundle Adjustment (Day 48).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: SfM vs SLAM
*   **SLAM:** Prioritizes speed. "Where am I now?". Uses sequential frames.
*   **SfM:** Prioritizes accuracy. "What is the structure of the scene?". Uses unordered collections of images (e.g., from Flickr or a drone survey).
*   **Global Bundle Adjustment:** SfM optimizes *all* cameras and *all* points simultaneously, which is computationally expensive but yields sub-pixel accuracy.

### 🔹 Part 2: The Pipeline
1.  **Feature Extraction:** SIFT (Scale-Invariant Feature Transform) is preferred over ORB for accuracy.
2.  **Matching:** Find matches between *all pairs* of images.
3.  **Geometric Verification:** RANSAC to find Fundamental Matrices.
4.  **Incremental Reconstruction:**
    *   Start with a good pair (Seed).
    *   Triangulate points.
    *   Register new images (PnP).
    *   Bundle Adjust.
    *   Repeat.
5.  **Dense Reconstruction (MVS):** Once camera poses are known, compute depth maps for every pixel and fuse them.

### 🔹 Part 3: Photogrammetry
*   **Science of making measurements from photographs.**
*   **Applications:** Surveying, 3D Asset Creation, Cultural Heritage preservation.
*   **Constraints:** High overlap (80%), diffuse lighting (no shadows), static scene.

---

## 💻 Implementation Examples

### Example 1: Running COLMAP (CLI)

Automated reconstruction pipeline.

```bash
# 1. Feature Extraction
colmap feature_extractor \
   --database_path database.db \
   --image_path images/

# 2. Exhaustive Matching
colmap exhaustive_matcher \
   --database_path database.db

# 3. Sparse Reconstruction
mkdir sparse
colmap mapper \
    --database_path database.db \
    --image_path images/ \
    --output_path sparse

# 4. Undistort Images (Prepare for MVS)
mkdir dense
colmap image_undistorter \
    --image_path images \
    --input_path sparse/0 \
    --output_path dense \
    --output_type COLMAP \
    --max_image_size 2000

# 5. Dense Reconstruction (PatchMatch Stereo)
colmap patch_match_stereo \
    --workspace_path dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

# 6. Fusion (Point Cloud to Mesh)
colmap stereo_fusion \
    --workspace_path dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path dense/fused.ply
```

### Example 2: Poisson Surface Reconstruction (MeshLab)

After getting the point cloud (`fused.ply`), we need a mesh.
1.  **Import** PLY into MeshLab.
2.  **Compute Normals:** Filters -> Point Set -> Compute normals for point sets.
3.  **Reconstruct:** Filters -> Remeshing -> Screened Poisson Surface Reconstruction.
    *   **Reconstruction Depth:** 10 (Higher = More Detail).
4.  **Export:** Save as `.obj` or `.stl`.

### Example 3: Incremental SfM Logic (Pseudo-Code)

```cpp
void run_incremental_sfm() {
    // 1. Find Initial Pair
    Pair seed = find_best_pair(matches);
    reconstruct(seed);
    
    while (!remaining_images.empty()) {
        // 2. Select Next Image
        // Pick image that sees the most existing 3D points
        Image next = select_best_view(reconstructed_points);
        
        // 3. Register (PnP)
        Pose pose = solve_pnp(next, reconstructed_points);
        
        // 4. Triangulate New Points
        add_new_points(next, pose);
        
        // 5. Bundle Adjustment
        // Optimize ALL poses and points
        run_global_bundle_adjustment();
        
        // 6. Filter Outliers
        remove_large_reprojection_errors();
    }
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: The "Statue" Scan

**Objective:** Reconstruct a 3D object.

**Steps:**
1.  Place an object (e.g., a shoe or toy) on a textured surface.
2.  Take 30-50 photos moving in a circle around it (360 degrees).
    *   Ensure good overlap.
    *   Do not move the object.
3.  Run COLMAP pipeline.
4.  **Result:** A dense point cloud of the object.

### Lab 2: Drone Mapping (Aerial)

**Objective:** Create a map from top-down photos.

**Steps:**
1.  Use a drone (or simulate with Google Earth screenshots).
2.  Take a grid of nadir (down-looking) photos.
3.  Run COLMAP.
4.  **Result:** A Digital Elevation Model (DEM) of the terrain.

### Lab 3: Scale Ambiguity

**Objective:** Measure the real size.

**Steps:**
1.  Reconstruct the object from Lab 1.
2.  Measure the length in MeshLab (e.g., 5.2 units).
3.  Measure the real object (e.g., 26 cm).
4.  **Scale Factor:** $26 / 5.2 = 5$.
5.  Scale the mesh by 5 to get real-world dimensions.

---

## 🐛 Debugging Techniques

### Debug 1: "Bowl Effect" (Drift)

**Symptom:** Flat ground curves up like a bowl.

**Cause:**
*   Radial distortion not modeled correctly.
*   Accumulated drift in long strips of photos.
*   **Fix:** Use "Self-Calibration" in Bundle Adjustment to refine lens parameters. Use GPS priors if available.

### Debug 2: Failed Registration

**Symptom:** Multiple disconnected models (Split reconstruction).

**Cause:**
*   Not enough overlap between image sequences.
*   Symmetric object (building looks same from front and back).
*   **Fix:** Add "Bridge" photos connecting the disconnected areas.

---

## ⚡ Performance Optimization

### Optimization 1: Vocabulary Tree Matching

*   Instead of matching *all* pairs ($N^2$), use a Vocabulary Tree (Bag of Words) to find top K candidates for each image.
*   Reduces complexity to $O(N \cdot K)$. Essential for > 1000 images.

### Optimization 2: GPU SIFT

*   Feature extraction is the bottleneck.
*   Use SiftGPU (CUDA) implementation in COLMAP.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is SIFT better than Harris for SfM?** (Scale and Rotation invariance).
2.  **What is the "Fundamental Matrix"?**
3.  **Why does SfM fail on shiny/reflective objects?**
4.  **What is "Multi-View Stereo" (MVS)?**

### Practical Challenges

1.  **Create a "Turntable" setup:** Instead of moving the camera, rotate the object. Note: You must mask out the background, or SfM will think the camera is stationary and the object is exploding.
2.  **Implement "Scale Bars":** Place a ruler in the scene. Automatically detect the ruler ticks to set the scale of the reconstruction.

---

## 📚 Further Reading & Resources

### Software
*   **COLMAP:** The gold standard open-source SfM.
*   **OpenMVS:** Excellent for dense reconstruction and texturing.
*   **Meshroom:** Node-based GUI for photogrammetry (uses AliceVision).

---

## 🎓 Summary

Today we covered:
- ✅ **SfM:** Structure from Motion.
- ✅ **Photogrammetry:** Creating 3D models from 2D photos.
- ✅ **COLMAP:** The pipeline (Sparse -> Dense).
- ✅ **MVS:** Dense depth estimation.
- ✅ **Meshing:** Creating surfaces.

**Next:** Day 50 - Week 8 Review & 3D Scanning Project.

---

**Day 49 Complete** | Phase 3: Camera Systems & ISP | Week 8: Depth Sensing & 3D Vision
