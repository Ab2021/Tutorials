# Day 153: Week 24 Review & Project (The DIY 3D Scanner)
## Phase 3: Camera Systems & ISP | Week 24: Camera Calibration & 3D Vision

---

## 🎯 Learning Objectives
1.  **Synthesize** the concepts of Calibration, Stereo, and Reconstruction.
2.  **Build** a "Turntable 3D Scanner" using a single camera and a rotating platform.
3.  **Implement** the software pipeline: Capture -> Calibrate -> SfM -> Mesh.
4.  **Validate** the accuracy of the scanner against a known object (e.g., a cube).
5.  **Prepare** for Week 47 (Machine Learning).

---

## 📚 Week 24 Recap

### Topics Covered

**Day 147: Calibration Fundamentals**
- Pinhole Model, Distortion, Checkerboards.

**Day 148: Implementing Calibration**
- `cv2.calibrateCamera`, Reprojection Error, Undistortion.

**Day 149: Stereo Vision**
- Epipolar Geometry, Rectification, Disparity ($Z = fB/d$).

**Day 150: Depth Estimation**
- SGBM Tuning, WLS Filter, Point Cloud Generation.

**Day 151: 3D Reconstruction**
- Downsampling, Outlier Removal, ICP, Poisson Meshing.

**Day 152: Structure from Motion**
- COLMAP, Sparse vs Dense, Bundle Adjustment.

---

## 💻 Week 24 Project: The Turntable 3D Scanner

### Objective
Create a low-cost 3D scanner using a webcam and a printed ArUco marker sheet.

### Hardware Setup
1.  **Camera:** Webcam on a tripod (fixed).
2.  **Turntable:** A "Lazy Susan" or a rotating display stand.
3.  **Target:** An object (e.g., a toy figure) placed on the turntable.
4.  **Markers:** Print a ring of ArUco markers and stick them to the *base* of the turntable.

### Software Pipeline

#### Step 1: Pose Estimation (The Trick)
Instead of using generic SfM (which is slow and scale-ambiguous), we use the ArUco markers on the turntable to calculate the camera pose relative to the object for every frame.
*   Since the object rotates and camera is fixed, it's mathematically equivalent to the camera rotating around the object.
*   We compute $T_{cam\_marker}$ for each frame.

#### Step 2: Space Carving (Voxel Hashing)
1.  Initialize a Voxel Grid around the center of the turntable.
2.  Mark all voxels as "Occupied".
3.  For each frame:
    *   Project every voxel center $(X, Y, Z)$ into the image $(u, v)$.
    *   Check the "Silhouette" (Mask) of the object.
    *   If a voxel projects to a "Background" pixel, carve it away (set to Empty).
4.  **Result:** The remaining voxels form the shape of the object (Visual Hull).

#### Step 3: Texturing
1.  For each surface voxel, find the best camera view (closest normal).
2.  Project the voxel to that image and read the RGB color.

### Implementation Snippet (Space Carving)

```python
import numpy as np
import cv2

def carve_voxels(voxels, camera_poses, masks, K):
    """
    voxels: Nx3 array of voxel centers
    camera_poses: List of 4x4 matrices (World -> Camera)
    masks: List of binary images (0=Background, 1=Object)
    K: 3x3 Intrinsic Matrix
    """
    occupied = np.ones(len(voxels), dtype=bool)
    
    for i, pose in enumerate(camera_poses):
        mask = masks[i]
        
        # Transform voxels to Camera Frame
        # P_cam = R * P_world + t
        R = pose[:3, :3]
        t = pose[:3, 3]
        voxels_cam = (R @ voxels.T).T + t
        
        # Project to Image
        # p = K * P_cam
        points_2d = (K @ voxels_cam.T).T
        u = points_2d[:, 0] / points_2d[:, 2]
        v = points_2d[:, 1] / points_2d[:, 2]
        
        # Check bounds
        h, w = mask.shape
        valid_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        # Check mask
        # If projected to background (0), set occupied = False
        # Note: Need integer coordinates
        u_int = u.astype(int)
        v_int = v.astype(int)
        
        # Vectorized check
        # Only check valid UVs
        is_background = mask[v_int[valid_uv], u_int[valid_uv]] == 0
        
        # Update occupied
        # If valid_uv is True AND is_background is True -> Carve
        occupied[valid_uv] &= ~is_background
        
    return voxels[occupied]
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Build the Rig

**Objective:** Physical setup.

**Steps:**
1.  Print the ArUco ring.
2.  Place object in center.
3.  Ensure lighting is diffuse (no hard shadows).
4.  Record a video of one full rotation.

### Lab 2: Background Subtraction

**Objective:** Get the masks.

**Steps:**
1.  Since the camera is static, take a photo of the *empty* turntable first (Background).
2.  For each frame, compute `diff = abs(frame - background)`.
3.  Threshold to get the Mask.
4.  **Challenge:** The ArUco markers are moving! You might need to mask them out or use a green screen.

### Lab 3: Run the Reconstruction

**Objective:** Get the mesh.

**Steps:**
1.  Run the Space Carving script.
2.  Export to `.ply`.
3.  Open in MeshLab.
4.  **Result:** A blocky voxel model of your object.
5.  **Refine:** Use Marching Cubes to smooth it.

---

## 📝 Assessment Questions

### Comprehensive Questions

1.  **Why is "Visual Hull" (Space Carving) faster than "Stereo Matching"?** (It relies on binary silhouettes, not texture matching. It's robust for textureless objects, but cannot capture concavities like a coffee mug handle hole if not seen).
2.  **How does the ArUco marker help with Scale?** (We know the physical size of the marker, e.g., 50mm. This fixes the scale of the entire reconstruction).
3.  **What is the "Bas-Relief Ambiguity"?** (In SfM, it's hard to distinguish between a deep object and a shallow object if the camera angle doesn't change enough).

### Practical Challenges

1.  **Scan a Shiny Object:** Try scanning a soda can.
    *   **Observation:** It fails. Reflections move.
    *   **Fix:** Spray it with matte primer or baby powder (if allowed).
2.  **Export to 3D Printer:** Convert the mesh to STL, slice it, and print a copy of your object!

---

## 📚 Resources & Next Steps

### Week 24 Summary

**Completed:**
- ✅ **Calibration:** The foundation.
- ✅ **Stereo:** Depth from two eyes.
- ✅ **SfM:** Depth from motion.
- ✅ **Project:** A working 3D scanner.

### Week 47 Preview (Machine Learning)

**Topics:**
- **AI on Edge:** Running Neural Networks on the Camera.
- **Object Detection:** YOLO.
- **Segmentation:** Understanding pixels.
- **Data:** Creating your own dataset.

---

**Day 153 Complete** | Phase 3: Camera Systems & ISP | Week 24: Camera Calibration & 3D Vision


