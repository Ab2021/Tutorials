# Day 149: Stereo Vision Fundamentals (Epipolar Geometry)
## Phase 3: Camera Systems & ISP | Week 24: Camera Calibration & 3D Vision

---

## 🎯 Learning Objectives
1.  **Understand** how two eyes (cameras) perceive depth.
2.  **Define** Epipolar Geometry: Epipoles, Epipolar Lines, and Epipolar Plane.
3.  **Calculate** Depth from Disparity: $Z = \frac{f \times B}{d}$.
4.  **Perform** Stereo Rectification to align epipolar lines horizontally.
5.  **Visualize** Disparity Maps using OpenCV.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Stereo Camera (or two webcams mounted rigidly).
*   **Software:** Python, OpenCV.
*   **Concept:** Parallax. Hold your finger up and close one eye, then the other. The finger moves. Background doesn't.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Geometry of Depth
*   **Baseline ($B$):** Distance between the two camera centers.
*   **Focal Length ($f$):** Distance from lens to sensor.
*   **Disparity ($d$):** The difference in pixel position of the same object in Left vs Right image ($x_L - x_R$).
*   **Depth ($Z$):** Distance from the camera.
*   **Formula:**
    $$ \frac{Z}{f} = \frac{B}{d} \implies Z = \frac{f \cdot B}{d} $$
*   **Insight:**
    *   Large $d$ (Big shift) = Close object.
    *   Small $d$ (Small shift) = Far object.
    *   $d = 0$ = Infinite distance.

### 🔹 Part 2: Epipolar Geometry
*   **Problem:** To find a matching point in the Right image, do we search the *entire* image? No.
*   **Epipolar Constraint:** The point in the Right image *must* lie on a specific line called the **Epipolar Line**.
*   **Rectification:** We warp both images so that the Epipolar Lines become horizontal scanlines.
    *   Now, for a pixel $(x, y)$ in Left image, we only search row $y$ in the Right image.
    *   This makes stereo matching fast ($O(W \times H \times Range)$).

---

## 💻 Implementation Examples

### Example 1: Stereo Rectification (Concept)

Assuming we have calibrated both cameras ($K_1, D_1, K_2, D_2$) and the relation between them ($R, T$).

```python
# 1. Stereo Rectify
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T)

# 2. Compute Maps
map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)

# 3. Remap (Apply Rectification)
rectified_left = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectified_right = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

# 4. Check
# Concatenate images side-by-side. Draw horizontal lines.
# Features should align perfectly on the Y-axis.
```

### Example 2: Computing Disparity (Block Matching)

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

imgL = cv2.imread('tsukuba_l.png', 0)
imgR = cv2.imread('tsukuba_r.png', 0)

# Create StereoBM object
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Compute Disparity
disparity = stereo.compute(imgL, imgR)

# Normalize for visualization
norm_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

plt.imshow(norm_disparity, 'gray')
plt.show()
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Verify Rectification

**Objective:** Ensure row alignment.

**Steps:**
1.  Load your stereo pair images.
2.  Undistort and Rectify them (using calibration data).
3.  Combine them: `vis = np.hstack((rect_L, rect_R))`.
4.  Draw horizontal lines every 50 pixels.
5.  **Check:** Does the nose of the person in Left image lie on the *same line* as the nose in Right image?
6.  **Fail:** If there is a Y-shift, stereo matching will fail.

### Lab 2: Tuning Block Matching

**Objective:** Get a clean depth map.

**Parameters:**
*   `numDisparities`: Range of search. Must be divisible by 16. Larger = closer objects detected, but slower.
*   `blockSize`: Window size (e.g., 5, 15, 21). Larger = smoother but less detail.
*   **Task:** Try `blockSize=5` (noisy) vs `blockSize=21` (blobby). Find the sweet spot.

### Lab 3: Calculate Real Depth

**Objective:** Pixels to Meters.

**Steps:**
1.  Pick a point in the disparity map. Read value $d$ (pixels).
2.  Get $f$ (pixels) from $P1$ matrix (after rectification).
3.  Get $B$ (meters) from your physical setup (e.g., 0.1m).
4.  Calculate $Z = (f \times B) / d$.
5.  **Verify:** Measure with a tape measure.

---

## 🐛 Debugging Stereo Vision

### Debug 1: Noisy Disparity Map (Speckles)

**Symptom:** Random white/black dots in textureless areas (white wall).

**Cause:**
*   Block matching fails when there is no texture to match.
*   **Fix:** Use SGBM (Semi-Global Block Matching) which enforces smoothness. Or project a pattern (Active Stereo).

### Debug 2: "Stripes" in Disparity

**Symptom:** Discrete steps in depth.

**Cause:**
*   Sub-pixel interpolation is off.
*   Low resolution.
*   **Fix:** Use `cv2.StereoSGBM` with sub-pixel enabled.

---

## ⚡ Performance Optimization

### Optimization 1: SGBM (Semi-Global Matching)

*   `cv2.StereoSGBM` is slower than `StereoBM` but much better quality.
*   It optimizes a global energy function along several paths.
*   **Tuning:** `P1` and `P2` penalties control smoothness.

### Optimization 2: GPU Stereo

*   NVIDIA VPI (Vision Programming Interface) or CUDA Stereo.
*   Can run at > 100 FPS.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why does Stereo Vision fail on a white wall?** (No features to match. The correspondence problem is ambiguous).
2.  **What is the "Minimum Depth" of a stereo camera?** (Determined by the Baseline and Max Disparity search range. $Z_{min} = f \times B / d_{max}$).
3.  **Why do we need Rectification?** (To reduce the 2D search problem to a 1D search problem).

### Practical Challenges

1.  **Build a Stereo Rig:** Tape two webcams to a ruler. Capture images. Calibrate. Compute Depth.
2.  **Anaglyph:** Create a Red-Cyan 3D image from your stereo pair. `Result = Left_Red + Right_Cyan`.

---

## 📚 Further Reading & Resources

### Documentation
*   **OpenCV Stereo Camera Tutorial.**
*   **Middlebury Stereo Dataset (The benchmark).**

---

## 🎓 Summary

Today we covered:
- ✅ **Geometry:** $Z = fB/d$.
- ✅ **Rectification:** Aligning the eyes.
- ✅ **Matching:** BM vs SGBM.
- ✅ **Disparity:** The inverse of depth.
- ✅ **Texture:** The fuel for stereo.

**Next:** Day 150 - Depth Estimation & Disparity Tuning.

---

**Day 149 Complete** | Phase 3: Camera Systems & ISP | Week 24: Camera Calibration & 3D Vision


