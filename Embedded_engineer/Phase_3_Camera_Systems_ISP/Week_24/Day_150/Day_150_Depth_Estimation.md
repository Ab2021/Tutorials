# Day 150: Depth Estimation & Disparity Tuning
## Phase 3: Camera Systems & ISP | Week 24: Camera Calibration & 3D Vision

---

## 🎯 Learning Objectives
1.  **Tune** StereoSGBM parameters (P1, P2, Uniqueness Ratio) for optimal quality.
2.  **Filter** the Disparity Map using WLS (Weighted Least Squares) Filter.
3.  **Handle** Occlusions (Left-Right Consistency Check).
4.  **Generate** a Point Cloud (XYZ) from the Disparity Map.
5.  **Visualize** the 3D Point Cloud in MeshLab or Open3D.

---

## 📚 Prerequisites & Preparation
*   **Software:** Python, OpenCV (`pip install opencv-contrib-python`), Open3D (`pip install open3d`).
*   **Data:** Rectified Stereo Pair (from Day 149).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: SGBM Tuning Guide
*   **minDisparity:** Usually 0. If cameras are verged (toed-in), can be negative.
*   **numDisparities:** Search range. Must be divisible by 16.
*   **blockSize:** Matching window size (3-11).
*   **P1:** Penalty for disparity change of 1 pixel (Smoothness). $8 \times Channels \times blockSize^2$.
*   **P2:** Penalty for disparity change > 1 pixel (Discontinuity). $32 \times Channels \times blockSize^2$.
*   **disp12MaxDiff:** Max allowed difference in Left-Right check.
*   **uniquenessRatio:** Margin by which the best match must beat the second best.
*   **speckleWindowSize:** Max size of smooth disparity regions.
*   **speckleRange:** Max disparity variation within each connected component.

### 🔹 Part 2: Disparity Filtering (WLS)
*   Raw SGBM output is noisy and has gaps.
*   **WLS Filter:** Uses the original image as a guide to smooth the disparity map while preserving edges (Edge-Preserving Smoothing).
*   Fills holes caused by occlusions.

---

## 💻 Implementation Examples

### Example 1: Advanced SGBM + WLS Filter

```python
import cv2
import numpy as np

# Load Images
imgL = cv2.imread('left.png')
imgR = cv2.imread('right.png')

# 1. Setup SGBM
window_size = 3
min_disp = 0
num_disp = 16 * 10 # 160

left_matcher = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# 2. Setup WLS Filter
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.5)

# 3. Compute
displ = left_matcher.compute(imgL, imgR)
dispr = right_matcher.compute(imgR, imgL) # Need Right-to-Left for check

displ = np.int16(displ)
dispr = np.int16(dispr)

filtered_disp = wls_filter.filter(displ, imgL, disparity_map_right=dispr)

# 4. Normalize
filtered_disp_vis = cv2.normalize(src=filtered_disp, dst=filtered_disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filtered_disp_vis = np.uint8(filtered_disp_vis)

cv2.imshow("Filtered Disparity", filtered_disp_vis)
cv2.waitKey(0)
```

### Example 2: Reproject to 3D

Generating the Point Cloud.

```python
# Q Matrix comes from stereoRectify
# Q = [[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, f], [0, 0, -1/Tx, (cx-cx')/Tx]]

points = cv2.reprojectImageTo3D(filtered_disp, Q)
colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

# Mask out invalid points (infinite distance)
mask = filtered_disp > filtered_disp.min()
out_points = points[mask]
out_colors = colors[mask]

# Save as PLY
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

write_ply('out.ply', out_points, out_colors)
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Parameter Tuning GUI

**Objective:** Find the best settings interactively.

**Steps:**
1.  Create a Python script with `cv2.createTrackbar`.
2.  Bind trackbars to `numDisparities`, `blockSize`, `uniquenessRatio`.
3.  In the loop, update SGBM and show the result.
4.  **Task:** Tune until the "holes" in the disparity map disappear but edges remain sharp.

### Lab 2: Occlusion Handling

**Objective:** Understand the "Shadow".

**Steps:**
1.  Place an object in front of a background.
2.  Compute Disparity.
3.  **Observe:** There is a black shadow to the *left* of the object in the disparity map.
4.  **Why?** The Left camera sees the background, but the Right camera is blocked by the object. No match possible.
5.  **WLS Filter:** Should fill this shadow with the background depth.

### Lab 3: Open3D Visualization

**Objective:** Spin the world.

**Steps:**
1.  Load the saved `.ply` file in Open3D.
2.  `pcd = o3d.io.read_point_cloud("out.ply")`
3.  `o3d.visualization.draw_geometries([pcd])`
4.  **Verify:** Does the 3D structure look flat or distorted? If distorted, $Q$ matrix (Calibration) is wrong.

---

## 🐛 Debugging Depth Maps

### Debug 1: "Staircase" Effect

**Symptom:** Flat surfaces look like stairs.

**Cause:**
*   Disparity resolution is too low (integer pixels).
*   **Fix:** Use `cv2.StereoSGBM` (which has 1/16 pixel resolution) and ensure `Q` matrix handles the 16x scaling correctly.

### Debug 2: Wrong Scale

**Symptom:** Object looks 10m away but is actually 1m.

**Cause:**
*   Baseline ($T_x$) in $Q$ matrix is wrong.
*   Focal Length ($f$) is wrong.
*   **Fix:** Measure Baseline precisely (center to center).

---

## ⚡ Performance Optimization

### Optimization 1: Downsampling

*   Compute disparity at half resolution (width/2, height/2).
*   4x faster.
*   Upsample the result (guided filter) to full resolution.

### Optimization 2: ROI Processing

*   If you only care about the road (bottom half), crop the image before SGBM.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is the "Uniqueness Ratio"?** (Ensures the best match is significantly better than the 2nd best match to avoid ambiguity).
2.  **Why does SGBM need `P1` and `P2`?** (To enforce smoothness constraints. Depth usually doesn't jump randomly unless there is an edge).
3.  **What is the output format of `reprojectImageTo3D`?** (A 3-channel float image where each pixel contains $(X, Y, Z)$ coordinates).

### Practical Challenges

1.  **Measure Volume:** Use the Point Cloud to estimate the volume of a box. (Bounding Box dimensions).
2.  **Obstacle Warning:** If any point in the center region has $Z < 1.0m$, print "STOP".

---

## 📚 Further Reading & Resources

### Documentation
*   **OpenCV `ximgproc` (Extended Image Processing).**
*   **Open3D Tutorial.**

---

## 🎓 Summary

Today we covered:
- ✅ **SGBM Tuning:** The art of parameters.
- ✅ **WLS Filter:** Smoothing the noise.
- ✅ **Point Cloud:** 2D to 3D.
- ✅ **Visualization:** PLY files.
- ✅ **Occlusion:** The missing data.

**Next:** Day 151 - 3D Reconstruction (Point Clouds).

---

**Day 150 Complete** | Phase 3: Camera Systems & ISP | Week 24: Camera Calibration & 3D Vision


