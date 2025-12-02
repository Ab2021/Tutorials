# Day 46: Stereo Vision Fundamentals
## Phase 3: Camera Systems & ISP | Week 8: Depth Sensing & 3D Vision

---

## 🎯 Learning Objectives
1.  **Understand** the geometry of Stereo Vision (Epipolar Geometry, Baseline, Focal Length).
2.  **Calculate** Depth from Disparity ($Z = f \cdot B / d$).
3.  **Perform** Stereo Calibration (Rectification) to align images row-by-row.
4.  **Compute** Disparity Maps using Block Matching (SAD/SSD) and Semi-Global Matching (SGM).
5.  **Analyze** the trade-offs between Baseline width and Depth Range.
6.  **Debug** common stereo issues (Textureless regions, Repetitive patterns).

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Stereo Camera (or two synchronized webcams).
*   **Software:** OpenCV (`calib3d` module).
*   **Knowledge:** Pinhole Camera Model, Intrinsic/Extrinsic Matrices.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Epipolar Geometry
*   **Concept:** A point in 3D space projects to $p_L$ in the Left image and $p_R$ in the Right image.
*   **Epipolar Constraint:** $p_R$ must lie on a specific line (Epipolar Line) in the Right image defined by $p_L$ and the camera geometry.
*   **Rectification:** Warping both images so that Epipolar Lines become horizontal scanlines. This simplifies the search for $p_R$ to a 1D search along the same row.

### 🔹 Part 2: Depth from Disparity
*   **Disparity ($d$):** The shift in pixels between the left and right projection ($d = x_L - x_R$).
*   **Formula:** $Z = \frac{f \cdot B}{d}$
    *   $Z$: Depth (Distance).
    *   $f$: Focal length (in pixels).
    *   $B$: Baseline (Distance between cameras in meters).
*   **Insight:**
    *   Large $d$ (Shift) -> Close object.
    *   Small $d$ (Shift) -> Far object.
    *   $d = 0$ -> Infinite depth.

### 🔹 Part 3: Matching Algorithms
*   **Block Matching (BM):** For each pixel in Left, search a window in Right (along the row) to find the best match (Lowest Sum of Absolute Differences - SAD). Fast but noisy.
*   **Semi-Global Matching (SGM):** Optimizes a global energy function (smoothness constraint) along multiple paths. Slower but much smoother and fills gaps.

---

## 💻 Implementation Examples

### Example 1: Stereo Calibration & Rectification

Before computing depth, we MUST rectify the images.

```cpp
/**
 * @brief Rectify Stereo Images
 * Assumes calibration matrices (M1, D1, M2, D2, R, T) are known.
 */
void rectify_stereo(cv::Mat& imgL, cv::Mat& imgR, cv::Mat& rectL, cv::Mat& rectR) {
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(M1, D1, M2, D2, imgL.size(), R, T, R1, R2, P1, P2, Q);
    
    cv::Mat map11, map12, map21, map22;
    cv::initUndistortRectifyMap(M1, D1, R1, P1, imgL.size(), CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(M2, D2, R2, P2, imgL.size(), CV_16SC2, map21, map22);
    
    cv::remap(imgL, rectL, map11, map12, cv::INTER_LINEAR);
    cv::remap(imgR, rectR, map21, map22, cv::INTER_LINEAR);
}
```

### Example 2: Computing Disparity (StereoBM)

```cpp
/**
 * @brief Compute Disparity Map
 */
void compute_disparity(cv::Mat& left, cv::Mat& right) {
    cv::Mat left_gray, right_gray;
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
    
    // 1. Create Matcher
    // numDisparities must be divisible by 16
    // blockSize (odd): 5-21
    int numDisparities = 16 * 5;
    int blockSize = 15;
    auto matcher = cv::StereoBM::create(numDisparities, blockSize);
    
    // 2. Compute
    cv::Mat disparity;
    matcher->compute(left_gray, right_gray, disparity);
    
    // 3. Visualize
    // Disparity is 16-bit fixed point (scaled by 16). Convert to 8-bit.
    cv::Mat disp8;
    disparity.convertTo(disp8, CV_8U, 255.0 / (numDisparities * 16.0));
    
    cv::imshow("Disparity", disp8);
}
```

### Example 3: Disparity to Depth (3D Point Cloud)

Using the `Q` matrix from `stereoRectify`.

```cpp
void reproject_to_3d(cv::Mat& disparity, cv::Mat& Q) {
    cv::Mat points3D;
    cv::reprojectImageTo3D(disparity, points3D, Q);
    
    // points3D contains (X, Y, Z) float coordinates for every pixel
    // Filter out infinite points (where disparity was 0 or -1)
}
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: The "Finger Test" (Parallax)

**Objective:** Intuitively understand disparity.

**Steps:**
1.  Hold your finger in front of your face.
2.  Close Left eye, Open Right.
3.  Close Right eye, Open Left.
4.  **Observation:** The finger "jumps" horizontally against the background.
5.  Move finger closer. The jump is larger.
6.  Move finger away. The jump is smaller.

### Lab 2: Tuning Block Size

**Objective:** Trade-off between detail and noise.

**Steps:**
1.  Run StereoBM with `blockSize = 5`.
    *   Result: High detail, lots of speckle noise (mismatches).
2.  Run StereoBM with `blockSize = 21`.
    *   Result: Smooth, low noise, but fine details (thin objects) are lost/bloated.

### Lab 3: The Textureless Wall Problem

**Objective:** Understand failure modes.

**Steps:**
1.  Point stereo camera at a plain white wall.
2.  Compute Disparity.
3.  **Result:** Garbage or holes.
4.  **Reason:** Block matching cannot find a unique match if all pixels look the same.
5.  **Fix:** Project a pattern (Active Stereo) or use SGM (smoothness constraint helps fill gaps).

---

## 🐛 Debugging Techniques

### Debug 1: Vertical Shift

**Symptom:** Disparity map looks completely wrong/random.

**Cause:**
*   Images are not Rectified. Scanlines don't align.
*   Check by drawing horizontal lines on both images. Features MUST be on the same line.
*   **Fix:** Re-calibrate.

### Debug 2: Min/Max Disparity Range

**Symptom:** Close objects are missing (black).

**Cause:**
*   `numDisparities` is too small. The object is too close, so the shift > max search range.
*   **Fix:** Increase `numDisparities`. Note: This increases computation time linearly.

---

## ⚡ Performance Optimization

### Optimization 1: Downscaling

*   Stereo matching is $O(W \cdot H \cdot D)$.
*   Halving the resolution (1080p -> 540p) reduces computation by 4x (pixels) * 2x (disparity range) = 8x!
*   **Strategy:** Compute disparity at low res, then upsample (with edge-aware filter).

### Optimization 2: GPU Acceleration

*   `cv::cuda::StereoBM` and `cv::cuda::StereoBeliefPropagation`.
*   Real-time 60fps stereo is possible on Jetson Nano using CUDA.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why does the Baseline ($B$) affect the maximum measurable depth?**
2.  **What happens to depth resolution ($dZ$) as distance ($Z$) increases?** (Hint: Error grows quadratically $Z^2$).
3.  **Why is Rectification necessary?**
4.  **What is the difference between Passive and Active Stereo?**

### Practical Challenges

1.  **Implement a "Virtual Tape Measure":** Click on two points in the Left image. Calculate the Euclidean distance between them in 3D space.
2.  **Build a "Collision Warning":** If the average depth in the center region is < 1 meter, sound an alarm.

---

## 📚 Further Reading & Resources

### Papers
*   **"Stereo Processing by Semiglobal Matching and Mutual Information"** - Heiko Hirschmuller (The SGM paper).

### Tools
*   **ROS (Robot Operating System):** `stereo_image_proc` package.

---

## 🎓 Summary

Today we covered:
- ✅ **Geometry:** Epipolar lines and Rectification.
- ✅ **Math:** $Z = fB/d$.
- ✅ **Algorithms:** Block Matching vs SGM.
- ✅ **Calibration:** The key to success.
- ✅ **Limitations:** Textureless surfaces and Range limits.

**Next:** Day 47 - 3D Reconstruction (Point Clouds & Mesh).

---

**Day 46 Complete** | Phase 3: Camera Systems & ISP | Week 8: Depth Sensing & 3D Vision
